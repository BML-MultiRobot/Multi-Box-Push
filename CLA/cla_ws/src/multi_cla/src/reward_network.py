# ! /usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class RewardNetwork(nn.Module):
    def __init__(self, neurons, num_actions, num_bots, params):
        super(RewardNetwork, self).__init__()
        self.layers, self.neurons = [], neurons
        self.n_layers = len(self.neurons) - 2
        self.num_actions = num_actions
        self.num_bots = num_bots
        self.output = None
        self.lr = params['reward_lr']
        self.noise_std = params['noise_std']

        self.reward_buff = params['reward_weight']
        self.counterfactual = params['counterfactual']

        self.createFeatures()
        self.optimizer = optim.Adam(super(RewardNetwork, self).parameters(), lr=self.lr)
        return

    def createFeatures(self):
        for i, (fan_in, fan_out) in enumerate(zip(self.neurons[:-2], self.neurons[1:-1])):
            layer = nn.Linear(fan_in, fan_out)
            torch.nn.init.uniform_(layer.weight, -1. / np.sqrt(fan_in), 1. / np.sqrt(fan_in))
            torch.nn.init.uniform_(layer.bias, -1. / np.sqrt(fan_in), 1. / np.sqrt(fan_in))
            exec('self.fc{} = layer'.format(i + 1))

        layer = nn.Linear(self.neurons[-2], self.neurons[-1])
        torch.nn.init.uniform_(layer.weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(layer.bias, -3e-3, 3e-3)
        self.output = layer

    def forward_from_numpy_all_actions(self, s, a, robot_ids):
        """ s: n x d array of global states
            a: n x r array of global action indices
            robot_ids: n x 1 array of the agent id in question

            Input to the neural network: all the states, filter action without index robot_id, robot_id
            Output: rewards for each of the actions robot_id can take given current state and other robot actions """
        output = None
        a = a.copy()

        curr_s = torch.from_numpy(s) if isinstance(s, np.ndarray) else s
        mask = np.zeros((a.shape[0], a.shape[-1]))
        mask[range(mask.shape[0]), robot_ids] = 1
        mask = np.repeat(np.expand_dims(mask, axis=1), repeats=a.shape[1], axis=1) if mask.shape != a.shape else mask
        for i in range(self.num_actions):
            a[mask > .5] = i
            curr_a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
            curr_a = (curr_a - (self.num_actions / 2.0)) / (self.num_actions / 2.0)
            x = torch.cat((curr_s.float(), curr_a.float()), dim=curr_s.dim()-1)
            curr_out = self.forward(x)
            if i == 0:
                output = curr_out
            else:
                output = torch.cat((output, curr_out), dim=curr_s.dim()-1)
        return output

    def forward(self, x):
        for i in range(self.n_layers):
            x = eval(('F.relu' + '(self.fc{}(x))').format(i + 1))
        return self.output(x)

    def delete_indices(self, a, indices):
        size = a.shape[1]
        indices = [(i * size) + val for i, val in enumerate(indices)]
        s = np.delete(a, indices, axis=None)  # flattened
        return s.reshape(-1, size-1)

    def forward_from_numpy_particular_action(self, s, a):
        s, a = torch.from_numpy(s), torch.from_numpy(a)
        a = (a - (self.num_actions / 2.0)) / (self.num_actions / 2.0)
        x = torch.cat((s.float(), a.float()), dim=1)
        return self.forward(x)

    def get_advantage(self, s, a, robot_ids, local_s, local_s_prime, reward):
        """ s: n x d array of global states
            a: n x r array of global action indices
            robot_ids: n x 1 array of the agent id in question
            p: n x u array of current automata action probabilities

            Output: beta values for each action (0 to 1), 1 being most favorable """

        output = self.forward_from_numpy_all_actions(s, a, robot_ids)
        expected_value = torch.mean(output, dim=1).unsqueeze(1)
        if self.counterfactual:
            advantages = self.reward_buff * (output - expected_value)
        else:
            return self.reward_buff * torch.from_numpy(reward.reshape((-1, 1)))
        if any(torch.isnan(advantages).flatten()):
            print('NAN DETECTED IN GET_ADVANTAGE: ', output, '#### EXPECTED VALUE ####', expected_value)

        robot_ids = torch.from_numpy(robot_ids).unsqueeze(1)
        particular_actions = torch.gather(torch.from_numpy(a), 1, robot_ids)
        advantages = torch.gather(advantages, 1, particular_actions).detach().numpy()

        return advantages.flatten()

    def update(self, s, a,  r):
        s, a, r = np.array(s), np.array(a), np.array(r)

        if self.noise_std > 0:
            s = s + np.random.normal(0, self.noise_std, s.shape)
        values = self.forward_from_numpy_particular_action(s, a)

        criterion = nn.MSELoss()
        loss = criterion(values.squeeze(), torch.from_numpy(r.flatten()).float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_single_transition(self, transition):
        return self.update(transition.global_state, transition.global_action, transition.reward)

    def train_multiple_transitions(self, transition_list):
        losses = []
        for transition in transition_list:
            losses.append(self.train_single_transition(transition))
        return np.mean(losses)
