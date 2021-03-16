# ! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Model(nn.Module):
    def __init__(self, neurons, lr, num_actions, num_bots, noise=.02):
        super(Model, self).__init__()
        self.layers, self.neurons = [], neurons
        self.n_layers = len(self.neurons) - 2
        self.num_actions = num_actions
        self.num_bots = num_bots
        self.output = None
        self.lr = lr
        self.noise_std = noise

        self.createFeatures()

        self.optimizer = optim.Adam(super(Model, self).parameters(), lr=self.lr)
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
        for i in range(self.num_actions):
            a[range(a.shape[0]), robot_ids] = i
            curr_s, curr_a = torch.from_numpy(s), torch.from_numpy(a)
            curr_a = (curr_a - (self.num_actions / 2.0)) / (self.num_actions / 2.0)
            x = torch.cat((curr_s.float(), curr_a.float()), dim=1)
            curr_out = self.forward(x)
            if i == 0:
                output = curr_out
            else:
                output = torch.cat((output, curr_out), dim=2)
        return output

    def forward(self, x):
        for i in range(self.n_layers):
            x = eval(('F.relu' + '(self.fc{}(x))').format(i + 1))
        return self.output(x)

    def predict_numpy(self, x):
        x = torch.from_numpy(x)
        delta = self.forward(x)
        return (x + delta).detach().numpy()

    def update(self, s, a, s_prime):
        s, a, s_prime = np.array(s), np.array(a), np.array(s_prime)
        deltas = s_prime - s

        if self.noise_std > 0:
            s = s + np.random.normal(0, self.noise_std, s.shape)
        curr_a = (a - (self.num_actions / 2.0)) / (self.num_actions / 2.0)
        x = torch.cat((s.float(), curr_a.float()), dim=1)
        values = self.forward(x)

        criterion = nn.MSELoss()
        loss = criterion(values.squeeze(), torch.from_numpy(deltas).float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_single_transition(self, transition):
        return self.update(transition.global_state, transition.global_action, transition.global_state_prime)

    def train_multiple_transitions(self, transition_list):
        losses = []
        for transition in transition_list:
            losses.append(self.train_single_transition(transition))
        return np.mean(losses)
