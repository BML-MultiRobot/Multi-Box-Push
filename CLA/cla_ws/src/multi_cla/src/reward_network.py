# ! /usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class RewardNetwork(nn.Module):
    def __init__(self, neurons, lr):
        super(RewardNetwork, self).__init__()
        self.layers, self.neurons = [], neurons
        self.n_layers = len(self.neurons) - 2
        self.output = None
        self.lr = lr

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
        a_minus_bot = self.delete_indices(a, robot_ids)
        s, a_minus_bot, robot_ids = torch.from_numpy(s), torch.from_numpy(a_minus_bot), torch.from_numpy(robot_ids.reshape(-1, 1))
        x = torch.cat((s.float(), a_minus_bot.float(), robot_ids.float()), dim=1)
        for i in range(self.n_layers):
            x = eval(('F.relu' + '(self.fc{}(x))').format(i + 1))

        return self.output(x)

    def delete_indices(self, s, indices):
        size = s.shape[1]
        indices = [(i * size) + val for i, val in enumerate(indices)]
        s = np.delete(s, indices, axis=None) # flattened
        return s.reshape(-1, size-1)

    def forward_from_numpy_particular_action(self, s, a, robot_ids):
        output = self.forward_from_numpy_all_actions(s, a, robot_ids)
        return torch.gather(output, 1, torch.from_numpy(robot_ids.reshape(-1, 1)))

    def predict(self, s, a, robot_ids):
        return self.forward(s, a, robot_ids).detach()

    def get_advantage(self, s, a, robot_ids, p):
        """ s: n x d array of global states
            a: n x r array of global action indices
            robot_ids: n x 1 array of the agent id in question
            p: n x u array of current automata action probabilities

            Output: beta values for each action (0 to 1), 1 being most favorable """
        p = torch.from_numpy(p)
        output = self.forward_from_numpy_all_actions(s, a, robot_ids)
        expected_value = torch.sum(output * p, dim=1)

        advantages = output - expected_value
        minimum = torch.min(advantages, dim=1)
        advantages = advantages - minimum
        maximum, _ = torch.max(advantages, dim=1)

        normalized = advantages / maximum
        normalized = torch.gather(normalized, 1, robot_ids).detach().numpy()

        unnormalized = torch.gather(advantages, 1, robot_ids).detach().numpy()
        maximum = maximum.detach().numpy()

        r = np.where(maximum > 1.0, normalized, unnormalized)
        return r

    def update(self, s, a, robot_ids, r):
        s, a, robot_ids, r = np.array(s), np.array(a), np.array(robot_ids), np.array(r)
        values = self.forward_from_numpy_particular_action(s, a, robot_ids)

        criterion = nn.MSELoss()
        loss = criterion(values.squeeze(), torch.from_numpy(r.flatten()).float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return
