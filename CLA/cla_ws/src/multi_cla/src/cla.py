#! /usr/bin/env python

import numpy as np


class CLA:
    def __init__(self, state_indicators, num_actions, a, b, explore=1, explore_decay=.8, min_explore=0):
        # Learning rates
        self.b = b
        self.a = a

        states = self.recursive_state_add(state_indicators, 0)
        self.explore = explore
        self.explore_decay = explore_decay
        self.min_explore = min_explore

        # Policy mapping strings to automata policy
        self.num_actions = num_actions
        self.policy = {s: np.array([1.0/num_actions] * num_actions) for s in states}
        self.indices = np.arange(num_actions)
        return

    def recursive_state_add(self, state_indicators, curr_element_index):
        values = state_indicators[curr_element_index]
        if curr_element_index == len(state_indicators) - 1:
            return np.array(values) * (10 ** curr_element_index)

        rest = self.recursive_state_add(state_indicators, curr_element_index + 1)
        result = []
        for v in values:
            factor = (10**curr_element_index) * v
            curr = rest + factor
            result.append(curr)
        return np.array(result).flatten()

    def update_policy(self, s, a, global_s, global_a, robot_ids, reward_net):
        """ s: integer input encoding state
            a: index denoting action

            On-policy updates based on current observations
        """
        s, a, global_s, global_a, robot_ids = np.array(s), np.array(a), np.array(global_s), np.array(global_a), np.array(robot_ids)
        p = np.array([self.policy[state] for state in s])
        beta = reward_net.get_advantage(global_s, global_a, robot_ids, p)
        num_samples = s.shape[0]
        s_a = {(s[i], a[i]): [] for i in range(num_samples)}
        for i in range(num_samples):
            s_a[(s[i], a[i])].append(beta[i])

        s_a = {s_a: np.mean(beta_samples) for s_a, beta_samples in s_a.items()}
        for s_a, b in s_a.items():
            state, action = s_a
            vector = self.policy[state]
            new_vector = vector - (self.a * b * vector) + (self.b * (1 - b) * (1.0 / (self.num_actions - 1) - vector))
            new_vector[action] = vector[action] + (self.a * b * (1 - vector[action])) - (self.b * (1 - b) * vector[action])
            self.policy[state] = new_vector
        self.explore = max(self.explore*self.explore_decay, self.min_explore)
        return self.average_entropy()

    def get_action(self, s, probabilistic=True):
        automata = self.policy[s]
        p = self.explore * np.array([1.0/self.num_actions] * self.num_actions) + (1 - self.explore) * automata
        action = np.random.choice(self.indices, p=p)
        return action

    def get_entropy(self, s):
        automata = self.policy[s]
        entropy = np.sum([-p * np.log(p) for p in automata])
        return entropy

    def average_entropy(self):
        states = list(self.policy.keys())
        total_entropy = 0
        for s in states:
            total_entropy += self.get_entropy(s)
        return total_entropy / len(states)
