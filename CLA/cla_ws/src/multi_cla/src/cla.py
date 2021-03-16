#! /usr/bin/env python

import numpy as np
import torch
from env_util import load_data


class CLA:
    def __init__(self, state_indicators, num_actions, a, b, q_learn=False, gamma=.8, td_lambda=.95,
                 alpha=1.0, steps_per_train=10, explore=1, explore_decay=.8, min_explore=0, test_mode=False):
        # Learning rates
        self.b = b
        self.a = a

        states = self.recursive_state_add(state_indicators, 0)
        self.explore = explore
        self.explore_decay = explore_decay
        self.test_mode = test_mode
        self.min_explore = min_explore
        self.q_learn = q_learn
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.alpha = alpha
        self.steps_per_train = steps_per_train
        self.lr = a

        # Policy mapping strings to automata policy
        self.num_actions = num_actions
        self.policy = {s: torch.tensor([1.0/num_actions] * num_actions, requires_grad=True) for s in states}
        self.q_values = {s: np.zeros(num_actions) for s in states}
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

    def update_policy(self, rollouts, reward_net):
        if self.q_learn:
            self.update_q_values(rollouts, reward_net)
        if self.q_learn:
            for _ in range(self.steps_per_train):
                entropy = self._update_policy(rollouts[0], reward_net)
        else:
            for r in rollouts:
                entropy = self._update_policy(r, reward_net)
        return entropy

    def update_q_values(self, rollouts, reward_net):
        # All inputs are in order of sampling meaning we can do Q-value estimates directly
        s_a = {}
        for r in rollouts:
            s, a, next_s, next_a, global_s, global_a, robot_ids, done = CLA.unpack(r)
            num_samples = s.shape[0]
            p = np.array([self.softmax(state).detach().numpy() for state in s])
            advantages = reward_net.get_advantage(global_s, global_a, robot_ids, p, normalize=False)
            r = self.alpha * advantages  # + (1 - self.alpha) * rewards
            q = self.q_value_estimate(next_s, next_a, r, 1 - done)

            for i in range(num_samples):
                if (s[i], a[i]) in s_a:
                    s_a[(s[i], a[i])].append(q[i])
                else:
                    s_a[(s[i], a[i])] = [q[i]]

        for s_a_pair, q_target in s_a.items():
            state, action = s_a_pair
            self.q_values[state][action] = np.mean(q_target)

    @staticmethod
    def unpack(p_trans):
        s = np.array(p_trans.local_state)
        a = np.array(p_trans.local_action)
        next_s = np.array(p_trans.next_state)
        next_a = np.array(p_trans.next_action)
        global_s = np.array(p_trans.global_state)
        global_a = np.array(p_trans.global_action)
        robot_ids = np.array(p_trans.robot_id)
        done = np.array(p_trans.done)
        return s, a, next_s, next_a, global_s, global_a, robot_ids, done

    def _update_policy(self, p_trans, reward_net):
        """ s: integer input encoding state
            a: index denoting action

            On-policy updates based on single rollout/trajectory
        """
        s, a, next_s, next_a, global_s, global_a, robot_ids, done = CLA.unpack(p_trans)

        if self.q_learn:
            num_samples = s.shape[0]
            s_a = {(s[i], a[i]): [] for i in range(num_samples)}
            loss = torch.zeros(1)
            for s_a_pair, _ in s_a.items():
                state, action = s_a_pair
                adv = (self.q_values[state][action] - self.value_estimate(state)) / np.std(self.q_values[state])
                loss += torch.log(self.softmax(state)[action]) * adv
            loss = loss / len(s_a)
            assert not any(torch.isnan(loss))
            loss.backward()
            for s, automata in self.policy.items():
                if torch.is_tensor(automata.grad):
                    with torch.no_grad():
                        if not any(torch.isnan(automata.grad)):
                            automata = automata + automata.grad * self.lr
                        if any(automata > 25):
                            automata = automata - torch.max(automata) + 25
                    automata.requires_grad = True
                    self.policy[s] = automata

        else:
            p = np.array([self.policy[state].detach().numpy() for state in s])
            beta = reward_net.get_advantage(global_s, global_a, robot_ids, p, normalize=True)
            num_samples = s.shape[0]
            s_a = {(s[i], a[i]): [] for i in range(num_samples)}
            for i in range(num_samples):
                s_a[(s[i], a[i])].append(beta[i])

            s_a = {s_a: np.mean(beta_samples) for s_a, beta_samples in s_a.items()}
            for s_a, b in s_a.items():
                state, action = s_a
                vector = self.policy[state].detach().numpy()
                new_vector = vector - (self.a * b * vector) + (self.b * (1 - b) * (1.0 / (self.num_actions - 1) - vector))
                new_vector[action] = vector[action] + (self.a * b * (1 - vector[action])) - (self.b * (1 - b) * vector[action])
                self.policy[state] = torch.tensor(new_vector)
        return self.average_entropy()

    def update_explore(self):
        self.explore = max(self.explore*self.explore_decay, self.min_explore)

    def q_value_estimate(self, next_states, next_actions, rewards, mask):
        target_qs = np.array([self.q_values[s][a] for s, a in zip(next_states, next_actions)])
        ret = np.zeros(next_states.size)
        ret[-1] = rewards[-1] + target_qs[-1]*mask[-1]

        for t in range(ret.size - 2, -1,  -1):
            ret[t] = rewards[t] + self.gamma * mask[t] * (self.td_lambda * ret[t + 1] + (1 - self.td_lambda) * target_qs[t])
        assert not any(np.isnan(ret))
        return ret

    def value_estimate(self, state):
        policy = self.softmax(state).detach().numpy()
        q = self.q_values[state]
        return np.sum(q * policy)

    def softmax(self, state):
        automata = self.policy[state]
        p = torch.exp(automata)
        return p / torch.sum(p)

    def get_action(self, s, probabilistic=True):
        automata = self.softmax(s).detach().numpy() if self.q_learn else self.policy[s].detach().numpy()
        p = np.array([1.0/self.num_actions] * self.num_actions) if np.random.random() < self.explore and not self.test_mode else automata
        action = np.random.choice(self.indices, p=p)
        return action

    def get_entropy(self, s):
        automata = self.softmax(s).detach().numpy() if self.q_learn else self.policy[s].detach().numpy()
        entropy = np.sum([-p * np.log(p) for p in automata])
        return entropy

    def average_entropy(self):
        states = list(self.policy.keys())
        total_entropy = 0
        for s in states:
            total_entropy += self.get_entropy(s)
        return total_entropy / len(states)

    def load_policy(self, path):
        self.policy = load_data(path)
