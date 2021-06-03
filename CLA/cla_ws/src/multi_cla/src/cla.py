#! /usr/bin/env python

import numpy as np
import torch
from env_util import load_data


class CLA:
    def __init__(self, state_indicators, num_actions, params):
        # Learning rates
        self.a = params['a']

        states = self.recursive_state_add(state_indicators, 0)
        self.explore = params['explore']

        self.final_explore = params['final_explore']
        self.final_decrement = params['final_explore_decrement']

        self.gamma = params['gamma']
        self.q_lr = params['q_lr']
        self.lr = params['a']
        self.td_lambda = params['td_lambda']

        # Policy mapping strings to automata policy
        self.num_actions = num_actions
        self.policy = {s: torch.tensor([1.0/num_actions] * num_actions, requires_grad=True) for s in states}
        self.q_values = {s: np.zeros(num_actions) for s in states}

        self.q_final = {s: np.zeros(num_actions) for s in states}
        self.policy_final = {s: torch.tensor([1.0/num_actions] * num_actions, requires_grad=True) for s in states}

        self.indices = np.arange(num_actions)
        return

    def recursive_state_add(self, state_indicators, curr_element_index):
        values = state_indicators[curr_element_index]
        if curr_element_index == len(state_indicators) - 1:
            return np.array(values) * (100 ** curr_element_index)

        rest = self.recursive_state_add(state_indicators, curr_element_index + 1)
        result = []
        for v in values:
            factor = (100**curr_element_index) * v
            curr = rest + factor
            result.append(curr)
        return np.array(result).flatten()

    def update_policy(self, samples, reward_net, on_policy):
        loss = self.update_q_values(samples, reward_net, on_policy)
        entropy = self._update_policy(samples, on_policy)
        return entropy, loss

    def update_explore(self, use_final):
        if use_final:
            self.final_explore = max(0, self.final_explore - self.final_decrement)

    def update_q_values(self, samples, reward_net, on_policy):
        # All inputs are in order of sampling meaning we can do Q-value estimates directly
        s_a = {}
        if on_policy:
            for r in samples:
                s, a, next_s, next_a, global_s, global_a, robot_ids, done, reward = CLA.unpack(r)
                num_samples = s.shape[0]

                r = reward
                q = self.td_lambda_q_value_estimate(next_s, next_a, r, 1 - done)

                for i in range(num_samples):
                    if (s[i], a[i]) in s_a:
                        s_a[(s[i], a[i])].append(q[i])
                    else:
                        s_a[(s[i], a[i])] = [q[i]]
        else:
            s, a, next_s, next_a, global_s, global_a, robot_ids, done, reward = CLA.unpack(samples)
            num_samples = s.shape[0]
            advantages = reward_net.get_advantage(global_s, global_a, robot_ids, s, next_s, reward)

            r = advantages
            q = self.q_value_estimate(next_s, next_a, r, 1 - done)

            for i in range(num_samples):
                if (s[i], a[i]) in s_a:
                    s_a[(s[i], a[i])].append(q[i])
                else:
                    s_a[(s[i], a[i])] = [q[i]]

        diff = 0
        q = self.q_final if on_policy else self.q_values
        for s_a_pair, q_target in s_a.items():
            state, action = s_a_pair
            diff += abs(q[state][action] - np.mean(q_target))
            q[state][action] = (q[state][action] * (1 - self.q_lr)) + self.q_lr * np.mean(q_target)
        return diff / (len(s_a))

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
        reward = np.array(p_trans.reward)
        return s, a, next_s, next_a, global_s, global_a, robot_ids, done, reward

    def _update_policy(self, p_trans, on_policy):
        """ s: integer input encoding state
            a: index denoting action

            On-policy updates based on single rollout/trajectory
        """
        # TODO: Update on_policy or off_policy?
        s, a, next_s, next_a, global_s, global_a, robot_ids, done, _ = CLA.unpack(p_trans)  # TODO: Unpack greedu
        num_samples = s.shape[0]
        q = self.q_final if on_policy else self.q_values

        s_a = set([s[i] for i in range(num_samples)])
        assert len(s_a) <= len(self.policy)
        loss = torch.zeros(1)
        for state in s_a:
            adv = (q[state] - self.value_estimate(state, False)) / (1e-30 + np.std(q[state]))
            loss += (torch.sum(torch.log(self.softmax(state, False)).flatten() * torch.from_numpy(adv).flatten())) / self.num_actions

        assert not any(torch.isnan(loss))
        loss.backward()
        policy = self.policy_final if on_policy else self.policy

        for s, automata in policy.items():
            if torch.is_tensor(automata.grad):
                with torch.no_grad():
                    if not any(torch.isnan(automata.grad)):
                        automata = automata + automata.grad * self.lr
                    if any(automata > 25):
                        automata = automata - torch.max(automata) + 25
                automata.requires_grad = True
                policy[s] = automata
        return self.average_entropy(on_policy)

    def q_value_estimate(self, next_states, next_actions, rewards, mask):
        target_qs = np.array([np.max(self.q_values[s]) for s, a in zip(next_states, next_actions)])
        ret = rewards.flatten() + self.gamma * target_qs * mask
        # assert not np.any(np.isnan(ret))
        return ret

    def td_lambda_q_value_estimate(self, next_states, next_actions, rewards, mask):
        target_qs = np.array([self.q_values[s][a] for s, a in zip(next_states, next_actions)])
        ret = np.zeros(next_states.size)
        ret[-1] = rewards[-1] + target_qs[-1]*mask[-1]

        for t in range(ret.size - 2, -1,  -1):
            ret[t] = rewards[t] + self.gamma * mask[t] * (self.td_lambda * ret[t + 1] + (1 - self.td_lambda) * target_qs[t])
        assert not any(np.isnan(ret))
        return ret

    def value_estimate(self, state, use_final):
        policy = self.softmax(state, use_final).detach().numpy()
        q = self.q_values[state]
        return np.sum(q * policy)

    def softmax(self, state, on_policy_final):
        automata = self.policy_final[state] if on_policy_final else self.policy[state]
        p = torch.exp(automata)
        return p / torch.sum(p)

    def get_action(self, s, on_policy, probabilistic=True):
        if on_policy:
            automata = self.softmax(s, True).detach().numpy()
            if probabilistic:
                if np.random.random() < self.explore:
                    p = np.array([1.0/self.num_actions] * self.num_actions)
                else:
                    exploration_policy = self.softmax(s, on_policy_final=False).detach().numpy()
                    p = exploration_policy if np.random.random() < self.final_explore else automata
                action = np.random.choice(self.indices, p=p)
            else:
                action = self.indices[np.argmax(automata)]
        else:
            automata = self.softmax(s, False).detach().numpy()
            if probabilistic:
                p = np.array([1.0/self.num_actions] * self.num_actions) if np.random.random() < self.explore else automata
                action = np.random.choice(self.indices, p=p)
            else:
                action = self.indices[np.argmax(automata)]
            # return 0 if 5 <= s % 100 <= 11 else 1
        return action

    def get_entropy(self, s, use_coma):
        automata = self.softmax(s, use_coma).detach().numpy()
        entropy = np.sum([-p * np.log(p) for p in automata])
        return entropy

    def average_entropy(self, use_coma):
        states = list(self.policy.keys())
        total_entropy = 0
        for s in states:
            total_entropy += self.get_entropy(s, use_coma)
        return total_entropy / len(states)

    def load_policy(self, path):
        self.policy = load_data(path)
