from collections import namedtuple
import numpy as np
import pickle

Transition = namedtuple('Transition', ('local_state', 'local_action', 'next_state', 'next_action', 'robot_id', 'done',
                                       'global_state', 'global_action', 'next_global_state', 'next_global_action'))


class PolicyMemory(object):
    def __init__(self, size=100000):
        self.memory = []
        self.curr_rollout = self.curr_rollout = {i: [] for i in range(100)}
        self.size = size

    def push(self, local_state, local_action, next_s, next_a, local_index, done,
             global_state, global_action, next_global_state, next_global_action, end, robot_id):
        """ Local_state: string indicator for local state
            Local_action: action index
            Local_index: the robot index associated to this sample
            Global_state: state inputted into neural network for global reward
            Global_action: joint action taken in this sample """
        if len(self) >= self.size:
            self.memory.pop(0)
        self.curr_rollout[robot_id].append(Transition(local_state, local_action, next_s, next_a, local_index, done,
                                                      global_state, global_action, next_global_state, next_global_action))
        if end:
            self.memory.append(self.curr_rollout[robot_id])
            self.curr_rollout[robot_id] = []

    def batch_all_memory(self, shuffle=False):
        indices = np.arange(len(self.memory))
        if shuffle:
            np.random.shuffle(indices)
        transition_list = []
        for i in indices:
            curr_rollout = self.memory[i]
            transition_list.append(Transition(*zip(*curr_rollout)))
        return transition_list

    def load_data(self, path):
        with open(path, "rb") as input_file:
            self.memory = pickle.load(input_file)

    def clear(self, leave=0):
        if leave == 0:
            self.memory = []
        else:
            self.memory = self.memory[:leave]

    def __len__(self):
        return sum([len(r) for r in self.memory]) + sum(len(s) for _, s in self.curr_rollout.items())
