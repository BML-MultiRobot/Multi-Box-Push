from collections import namedtuple
import numpy as np
import pickle

Transition = namedtuple('Transition', ('local_state', 'local_action', 'next_state', 'next_action', 'robot_id', 'done', 'reward',
                                       'global_state', 'global_action'))

class PolicyMemory(object):
    def __init__(self, size=100000):
        self.memory = []
        self.size = size

    def push(self, local_state, local_action, next_s, next_a, local_index, done, reward,
             global_state, global_action):
        """ Local_state: string indicator for local state
            Local_action: action index
            Local_index: the robot index associated to this sample
            Global_state: state inputted into neural network for global reward
            Global_action: joint action taken in this sample """
        if len(self) >= self.size:
            self.memory.pop(0)
        self.memory.append(Transition(local_state, local_action, next_s, next_a, local_index, done, reward,
                                                      global_state, global_action))

    def batch_all_memory(self, shuffle=False):
        indices = np.arange(len(self.memory))
        if shuffle:
            np.random.shuffle(indices)
        transition_list = []
        for i in indices:
            curr_rollout = self.memory[i]
            transition_list.append(Transition(*zip(*curr_rollout)))
        return transition_list

    def sample(self, batch=128):
        """ Randomly sample batch from replay """
        choices = np.random.choice(len(self.memory), batch)
        mem = map(self.memory.__getitem__, choices)
        transitions = Transition(*zip(*mem))
        return transitions

    def load_data(self, path):
        with open(path, "rb") as input_file:
            self.memory = pickle.load(input_file)

    def clear(self, leave=0):
        if leave == 0:
            self.memory = []
        else:
            self.memory = self.memory[-leave:]

    def __len__(self):
        return len(self.memory)
