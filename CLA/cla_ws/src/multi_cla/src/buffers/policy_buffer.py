from collections import namedtuple
import numpy as np
import pickle

Transition = namedtuple('Transition', ('local_state', 'local_action', 'robot_id', 'global_state', 'global_action'))


class PolicyMemory(object):
    def __init__(self, size=100000):
        self.memory = []
        self.size = size

    def push(self, local_state, local_action, local_index, global_state, global_action):
        """ Local_state: string indicator for local state
            Local_action: action index
            Local_index: the robot index associated to this sample
            Global_state: state inputted into neural network for global reward
            Global_action: joint action taken in this sample """
        if len(self.memory) >= self.size:
            self.memory.pop(0)
        self.memory.append(Transition(local_state, local_action, local_index, global_state, global_action))

    def sample(self, max_num=256):
        """ Get the samples from the entire replay. Not necessarily contiguous
            (because retrieving samples from multiple agents). """
        size = min(max_num, len(self.memory))
        sample = self.memory[:size]
        transitions = Transition(*zip(*sample))
        self.memory = self.memory[:-size]
        return transitions

    def batch_all_memory(self, batch):
        indices = np.arange(len(self.memory))
        np.random.shuffle(indices)
        total_segments = len(self.memory) // batch
        transition_list = []
        for i in range(total_segments):
            mem = map(self.memory.__getitem__, indices[i*batch: (i+1)*batch])
            transition_list.append(Transition(*zip(*mem)))
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
        return len(self.memory)
