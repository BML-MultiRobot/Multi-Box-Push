from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', ('local_state', 'local_action', 'local_index', 'global_state', 'global_action'))


class PolicyMemory(object):
    def __init__(self, size=10000):
        self.memory = []
        self.position = 0
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

    def sample(self):
        """ Get the samples from the entire replay. Not necessarily contiguous
            (because retrieving samples from multiple agents). """
        transitions = Transition(*zip(*self.memory))
        return transitions

    def __len__(self):
        return len(self.memory)
