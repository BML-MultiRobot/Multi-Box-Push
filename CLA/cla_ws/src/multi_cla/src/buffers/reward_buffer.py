from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', ('local_state', 'local_action', 'reward'))


class RewardMemory(object):
    def __init__(self, size=10000):
        self.memory = []
        self.position = 0
        self.size = size

    def push(self, global_state, global_action, reward):
        """ Global_state: state inputted into neural network for global reward
            Global_action: joint action taken in this sample
            reward: reward associated with this sample
        """
        if len(self.memory) >= self.size:
            self.memory.pop(0)
        self.memory.append(Transition(global_state, global_action, reward))

    def sample(self, batch=128):
        """ Randomly sample batch from replay """
        choices = np.random.choice(len(self.memory), batch)
        mem = map(self.memory.__getitem__, choices)
        transitions = Transition(*zip(*mem))
        return transitions

    def __len__(self):
        return len(self.memory)
