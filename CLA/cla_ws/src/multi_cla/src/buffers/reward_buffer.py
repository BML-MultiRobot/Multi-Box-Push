from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', ('global_state', 'global_action', 'reward', 'robot_id'))


class RewardMemory(object):
    def __init__(self, size=10000):
        self.memory = []
        self.position = 0
        self.size = size

    def push(self, global_state, global_action, reward, num_agents):
        """ Global_state: state inputted into neural network for global reward
            Global_action: joint action taken in this sample
            reward: reward associated with this sample
            num_agents: number of agents in environment
        """
        if len(self.memory) >= self.size:
            self.memory.pop(0)
        for i in range(num_agents):
            self.memory.append(Transition(global_state, global_action, reward, i))

    def sample(self, batch=128):
        """ Randomly sample batch from replay """
        choices = np.random.choice(len(self.memory), batch)
        mem = map(self.memory.__getitem__, choices)
        transitions = Transition(*zip(*mem))
        return transitions

    def __len__(self):
        return len(self.memory)
