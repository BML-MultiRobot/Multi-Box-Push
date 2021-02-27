from collections import namedtuple
import numpy as np
import pickle

Transition = namedtuple('Transition', ('global_state', 'global_action', 'reward', 'robot_id'))


class RewardMemory(object):
    def __init__(self, size=100000, coma_infra=False):
        self.memory = []
        self.position = 0
        self.size = size
        self.coma_infra = coma_infra

    def push(self, global_state, global_action, reward, num_agents):
        """ Global_state: state inputted into neural network for global reward
            Global_action: joint action taken in this sample
            reward: reward associated with this sample
            num_agents: number of agents in environment
        """
        if len(self.memory) >= self.size:
            self.memory.pop(0)
        if self.coma_infra:
            for i in range(num_agents):
                self.memory.append(Transition(global_state, global_action, reward, i))
        else:
            self.memory.append(Transition(global_state, global_action, reward, 0))

    def sample(self, batch=128):
        """ Randomly sample batch from replay """
        choices = np.random.choice(len(self.memory), batch)
        mem = map(self.memory.__getitem__, choices)
        transitions = Transition(*zip(*mem))
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
