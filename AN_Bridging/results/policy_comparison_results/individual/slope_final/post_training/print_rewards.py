import pickle
import numpy as np


def plot_past_rewards(path, description):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(description, np.mean(data), np.std(data))

plot_past_rewards('post_training_testing_rewards.txt', 'mean reward: ')
plot_past_rewards('success.txt', 'success proportion: ')
