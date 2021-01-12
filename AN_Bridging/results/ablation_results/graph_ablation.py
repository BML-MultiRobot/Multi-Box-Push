import pickle
import matplotlib.pyplot as plt
from os.path import join
import os

directory = 'Target Network Update Frequency'
experiment_names = [x[0] for x in os.walk(directory)][1:]


def get_moving_average(lst, resolution):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(lst, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= resolution:
            moving_ave = (cumsum[i] - cumsum[i - resolution]) / resolution
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)
        else:
            moving_aves.append(cumsum[i] / len(cumsum))
    return moving_aves

def plot_ablation_tests(resolution):
    for experiment in experiment_names:
        label = experiment.split('_')[2]
        path = join(experiment, 'testing_rewards.txt')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        data = get_moving_average(data, resolution)
        plt.plot(range(len(data)), data, label=label)
    plt.title('Ablation Tests on ' + directory)
    plt.legend()
    plt.show()

plot_ablation_tests(40)

