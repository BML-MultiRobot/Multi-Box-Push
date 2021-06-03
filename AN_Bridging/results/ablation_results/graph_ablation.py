import pickle
import matplotlib.pyplot as plt
from os.path import join
import os

# THINGS THAT ARE MARKED 'CHANGE ME' SHOULD BE CHANGED IF WANT A DIFFERENT ABLATION GRAPH! 

directory = 'Exploration Steps'# 'Exploration Steps'# 'Batch Size'# 'Target Network Update Frequency'  # CHANGE ME
experiment_names = [x[0] for x in os.walk(directory)][1:]


def get_moving_average(lst, resolution):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(lst, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= resolution:
            moving_ave = (cumsum[i] - cumsum[i - resolution]) / resolution
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return moving_aves

def plot_ablation_tests(resolution):
    sorted_names = sorted(experiment_names, key=lambda x: int(x.split('_')[-1]))
    for experiment in sorted_names:
        label = experiment.split('_')[2]
        path = join(experiment, 'testing_rewards.txt')
        with open(path, 'rb') as f:
            data = pickle.load(f)
	data = data[:380]
        data = get_moving_average(data, resolution) 
        plt.plot(range(len(data)), data, label=r'$S$ = ' + label)  # CHANGE ME
    plt.title('Performance Varying ' + directory + r' $S$')  # CHANGE ME
    plt.xlabel('Episode')  # CHANGE ME
    plt.ylabel('Episodic Accumulated Reward')  # CHANGE ME

    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    print(labels)

    
    plt.legend(handles, labels)
    # plt.legend()
    plt.savefig('new_graph.png')
    plt.show()


plot_ablation_tests(150)  # CHANGE ME

