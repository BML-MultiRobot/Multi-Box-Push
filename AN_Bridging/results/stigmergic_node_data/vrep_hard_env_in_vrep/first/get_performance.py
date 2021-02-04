import numpy as np
import pickle

exploration_steps = 2
exclude_exploration = False


def get_performance(metric):
    file_path = metric + '.txt'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if exclude_exploration:
        data = data[exploration_steps:]
    average = np.mean(data)
    std = np.std(data)

    print(' Metric: ', metric)
    print(' Mean: ', average)
    print(' Standard Deviation: ', std)

    
get_performance('total_steps')
get_performance('success')
