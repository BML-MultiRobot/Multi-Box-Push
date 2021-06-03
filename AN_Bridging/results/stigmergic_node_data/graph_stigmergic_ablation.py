import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

directory = 'vrep_env1_in_nx'  # CHANGE ME
file_names = os.listdir(directory)

def get_moving_average(lst, resolution):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(lst, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= resolution:
            moving_ave = (cumsum[i] - cumsum[i - resolution]) / resolution
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)
        # else:
        #     moving_aves.append(cumsum[i] / len(cumsum))
    return moving_aves


def plot_average_moving_average(resolution):
    parameters_to_dates = {}
    all_parameters = {}

    for file in file_names:
        if 'hyperparameters' in file:
            file_path = os.path.join(directory, file)
            with open(file_path, 'rb') as f:
                parameters = pickle.load(f)

            for param_name, param in parameters.items():
                curr = all_parameters.get(param_name, set())
                curr.add(param)
                all_parameters[param_name] = curr

            param_names = tuple(sorted(parameters.keys()))
            parameters = tuple([parameters[param] for param in param_names])
            curr = parameters_to_dates.get(parameters, [])
            date = '_'.join(file.split('_')[:7])
            curr.append(date)
            parameters_to_dates[parameters] = curr
    # with open(directory + '/all_parameters_info.txt', "wb") as fp:  # Pickling
    #     for name, parameters in all_parameters.items():
    #         fp.write(name + ': ')
    #         for p in sorted(parameters):
    #             fp.write(str(p) + ', ')
    #         fp.write('\n')
    #     fp.write('\n')

    metric = 'steps'# 'indicator_success_episode' #
    ablation = 'Box Preference Decay'  # CHANGE ME
    env = 'env_1' if metric == 'indicator_success_episode' else ''
    parameters_graph = {
        'Box Preference Decay': [.95],
        'Detection Radius': [3],
        'Distance Boltzmann': [.5],
        'Explore Decay': [.95],
        'Initial Explore': [1],
    }
    parameters_graph[ablation] = sorted(all_parameters[ablation])
    ablation_index = sorted(parameters_graph.keys()).index(ablation)
    for d in parameters_graph['Box Preference Decay']:
        for e in parameters_graph['Detection Radius']:
            for i_e in parameters_graph['Distance Boltzmann']:
                for b_pref in parameters_graph['Explore Decay']:
                    for b in parameters_graph['Initial Explore']:
                        p = (d, e, i_e, b_pref, b)
                        dates = parameters_to_dates[p]
                        runs = []
                        for date in dates:
                            descriptions = [date, env, metric]
                            descriptions = [description for description in descriptions if len(description) > 0]
                            file_path = '_'.join(descriptions)
                            file_path = os.path.join(directory, file_path)
                            file_path = file_path + '.txt' if metric == 'indicator_success_episode' else file_path
                            with open(file_path, 'rb') as f:
                                data = pickle.load(f)
                            runs.append(np.array(data))
                        average = np.mean(np.array(runs), axis=0).flatten()
                        average = np.array(get_moving_average(average, resolution))
                        label = p[ablation_index]
                        plt.plot(range(average.shape[0]), average, label=r'$B_{decay}$ = ' + str(label))  # CHANGE ME
    plt.xlabel('Episode')
    plt.ylabel('Steps Until Success')
    plt.title('Performance Varying ' + 'Box Decay' + r' $B_{decay}$')  # CHANGE ME
    plt.legend()
    
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=8)

    plt.savefig('ablation ' + ablation)
    plt.show()


def modify_dictionary():
    for file in file_names:
        if 'hyperparameters' in file:
            file_path = os.path.join(directory, file)
            with open(file_path, 'rb') as f:
                parameters = pickle.load(f)
            print(parameters.keys())
            delete = ['Exploration Pheromone Spatial Decay', 'Exploration Pheromone Temporal Decay']
            for key in delete:
                if key in parameters.keys():
                    del parameters[key]
            with open(file_path, "wb") as fp:  # Pickling
                pickle.dump(parameters, fp)


plot_average_moving_average(resolution=10)
