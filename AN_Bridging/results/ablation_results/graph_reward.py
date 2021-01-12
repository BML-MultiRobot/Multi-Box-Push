import pickle
import matplotlib.pyplot as plt


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

def plot_past_rewards(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    plt.title("Rewards Over Episodes w/ Moving Average")
    lineRewards = get_moving_average(data, 40)
    x = range(len(lineRewards))
    plt.plot(x, lineRewards)
    # plt.legend()
    plt.show()

plot_past_rewards('testing_rewards.txt')
