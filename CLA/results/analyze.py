import sys, pickle
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
            # moving_aves.append(cumsum[i] / len(cumsum))
	    continue
    return moving_aves

if __name__ == '__main__':
    function = sys.argv[1]
    path = sys.argv[2]
    if len(sys.argv) > 3:
        title = sys.argv[3]
    with open(path, "rb") as input_file:
        data = pickle.load(input_file)

    if function.lower() == 'print':
        print('data:')
        print(data)
    elif function.lower() == 'graph' or function.lower() == 'graph_ma':
        if len(sys.argv) <= 3:
            print('input valid title...structure: analyze.py [function] [path] [title]') 
        elif type(data) == list:
	    data = get_moving_average(data, 20) if function.lower()=='graph_ma' else data
            plt.plot(range(len(data)), data)
            plt.title(title)
        elif type(data) == np.ndarray:
	    data = np.array(get_moving_average(data, 20)) if function.lower()=='graph_ma' else data
            plt.plot(range(data.shape[0]), data)
            plt.title(title)
	plt.savefig('new_plot.png')
	plt.show()
    else:
        print('Input valid function: print or graph')

