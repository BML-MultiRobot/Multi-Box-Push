import sys, pickle
import matplotlib.pyplot as plt
import os


folders = ['off_policy_cf_ddac', 'off_policy_cf_ddac_2', 'off_policy_IQL']
folders = sorted(folders)

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
    file_name = sys.argv[2]
    if len(sys.argv) > 3:
        title = sys.argv[3]

    if function.lower() == 'graph' or function.lower() == 'graph_ma':
        if len(sys.argv) <= 3:
            print('input valid title...structure: analyze.py [function] [path] [title]') 
	    sys.exit(0)
        for folder in folders:
	    path = os.path.join(folder, file_name)
	    with open(path, "rb") as input_file:
		    data = pickle.load(input_file)
	    if 'sigma' in folder:
		    val = folder.split('=')[1][1:]
		    if type(data) == list:
			data = get_moving_average(data, 20) if function.lower()=='graph_ma' else data
		        plt.plot(range(len(data)), data, label=r'DDAC $\sigma = ' + val + '$')
		    elif type(data) == np.ndarray:
			data = np.array(get_moving_average(data, 20)) if function.lower()=='graph_ma' else data
		        plt.plot(range(data.shape[0]), data, label=r'$DDAC \sigma = ' + val + '$')
	    else:
		    if type(data) == list:
			data = get_moving_average(data, 20) if function.lower()=='graph_ma' else data
		        plt.plot(range(len(data)), data, label=folder)
		    elif type(data) == np.ndarray:
			data = np.array(get_moving_average(data, 20)) if function.lower()=='graph_ma' else data
		        plt.plot(range(data.shape[0]), data, label=folder)
        plt.title(title)
	plt.legend()
	plt.show()
    else:
        print('Input valid function: graph or graph_ma')

