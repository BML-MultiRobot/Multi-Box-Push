import sys, pickle
import matplotlib.pyplot as plt


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
    elif function.lower() == 'graph':
        if len(sys.argv) <= 3:
            print('input valid title...structure: analyze.py [function] [path] [title]') 
        elif type(data) == list:
            plt.plot(range(len(data)), data)
            plt.title(title)
        elif type(data) == np.ndarray:
            plt.plot(range(data.shape[0]), data)
            plt.title(title)
	plt.show()
    else:
        print('Input valid function: print or graph')

