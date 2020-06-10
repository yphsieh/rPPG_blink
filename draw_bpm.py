import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import glob


def Smooth(data, N):
    """
    data(np.array): first column is time, second column is bpm
    N(int): the smoothing window size, N must be an odd number
    return: average(np.array)
    """
    M = int((N - 1) / 2)
    average = []
    for i in range(M + 1, data.shape[0] - M):

        average_data = data[i]
        for j in range(1, M + 1):
            average_data += data[j]
            average_data += data[-1 * j]

        average_data /= N
        average.append(average_data)

    average = np.array(average)

    return average


def plotBPM(time, bpm):
    """
    time(np.array): 1-dim
    bpm(np.array): 1-dim
    """
    plt.plot(time, bpm)
    # plt.xlim(0, 20)
    plt.ylim([40, 150])


# plt.xlabel('Time')
# plt.ylabel('BPM')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Draw BPM of different files')
    parser.add_argument('--filename', nargs='+', help='file name to plot')
    parser.add_argument('-d', '--file_description', default=None, help='Draw all files in the directory')
    args = parser.parse_args()

    if args.file_description is None:
        filenum = len(args.filename)
        filenames = args.filename
    else:
        filenames = glob.glob(args.file_description)
        filenames.sort()
        filenum = len(filenames)

    print(filenames)

    for i in range(filenum):
        file = pd.read_csv(filenames[i], index_col=0)
        plt.subplot(int(filenum / 2 + 1), 2, i + 1)
        plotBPM(file['Time'].values, file['BPM'].values)
        plt.title(filenames[i][-6:])

    plt.show()
