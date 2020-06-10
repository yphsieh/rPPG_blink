from data import SubjectGroup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw_histogram(bins=100):
    """
    data(np.array)
    bins(int): How many bins in the x-axis
    """
    global subject_group
    data = []
    for i in range(len(subject_group)):
        try:
            data = np.append(data, subject_group.subject_list[i].pulse_length)
        except Exception as e:
            data = subject_group.subject_list[i].pulse_length
            print(e)
    hist, bin_edges = np.histogram(data, bins=bins)
    plt.hist(data, bins=bin_edges)
    plt.xlabel("sequence length")
    plt.title("Variable Length")
    plt.savefig("variable_length.png")

if __name__ == '__main__':
    subject_group = SubjectGroup(pulse_dir='data_pulse', label_filename='label.xlsx')
    print(len(subject_group))
    for i in range(len(subject_group)):
        if max(subject_group.subject_list[i].pulse_length) >= 600:
            print(subject_group.subject_list[i])
            # print(max(subject_group.subject_list[i].pulse_length))
    draw_histogram()
