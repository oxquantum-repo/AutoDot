from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt

def draw_graph(save_dirs, labels, save_name):
    time_all = list()
    good_acc_all = list()
    for i in range(len(save_dirs)):
        save_dir = save_dirs[i]
        label = labels[i]
        summary = np.loadtxt(str(save_dir/'result.csv'), delimiter=',')
        time_acc = summary[:,5]
        n = len(time_acc)
        good_idxs = np.loadtxt(str(save_dir/'idx_good.txt'), delimiter=',').astype(np.int32)
        good_acc = np.zeros(n)
        good_acc[good_idxs] = 1
        good_acc = np.cumsum(good_acc)

        time_all.append(time_acc)
        good_acc_all.append(good_acc)

    plt.figure(figsize=(5,5))
    lines = list()
    for time, good_acc, label in zip(time_all, good_acc_all, labels):
        line, = plt.plot(time, good_acc, label=label)
        lines.append(line)
    plt.legend(handles=lines)
    plt.xlim([0, 12])
    plt.ylim([0, 20])
    plt.savefig(save_name)


save_dirs_set1 = [
            Path('./save_Dominic_gpc2_run4'),
            Path('./save_Dominic_gpc2_run5'),
            Path('./save_Dominic_gpc2_run6'),
            Path('./save_Dominic_gpc2_run7'),
            Path('./save_Dominic_gpc2_run8'),
            Path('./save_Dominic_gpc2_run9')]
labels_set = ['Run1', 'Run2', 'Run3', 'Run4', 'Run5', 'Run6', 'Run7', 'Run8', 'Run9']

save_dirs_set2 = [Path('./save_Florian_gpc2_run1'),
            Path('./save_Florian_gpc2_run2'),
            Path('./save_Florian_gpc2_run3'),
            Path('./save_Florian_gpc2_run4'),
            Path('./save_Florian_gpc2_run5')]

draw_graph(save_dirs_set1, labels_set, 'Basel1.svg')
draw_graph(save_dirs_set2, labels_set, 'Basel2.svg')
