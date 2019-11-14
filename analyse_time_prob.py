from pathlib import Path
import os, sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#save_dir = Path('./save_Dominic_gpc2_run9')
save_dir = Path('./save_Florian_gpc2_run5')

vols = np.load(save_dir/'vols_poff.npy', allow_pickle=True)
n = num_active_gates = vols.shape[1]
num_examples = min(vols.shape[0], 500)
print('Total samples: ', num_examples)

time_meas = np.load(save_dir/'time_meas.npy', allow_pickle=True)
extra_meas = np.load(save_dir/'extra_measure.npy', allow_pickle=True)
origin_history = np.load(save_dir/'origin_history.npy', allow_pickle=True)
print('Origin: ', origin_history[-1])

time_meas = time_meas[:num_examples]
extra_meas = extra_meas[:num_examples]

exist = [len(ext) != n for ext in extra_meas]

num_poff = np.sum(exist)
print('Total pinched-off samples: ', num_poff)


num_peaks_all = list()
highres_all = list()
time_score_lowres_all = list()
time_score_highres_all = list()
time_dummy = [0., 0., 0., 0.]
for i in range(num_examples):
    if not exist[i]:
        num_peaks_all.append(0)
        highres_all.append(False)
        time_score_lowres_all.append(time_dummy)
        time_score_highres_all.append(time_dummy)
        continue

    num_pks = len(extra_meas[i][3+n])
    num_peaks_all.append(num_pks)

    exist_lowres = len(extra_meas[i])>n+4 
    assert exist_lowres is (num_pks > 0)
    if exist_lowres:
        time_score_lowres = extra_meas[i][9+n]
    else:
        time_score_lowres = time_dummy
        pass
    time_score_lowres_all.append(time_score_lowres)

    exist_highres = len(extra_meas[i])>n+10 
    highres_all.append(exist_highres)
    if exist_highres:
        time_score_highres = extra_meas[i][18+n]
    else:
        time_score_highres = time_dummy
    time_score_highres_all.append(time_score_highres)

time_score_lowres_all = np.array(time_score_lowres_all)
time_score_highres_all = np.array(time_score_highres_all)

time_trace_all = list()
time_scanl_all = list()
time_scanh_all = list()

for i in range(num_examples):
    time_trace = time_score_lowres_all[i,1] - time_score_lowres_all[i,0]
    time_scanl = time_score_lowres_all[i,3] - time_score_lowres_all[i,1]
    time_scanh = time_score_highres_all[i,3] - time_score_highres_all[i,0]
    time_trace_all.append(time_trace)
    time_scanl_all.append(time_scanl)
    time_scanh_all.append(time_scanh)

time_trace_all = np.array(time_trace_all)
time_scanl_all = np.array(time_scanl_all)
time_scanh_all = np.array(time_scanh_all)

num_ex_peaks = np.sum(np.array(num_peaks_all) > 0)
print('Total peak samples: ', num_ex_peaks)
num_ex_higres = np.sum(highres_all)
print('Total highres samples: ', num_ex_higres)


idx_good = np.loadtxt(str(save_dir/'idx_good.txt'), delimiter=',').astype(np.int32)
print('Total good samples: ', np.sum(idx_good <= num_examples))

print('Time ramping: ', np.mean(time_meas[:,1]))
print('Time computation: ', np.mean(time_meas[:,4]-time_meas[:,3]))
print('Time trace: ', np.mean(time_trace_all[exist]))
print('Time lowres: ', np.mean(time_scanl_all[np.array(num_peaks_all)>0]))
print('Time highres: ', np.mean(time_scanh_all[highres_all]))



######################

'''
save_dirs_set1 = [ Path('./save_Dominic_gpc2_run3'),
            Path('./save_Dominic_gpc2_run4'),
            Path('./save_Dominic_gpc2_run5'),
            Path('./save_Dominic_gpc2_run6'),
            Path('./save_Dominic_gpc2_run7'),
            Path('./save_Dominic_gpc2_run8'),
            Path('./save_Dominic_gpc2_run9')]
'''
save_dirs_set1 = [ Path('./save_Florian_gpc2_run1'),
            Path('./save_Florian_gpc2_run2'),
            Path('./save_Florian_gpc2_run3'),
            Path('./save_Florian_gpc2_run4'),
            Path('./save_Florian_gpc2_run5')]


time_all = list()
for d in save_dirs_set1:
    t = np.load(d/'time_meas.npy', allow_pickle=True)
    t[:,1:] = t[:,1:] - t[:,:-1]
    time_all.append(t[:500])
time_ave = np.mean(np.array(time_all), axis=0)
np.savetxt('time_ave.csv', time_ave, delimiter=',')

