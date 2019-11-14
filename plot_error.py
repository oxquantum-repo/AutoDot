import pickle
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

save_dir = Path('./save_Dominic_hs')
result_dir = save_dir/'pred_result'

origin = np.load(save_dir/'origin.npy')
test_result_all = pickle.load(open(result_dir/'test_result.dat', 'rb'))

n_all = list()
mae_all = list()
rmse_all = list()
for test_result in test_result_all:
    n = test_result['num_training']
    u_test = test_result['u_test']
    r_test = test_result['r_test']
    r_est = test_result['r_est']
    r_std = test_result['r_std']

    # error on estimating r
    diff = r_test - r_est
    if diff.size > 0:
        mae = np.mean(np.fabs(diff))
        rmse = np.sqrt(np.mean(np.square(diff)))

        n_all.append(n)
        mae_all.append(mae)
        rmse_all.append(rmse)

plt.figure()
plt.plot(n_all, rmse_all)
plt.savefig('rmse')
plt.close()

plt.figure()
plt.plot(n_all, mae_all)
plt.savefig('mae')
plt.close()
