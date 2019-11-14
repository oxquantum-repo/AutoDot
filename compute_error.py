import pickle
from pathlib import Path
import numpy as np
import GPy

import GP_util

# Load data
save_dir = Path('./save_Dominic_hs')
origin = np.load(save_dir/'origin.npy')
ndim = len(origin)
vols_poff = np.load(save_dir/'vols_poff.npy')
total_obs = vols_poff.shape[0]
names = ['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']

# Construct GP
d_r = 10
l_prior_mean = 0.4 * np.ones(ndim)
l_prior_var = 0.1*0.1 * np.ones(ndim)
r_min, r_max =  0.0, np.sqrt(ndim)* 2000.
v_prior_mean = ((r_max-r_min)/4.0)**2
v_prior_var = v_prior_mean**2
noise_var_r = np.square(d_r/2.0)

gp_r = GP_util.create_GP(ndim, 'Matern52', v_prior_mean, l_prior_mean, (r_max-r_min)/2.0)
GP_util.set_GP_prior(gp_r, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
GP_util.fix_hyperparams(gp_r, False, True)

# Convert vols_poff to u_all and r_all
v_origin = vols_poff - origin[np.newaxis,:]
r_all = np.sqrt(np.sum(np.square(v_origin), axis=1, keepdims=True))
u_all = v_origin / r_all


# The number of observations to compute error
step_obs = 100
steps = np.arange(step_obs,total_obs+1, step_obs).tolist()
steps = [10, 20, 30, 40, 50] + steps
test_result_all = list()
for num_training in steps:
    print('The number of data points: ', num_training)
    print('The number of points for prediction: ', total_obs - num_training)

    u_train = u_all[:num_training]
    r_train = r_all[:num_training]
    gp_r.create_model(u_train, r_train, noise_var_r, noise_prior='fixed')
    gp_r.optimize(num_restarts=10, opt_messages=False, print_result=True)

    u_test = u_all[num_training:]
    r_test = r_all[num_training:]

    #prediction test
    r_est, r_var = gp_r.predict_f(u_test)
    r_std = np.sqrt(r_var)

    # u_test,
    test_result = {'num_training':num_training, 'u_test': u_test, 'r_test': r_test, 'r_est':r_est, 'r_std':r_std}
    diff = r_test - r_est
    test_result_all.append(test_result)


result_dir = save_dir/'pred_result'
result_dir.mkdir(exist_ok=True)
pickle.dump(test_result_all, open(result_dir/'test_result.dat', 'wb'))


# plot the result
