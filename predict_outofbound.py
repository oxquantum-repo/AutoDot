from pathlib import Path

import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

from GPy_wrapper import GPyWrapper as GP
import GP_util

def L2_norm(arr, axis=None, keepdims=False):
    return np.sqrt(np.sum(np.square(arr), axis=axis, keepdims=keepdims))

def ur_from_vols_origin(vols, origin):
    vols = np.array(vols)
    num_data = vols.shape[0]
    if num_data == 0:
        return [], []
    ndim = vols.shape[1]

    if np.isscalar(origin):
        origin = origin * np.ones(ndim)

    diff = vols - origin[np.newaxis, :]
    r_all = L2_norm(diff, axis=1)
    u_all = diff / r_all[:,np.newaxis]
    return [u for u in u_all], r_all.tolist()

save_dir = Path('./save_Dominic_randomHS_origin0_fixed')

vols_poff = np.load(save_dir/'vols_poff.npy')
origin = np.load(save_dir/'origin.npy')
detected = np.load(save_dir/'detected.npy')

print(origin)
print(vols_poff.shape)

print(np.sum(detected))

num_training = 200
print(np.sum(detected[:num_training]))

num_active_gates = vols_poff.shape[-1]
###
# Gaussian process for r
###
l_prior_mean = 0.4 * np.ones(num_active_gates)
l_prior_var = 0.1*0.1 * np.ones(num_active_gates)
r_min, r_max =  0.0, np.sqrt(num_active_gates)* 2000.
v_prior_mean = ((r_max-r_min)/4.0)**2
v_prior_var = v_prior_mean**2
noise_var_r = np.square(10./2.0)

gp_r = GP_util.create_GP(num_active_gates, 'Matern52', v_prior_mean, l_prior_mean, (r_max-r_min)/2.0, const_kernel=True)
GP_util.set_GP_prior(gp_r, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
#GP_util.fix_hyperparams(gp_r, False, True)

# Training data
u_all, r_all = ur_from_vols_origin(vols_poff[:num_training], origin)
r_all_gp = np.array(r_all)
OoB = np.logical_not(detected[:num_training]) # Out of bound indice
r_all_gp[OoB] += 300. # Add dummy distance for OoB measurement 
gp_r.create_model(np.array(u_all), r_all_gp, noise_var_r, noise_prior='fixed')

# Optimize GP
gp_r.optimize(num_restarts=20, opt_messages=False, print_result=True)

# Predict distance 
u_test, _ = ur_from_vols_origin(vols_poff[num_training:], origin)
r_mean_test, r_var_test = gp_r.predict_f(np.array(u_test))

# Distance to the box
box_lb = -2000. * np.ones(num_active_gates)
dist_origin_box = origin - box_lb
r_box = list()
for u in u_test:
    steps_each = dist_origin_box / -u
    bounding_dim = np.argmin(steps_each)
    r = steps_each[bounding_dim]
    r_box.append(r)

# Calculate the probability r < r_box
prob_inside = norm.cdf(np.array(r_box), r_mean_test[:,0], np.sqrt(r_var_test[:,0]))
confusion_mat = np.zeros((2,2))
confusion_mat[0,0] = np.sum(np.logical_and(prob_inside > 0.5, detected[num_training:]))
confusion_mat[1,1] = np.sum(np.logical_and(prob_inside <= 0.5, np.logical_not(detected[num_training:])))
confusion_mat[0,1] = np.sum(np.logical_and(prob_inside <= 0.5, detected[num_training:]))
confusion_mat[1,0] = np.sum(np.logical_and(prob_inside > 0.5, np.logical_not(detected[num_training:])))
print(detected.shape, prob_inside.shape)
print(confusion_mat)
