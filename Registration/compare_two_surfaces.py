from pathlib import Path
import os, sys
import time

import numpy as np

from GPy_wrapper import GPyWrapper as GP
from GPy_wrapper import GPyWrapper_Classifier as GPC
import GP_util
from util import L2_norm

def create_GP(num_active_gates, v_prior_mean, v_prior_var, l_prior_mean, l_prior_var, ave):
    gp_r = GP_util.create_GP(num_active_gates, 'Matern52', v_prior_mean, l_prior_mean, ave)
    GP_util.set_GP_prior(gp_r, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
    GP_util.fix_hyperparams(gp_r, False, True)
    return gp_r

def convert_Basel2_to_Basel1(v):
    #Basel1: [nose, r barrier, r wall, r plunger, m plunger, l plunger, l wall, l barrier]
    #Basel2: [nose, r barrier, r wall, r plunger, m plunger, l plunger,  l barrier]
    #Basel2 to Basel1: np.append(v[:6], [0., v[6]] )
    if v.ndim == 1:
        return np.append( v[:6], [0., v[6]] )
    elif v.ndim == 2:
        n = v.shape[0]
        return np.concatenate( (v[:,:6], np.zeros((n,1)), v[:,6:]), axis = 1)
    else:
        raise ValueError()

def convert_Basel1_to_Basel2(v):
    #Basel1: [nose, r barrier, r wall, r plunger, m plunger, l plunger, l wall, l barrier]
    #Basel2: [nose, r barrier, r wall, r plunger, m plunger, l plunger,  l barrier]
    #Basel2 to Basel1: np.append(v[:6], [0., v[6]] )
    if v.ndim == 1:
        return np.append( v[:6], v[7] )
    elif v.ndim == 2:
        n = v.shape[0]
        return np.concatenate( (v[:,:6], v[:,7:]), axis = 1)
    else:
        raise ValueError()

save_dir = Path('./data/save_Dominic_redo_Basel2')
type_target = 'Basel2'
type_new = 'Basel2'
'''
save_dir = Path('./data/save_Florian_redo')
type_target = 'Basel2'
type_new = 'Basel2'
'''


"""
v_target = np.load(save_dir/'vols_poff_prev.npy', allow_pickle=True)
found_target = np.load(save_dir/'detected_prev.npy', allow_pickle=True)



v_new = np.load(save_dir/'vols_poff_after.npy', allow_pickle=True)
found_new = np.load(save_dir/'detected_after.npy', allow_pickle=True)
"""




v_target = np.load('data/target_B1t2_b2.npy', allow_pickle=True)
found_target = np.ones(v_target.shape[0],dtype=np.bool)



v_new = np.load('data/registrated_pointset_B1toB2.npy', allow_pickle=True)
found_new = np.ones(v_new.shape[0],dtype=np.bool)




print(v_new[0])

if type_target == 'Basel2' and type_new == 'Basel1':
    v_new = convert_Basel1_to_Basel2(v_new)

num_active_gates = v_target.shape[1]
assert num_active_gates == 7
r_min, r_max =  0.0, np.sqrt(num_active_gates)* 2000.
d_r = 10.
###
# Gaussian process for r
###


l_prior_mean = 0.4 * np.ones(num_active_gates)
l_prior_var = 0.1*0.1 * np.ones(num_active_gates)
r_min, r_max =  0.0, np.sqrt(num_active_gates)* 2000.
v_prior_mean = ((r_max-r_min)/4.0)**2
v_prior_var = v_prior_mean**2
noise_var_r = np.square(d_r/2.0)

origin = 0.

gp_r_target = create_GP(num_active_gates, v_prior_mean, v_prior_var, l_prior_mean, l_prior_var, (r_max-r_min)/2.0)
r_target = L2_norm(v_target, axis=1, keepdims=True)
u_target = v_target / r_target
gp_r_target.create_model(u_target[found_target], r_target[found_target], noise_var_r, noise_prior='fixed')
gp_r_target.optimize(num_restarts=3, opt_messages=False, print_result=True)

gp_r_new = create_GP(num_active_gates, v_prior_mean, v_prior_var, l_prior_mean, l_prior_var, (r_max-r_min)/2.0)
r_new = L2_norm(v_new, axis=1, keepdims=True)
u_new = v_new / r_new
gp_r_new.create_model(u_new[found_new], r_new[found_new], noise_var_r, noise_prior='fixed')
gp_r_new.optimize(num_restarts=3, opt_messages=False, print_result=True)


# Scale comparison
found_both = np.logical_and(found_target, found_new)
num_poff_both = np.sum(found_both)
print(np.sum(found_target), np.sum(found_new), num_poff_both)

v_target_poff = v_target[found_both]
v_new_poff = v_new[found_both]

func_diff_given_scale = lambda scale: np.mean(L2_norm(v_target_poff-scale*v_new_poff, axis=1))

resol=1000
scale_all = np.linspace(0.3, 2, resol)
result_all = np.zeros(resol)
for i, scale in enumerate(scale_all):
    result_all[i] = func_diff_given_scale(scale)

idx_min = np.argmin(result_all)
scale = scale_all[idx_min]
dist = L2_norm(v_target_poff-scale*v_new_poff, axis=1)
print(scale)
print(np.mean(dist), np.std(dist))



"""
np.save("data/target_B2t2_cd1.npy",v_target_poff)
np.save("data/moving_B2t2_cd2.npy",v_new_poff)

np.save("data/scaled_pointset_B2toB2.npy",scale*v_new_poff)




np.save("data/target_B1t2_b2.npy",v_target_poff)
np.save("data/moving_B1t2_b1.npy",v_new_poff)

np.save("data/scaled_pointset_B1toB2.npy",scale*v_new_poff)
"""





# Normal vector comparison

angle_all = list()
normal_target_all = GP_util.gradient_surface(gp_r_target, v_target_poff, origin)
normal_new_all = GP_util.gradient_surface(gp_r_new, v_new_poff, origin)
for idx in range(num_poff_both):
    normal_target = normal_target_all[idx]
    normal_new = normal_new_all[idx]
    angle = np.arccos(np.inner(normal_target, normal_new) / (L2_norm(normal_target)*L2_norm(normal_new)))
    angle_all.append(angle)

plt.hist(angle_all)
plt.show()

print(np.mean(angle_all), np.std(angle_all))
