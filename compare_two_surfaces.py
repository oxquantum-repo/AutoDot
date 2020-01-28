from pathlib import Path
import os, sys
import time

import numpy as np
from scipy import optimize

from GPy_wrapper import GPyWrapper as GP
from GPy_wrapper import GPyWrapper_Classifier as GPC
import GP_util
from util import L2_norm
import random_walk as rw

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functools import partial
from registration_core import notranslation_affine_registration,simple_affine_registration,deformable_registration
import numpy as np

def visualize(iteration, error, X, Y):
    
    min_l = np.minimum(X.shape[0],Y.shape[0])
    
    abs_error = np.linalg.norm(X[:min_l]-Y[:min_l],axis=-1)
    
    print(iteration,error,np.sum(abs_error)/min_l)
    #diffs = X[:min_l]-Y[:min_l]
    #print(np.std(diffs,axis=0))

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

def find_nearest_single(point, gp_r, origin, x0):
    f = lambda v : np.sqrt(np.sum(np.square(point-v)))
    grad = lambda v: -(point-v)/np.sqrt(np.sum(np.square(point-v)))

    const = lambda v: np.sqrt(np.sum(np.square(v-origin))) - gp_r.predict_f(v[np.newaxis,:]/L2_norm(v))[0][0,0]
    grad_const = lambda v: GP_util.gradient_surface(gp_r, v[np.newaxis,:], origin, False)
    cons = ({'type': 'eq',
        'fun':const, 'jac':grad_const})
    # initial point
    res = optimize.minimize(f, x0, jac=grad, constraints= cons)
    return res.x, res.success

def find_nearest(points, gp_r, origin):
    # Projection points on the surface for x0
    r = L2_norm(points-origin, axis=1, keepdims=True)
    u = (points-origin) / r
    r_est = gp_r.predict_f(u)[0]
    x0 = origin + r_est*u
    result = np.zeros_like(points)
    for i, point in enumerate(points): 
        result[i], success = find_nearest_single(point, gp_r, origin, x0[i])
        #print(i, success, point-result[i])
    return result


'''
save_dir = Path('./save_Dominic_redo_Basel2')
type_target = 'Basel2'
type_new = 'Basel1'
'''
save_dir = Path('./save_Florian_redo')
type_target = 'Basel2'
type_new = 'Basel2'

v_target = np.load(save_dir/'vols_poff_prev.npy', allow_pickle=True)
found_target = np.load(save_dir/'detected_prev.npy', allow_pickle=True)

v_new = np.load(save_dir/'vols_poff_after.npy', allow_pickle=True)
found_new = np.load(save_dir/'detected_after.npy', allow_pickle=True)

found_both = np.logical_and(found_target, found_new)

if type_target == 'Basel2' and type_new == 'Basel1':
    v_new = convert_Basel1_to_Basel2(v_new)

num_active_gates = v_target.shape[1]
assert num_active_gates == 7
lb_target = np.ones(num_active_gates)*-2000.
ub_target = np.zeros(num_active_gates)

# Scale comparison
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

################################################
# transformation
################################################
callback = partial(visualize)
reg = notranslation_affine_registration(**{ 'X':np.copy(v_target_poff) , 'Y': np.copy(v_new_poff)})
reg.register(callback)
device_change_a = reg.B-np.diag(np.ones(7))


r_min, r_max =  0.0, np.sqrt(num_active_gates)* 2000.
d_r = 10.
################################################
# Gaussian process for r
################################################
l_prior_mean = 0.4 * np.ones(num_active_gates)
l_prior_var = 0.1*0.1 * np.ones(num_active_gates)
r_min, r_max =  0.0, np.sqrt(num_active_gates)* 2000.
v_prior_mean = ((r_max-r_min)/4.0)**2
v_prior_var = v_prior_mean**2
noise_var_r = np.square(d_r/2.0)

origin = 0.

gp_r_target = create_GP(num_active_gates, v_prior_mean, v_prior_var, l_prior_mean, l_prior_var, (r_max-r_min)/2.0)
r_target = L2_norm(v_target-origin, axis=1, keepdims=True)
u_target = (v_target-origin) / r_target
gp_r_target.create_model(u_target[found_target], r_target[found_target], noise_var_r, noise_prior='fixed')
#gp_r_target.create_model(u_target[found_both], r_target[found_both], noise_var_r, noise_prior='fixed')
gp_r_target.optimize(num_restarts=3, opt_messages=False, print_result=True)


################################################
# Find the nearest point on the surface
################################################
v_t = reg.TY
#v_t = v_new_poff
print('Finding nearest points')
v_nearest = find_nearest(v_t, gp_r_target, origin)
print(np.mean(L2_norm(v_t-v_nearest, axis=1)), np.std(L2_norm(v_t-v_nearest, axis=1)))
print(np.mean(L2_norm(v_t-v_target_poff, axis=1)), np.std(L2_norm(v_t-v_target_poff, axis=1)))

plt.figure()
plt.hist(L2_norm(v_t-v_nearest, axis=1), bins=40)
plt.savefig('hist')
plt.close()

print('Finished.')

def compare_angles(gp_r_target, v_ontarget, v_transformed, origin):

    gp_r_t = create_GP(num_active_gates, v_prior_mean, v_prior_var, l_prior_mean, l_prior_var, (r_max-r_min)/2.0)
    r_t = L2_norm(v_transformed-origin, axis=1, keepdims=True)
    u_t = (v_transformed-origin) / r_t
    #gp_r_t.create_model(u_t[found_new], r_t[found_new], noise_var_r, noise_prior='fixed')
    gp_r_t.create_model(u_t, r_t, noise_var_r, noise_prior='fixed')
    gp_r_t.optimize(num_restarts=3, opt_messages=False, print_result=True)

    # Normal vector comparison
    angle_all = list()
    #normal_target_all = GP_util.gradient_surface(gp_r_target, v_target_poff, origin)
    normal_target_all = GP_util.gradient_surface(gp_r_target, v_ontarget, origin)
    normal_t_all = GP_util.gradient_surface(gp_r_t, v_transformed, origin)
    for idx in range(num_poff_both):
        normal_target = normal_target_all[idx]
        normal_t = normal_t_all[idx]
        angle = np.arccos(np.inner(normal_target, normal_t) / (L2_norm(normal_target)*L2_norm(normal_t)))
        angle_all.append(angle)
    #print(angle_all)
    print(np.mean(angle_all), np.std(angle_all))
    plt.figure()
    plt.hist(angle_all, bins=40)
    plt.savefig('hist_angle')
    plt.close()

compare_angles(gp_r_target, v_nearest, v_t, origin)
#sys.exit(0)

################################################
# Generate augmented points
################################################
# Initial samples that are inside of the hypersurface
num_samples = np.sum(found_target)
samples=v_target[found_target]
r_samples = L2_norm(samples-origin, axis=1, keepdims=True)
u_samples = (samples - origin) / r_samples
r_est_samples = gp_r_target.predict_f(u_samples)[0]
samples_inside = origin + 0.8*r_est_samples*u_samples

# Random walk settings
sampler = rw.create_sampler(gp_r_target, origin, lb_target, ub_target, sigma=50, history=False)
#sampler.reset(samples, max_steps=1000)
# Run the sampler
print('Sampling started.')
counter, samples, boundary_points = sampler(samples_inside, max_steps=2000)
points_cross = rw.project_crosses_to_boundary(boundary_points, gp_r_target, origin)
print('Sampling finished.')
print(points_cross.shape)
# Choose 10000 random points
points_cross = points_cross[np.random.choice(len(points_cross),size=10000, replace=False)]
points_augmented = np.concatenate((v_target_poff, points_cross), axis=0)

################################################
# transformation
################################################
callback = partial(visualize)
reg = notranslation_affine_registration(**{ 'X': np.copy(points_augmented), 'Y': np.copy(v_new_poff)})
reg.register(callback)
device_change_b = reg.B-np.diag(np.ones(7))
#print(device_change_b - device_change_a)

v_t = reg.TY
print('Finding nearest points')
v_nearest = find_nearest(v_t, gp_r_target, origin)
print(np.mean(L2_norm(v_t-v_nearest, axis=1)))
print(np.mean(L2_norm(v_t-v_target_poff, axis=1)))
print('Finished.')

compare_angles(gp_r_target, v_nearest, v_t, origin)
sys.exit(0)

# Transformation
v_t = scale*v_new
v_t_poff = v_t[found_both]

gp_r_t = create_GP(num_active_gates, v_prior_mean, v_prior_var, l_prior_mean, l_prior_var, (r_max-r_min)/2.0)
r_t = L2_norm(v_t, axis=1, keepdims=True)
u_t = v_t / r_t
#gp_r_t.create_model(u_t[found_new], r_t[found_new], noise_var_r, noise_prior='fixed')
gp_r_t.create_model(u_t[found_both], r_t[found_both], noise_var_r, noise_prior='fixed')
gp_r_t.optimize(num_restarts=3, opt_messages=False, print_result=True)

# Normal vector comparison
angle_all = list()
normal_target_all = GP_util.gradient_surface(gp_r_target, v_target_poff, origin)
normal_t_all = GP_util.gradient_surface(gp_r_t, v_t_poff, origin)
for idx in range(num_poff_both):
    normal_target = normal_target_all[idx]
    normal_t = normal_t_all[idx]
    angle = np.arccos(np.inner(normal_target, normal_t) / (L2_norm(normal_target)*L2_norm(normal_t)))
    angle_all.append(angle)
#print(angle_all)
print(np.mean(angle_all), np.std(angle_all))
