import multiprocessing
import threading
import time
from functools import partial
import numpy as np

import mkl
mkl.set_num_threads(4)

import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib import cm

import config
from test_common import Tester
import pygor_dummy
import util
from GPy_wrapper import GPyWrapper, GPyWrapper_Classifier
import GP_util
from BO_common import random_hypersphere
import random_walk as rw
from config_model import DummyExtMeas

# https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

class plot_crosses(object):
    def __init__(self, data, num=30):
        self.data = data
        self.num = num
    def __call__(self):
        for cross in self.data[-self.num:]:
            trj_idx, a, b = cross
            plt.plot([a[1], b[1]], [a[0], b[0]], color='black')

def plot_example(lb, ub, shape, gp_r, u_all, r_all, origin, name, func_more=None, history_all=None):
    # Draw the True hypersurface
    resol = 100
    v1_grid = np.linspace(lb[0], ub[0], resol)
    v2_grid = np.linspace(lb[1], ub[1], resol)
    V2, V1 = np.meshgrid(v2_grid, v1_grid)
    v_test = np.array([V1.ravel(), V2.ravel()]).transpose()
    inside = shape(v_test)
    plt.figure()
    plt.imshow(inside.reshape(V1.shape), cmap='bwr', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])

    #Draw the estimated hypersurface
    num_grid = 100
    u_grid = np.zeros((num_grid,2))
    u_grid[:,0] = -np.linspace(0.,1.,num_grid)
    u_grid[:,1] = -np.sqrt(1.0 - np.square(u_grid[:,0]))

    r_pred, _ = gp_r.predict_f(u_grid)
    v_pred = r_pred * u_grid + origin
    plt.plot(v_pred[:,1], v_pred[:,0], color='yellow')

    # Draw the observations
    v_all = u_all * r_all[:,np.newaxis]
    plt.plot(v_all[:,1], v_all[:,0], 'o', color='yellow', alpha=0.3)

    if func_more is not None:
        func_more()

    if history_all is not None:
        sample = history_all[:,0,:]
        #plt.plot(sample[:,1], sample[:,0], color='cyan', linewidth=1)
        colorline(sample[:,1], sample[:,0], np.linspace(0,1,len(sample)), cmap=plt.get_cmap('Greens'), linewidth=1)

    plt.xlim([-2000, 0])
    plt.ylim([-2000, 0])
    plt.savefig(name+'.svg', dpi=500)
    plt.close()

def create_path(v_all, ub):
    Path = mpath.Path
    path_data = [(Path.MOVETO, (ub[1], ub[0]))]
    for v in v_all:
        path_data.append((Path.LINETO, (v[1], v[0])))
    path_data.append((Path.CLOSEPOLY, (ub[1], ub[0])))
    return path_data

def create_patch(v_all, ub, facecolor, alpha):
    path_data = create_path(v_all, ub)
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=facecolor, alpha=alpha)
    return patch

def create_multicol_line(v_all, vals, cmap, linewidth):
    points = v_all[:,[1,0]]
    points = points[:,np.newaxis,:]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #norm = plt.Normalize(vals.min(), vals.max())
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, joinstyle='round', capstyle='round')
    lc.set_array(vals)
    lc.set_linewidth(linewidth)
    return lc

def fig2b(lb, ub, gp_r, gpc_dict, u_all, r_all, poff_all, pks_all, lowres_score, origin, name, particles):
    u_all = np.array(u_all)
    r_all = np.array(r_all)
    # Compute  the estimated hypersurface
    num_grid = 200
    u_grid = np.zeros((num_grid,2))
    u_grid[:,0] = -np.linspace(0.,1.,num_grid)
    u_grid[:,1] = -np.sqrt(1.0 - np.square(u_grid[:,0]))

    r_pred, _ = gp_r.predict_f(u_grid)
    v_pred = r_pred * u_grid + origin

    #
    ub = ub - 0.001
    resol = 20
    v1_grid = np.linspace(lb[0], ub[0], resol)
    v2_grid = np.linspace(lb[1], ub[1], resol)
    V2, V1 = np.meshgrid(v2_grid, v1_grid)
    v_test = np.array([V1.ravel(), V2.ravel()]).transpose()

    prob, logprob, prob_all = predict_probs(v_test, gpc_dict)

    #plt.figure(figsize=(3,3))
    fig, ax = plt.subplots()
    fig.set_size_inches(4,3)
    #plt.colorbar()
    patch = create_patch(v_pred, ub, 'beige', 1.0)
    ax.add_patch(patch)

    # calculate the probability on the line
    prob, _, _ = predict_probs(v_pred, gpc_dict)
    lc = create_multicol_line(v_pred, prob.ravel(), 'viridis', 5)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Probability')

    # Draw observations
    # Observation location
    v_all = u_all * r_all[:,np.newaxis] + origin

    nopoff = np.logical_not(poff_all)
    #if np.sum(nopoff) > 0:
    #    plt.plot(v_all[nopoff,1], v_all[nopoff,0], 'o', color='red', alpha=0.3)
    poff_peak = np.logical_and(poff_all, pks_all)
    if np.sum(poff_peak) > 0:
        dots_peak = ax.scatter(v_all[poff_peak,1], v_all[poff_peak,0], marker='+', c='m', s=60, alpha=1., zorder=3)
    poff_nopeak = np.logical_and(poff_all, np.logical_not(pks_all))
    if np.sum(poff_nopeak) > 0:
        dots_nopeak = ax.scatter(v_all[poff_nopeak,1], v_all[poff_nopeak,0], marker='x', c='m', s=60, alpha=1., zorder=3)

    if particles is not None:
        dots_p = plt.scatter(particles[:20,1], particles[:20,0], s=30, c='gray', marker='o', zorder=3) 
        for p in particles[:20]: 
            plt.arrow(p[1], p[0], 50*np.random.normal(), 50*np.random.normal(), ec='gray', fc='gray', zorder=3, head_width=15., head_length=10.)
        #print(particles)
        ax.legend((dots_peak, dots_nopeak, dots_p), ('Peaks', 'No peaks', 'Particles'), loc='lower left', fontsize='small')

    ax.set_xlim([-2000, 0])
    ax.set_ylim([-2000, 0])
    plt.xticks([0, -1000, -2000], ['0', '-1', '-2'])
    plt.yticks([0, -1000, -2000], ['0', '-1', '-2'])
    fig.tight_layout()
    fig.savefig(name+'.svg', dpi=500)
    plt.close()

def plot_fig2a(shape, lb, ub, gp_r, gpc_dict, u_all, r_all, poff_all, pks_all, lowres_score, origin, name, particles, path_all):
    u_all = np.array(u_all)
    r_all = np.array(r_all)
    # Draw the True hypersurface
    resol = 100
    v1_grid = np.linspace(lb[0], ub[0], resol)
    v2_grid = np.linspace(lb[1], ub[1], resol)
    V2, V1 = np.meshgrid(v2_grid, v1_grid)
    v_test = np.array([V1.ravel(), V2.ravel()]).transpose()
    inside = shape(v_test)

    fig, ax = plt.subplots()
    fig.set_size_inches(4,3)

    #plt.imshow(inside.reshape(V1.shape), cmap='Reds', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])

    #Draw the estimated hypersurface
    num_grid = 100
    u_grid = np.zeros((num_grid,2))
    u_grid[:,0] = -np.linspace(0.,1.,num_grid)
    u_grid[:,1] = -np.sqrt(1.0 - np.square(u_grid[:,0]))

    r_pred, r_var = gp_r.predict_f(u_grid)
    r_std = np.sqrt(r_var)
    v_pred = r_pred * u_grid + origin
    patch = create_patch(v_pred, ub, 'beige', 1.0)
    ax.add_patch(patch)

    v_pred_in = np.maximum(0.,r_pred-2.*r_std)*u_grid + origin
    v_pred_out = (r_pred+2.*r_std)*u_grid + origin
    plt.plot(v_pred[:,1], v_pred[:,0], color='k', linewidth=2)
    plt.plot(v_pred_in[:,1], v_pred_in[:,0], 'k--', linewidth=2)
    plt.plot(v_pred_out[:,1], v_pred_out[:,0], 'k--', linewidth=2)

    # Draw observations
    # Observation location
    v_all = u_all * r_all[:,np.newaxis] + origin
    v_all = v_all[:-1]

    nopoff = np.logical_not(poff_all)
    poff_peak = np.logical_and(poff_all, pks_all)
    if np.sum(poff_peak[:-1]) > 0:
        dots_peak = ax.scatter(v_all[poff_peak[:-1],1], v_all[poff_peak[:-1],0], marker='+', c='m', s=60, alpha=1., zorder=3)
    poff_nopeak = np.logical_and(poff_all, np.logical_not(pks_all))
    if np.sum(poff_nopeak[:-1]) > 0:
        dots_nopeak = ax.scatter(v_all[poff_nopeak[:-1],1], v_all[poff_nopeak[:-1],0], marker='x', c='m', s=60, alpha=1., zorder=3)

    # Draw the current point and path
    line_proj = ax.plot([origin[1], path_all[-1][1][1]], [origin[0], path_all[-1][1][0]], ':', color='purple', linewidth=2, label='Projection line')
    ax.plot([path_all[-1][0][1], path_all[-1][1][1]], [path_all[-1][0][0], path_all[-1][1][0]], ':', color='purple', linewidth=2)
    dot_pred = ax.scatter(path_all[-1][0][1], path_all[-1][0][0], marker='d', c='c', s=60, linewidth=5, label='Predicted', zorder=3)
    if poff_peak[-1]:
        dot_true = ax.scatter(path_all[-1][1][1], path_all[-1][1][0], marker='+', c='m', s=60, label='Projected', zorder=3)
    else:
        dot_true = ax.scatter(path_all[-1][1][1], path_all[-1][1][0], marker='x', c='m', s=60, label='Projected', zorder=3)

    #ax.legend((dot_pred, dot_true, line_proj[1]), ('Predicted', 'Projected', 'Projection line'), loc='lower left')
    ax.legend(loc='lower left', fontsize='small')


    plt.xlim([-2000, 0])
    plt.ylim([-2000, 0])
    plt.xticks([0, -1000, -2000], ['0', '-1', '-2'])
    plt.yticks([0, -1000, -2000], ['0', '-1', '-2'])
    plt.savefig(name+'.svg', dpi=500)
    plt.close()

def plot_probs(lb, ub, gp_r, gpc_dict, u_all, r_all, poff_all, pks_all, lowres_score, origin, name):
    u_all = np.array(u_all)
    r_all = np.array(r_all)
    # Compute  the estimated hypersurface
    num_grid = 100
    u_grid = np.zeros((num_grid,2))
    u_grid[:,0] = -np.linspace(0.,1.,num_grid)
    u_grid[:,1] = -np.sqrt(1.0 - np.square(u_grid[:,0]))

    r_pred, _ = gp_r.predict_f(u_grid)
    v_pred = r_pred * u_grid + origin

    # Observation location
    v_all = u_all * r_all[:,np.newaxis] + origin

    #
    ub = ub - 0.001
    resol = 20
    v1_grid = np.linspace(lb[0], ub[0], resol)
    v2_grid = np.linspace(lb[1], ub[1], resol)
    V2, V1 = np.meshgrid(v2_grid, v1_grid)
    v_test = np.array([V1.ravel(), V2.ravel()]).transpose()

    prob, logprob, prob_all = predict_probs(v_test, gpc_dict)

    plt.figure(figsize=(10,5))
    plt.subplot(2,3,1)
    plt.imshow(prob.reshape(V1.shape), cmap='viridis', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    plt.colorbar()
    plt.plot(v_pred[:,1], v_pred[:,0], color='yellow') # TODO: multi-colored line
    nopoff = np.logical_not(poff_all)
    #if np.sum(nopoff) > 0:
    #    plt.plot(v_all[nopoff,1], v_all[nopoff,0], 'o', color='red', alpha=0.3)
    poff_peak = np.logical_and(poff_all, pks_all)
    if np.sum(poff_peak) > 0:
        plt.plot(v_all[poff_peak,1], v_all[poff_peak,0], 'o', color='OrangeRed', alpha=0.5)
    poff_nopeak = np.logical_and(poff_all, np.logical_not(pks_all))
    if np.sum(poff_nopeak) > 0:
        plt.plot(v_all[poff_nopeak,1], v_all[poff_nopeak,0], 'x', color='OrangeRed', alpha=0.5)

    plt.xlim([-2000, 0])
    plt.ylim([-2000, 0])

    num_probs = len(prob_all)

    for i in range(num_probs):
        plt.subplot(2,3,2+i)
        plt.imshow(prob_all[i].reshape(V1.shape), cmap='viridis', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
        plt.colorbar()
        plt.plot(v_pred[:,1], v_pred[:,0], color='yellow')
        plt.plot(v_all[:,1], v_all[:,0], 'o', color='yellow', alpha=0.3)
        plt.xlim([-2000, 0])
        plt.ylim([-2000, 0])

    plt.tight_layout()
    plt.savefig(name+'.svg', dpi=500)
    plt.close()

def test():
    # Hypersurface shape and pygor
    box_dim = 2
    box_a = np.array([-1300., -1000.])
    box_b = 500. * np.ones(box_dim)
    shape = pygor_dummy.Box(ndim=box_dim, a=box_a, b=box_b)
    #th_leak = -500.*np.ones(2)
    #shape = pygor_dummy.Leakage(shape, th_leak)
    origin = 0. * np.ones(box_dim)

    # Dummy function for extra measurement
    box_peaks = ([-1400., -1100], [-750, -900])
    box_goodscore = ([-1400., -1100], [-1100, -900])
    do_extra_meas = DummyExtMeas(box_peaks, box_goodscore)

    pg, active_gate_idxs, lb_short, ub_short, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor_dummy(shape)


    # Poff detector
    step_back = 100 # important param for measuring d
    len_after_pinchoff=100
    threshold_high = 0.8
    threshold_low = 0.2
    d_r = 10

    detector_pinchoff = util.PinchoffDetectorThreshold(threshold_low) # pichoff detector
    detector_conducting = util.ConductingDetectorThreshold(threshold_high) # reverse direction
    tester = Tester(pg, lb_short, ub_short, detector_pinchoff, d_r=d_r, len_after_pinchoff=len_after_pinchoff, logging=True, detector_conducting=detector_conducting, set_big_jump = set_big_jump, set_small_jump = set_small_jump)

    # Initial observations
    #num_init = 10
    #u_all = random_hypersphere(box_dim, num_init)
    #u_all, r_all, d_all, poff_all, detected_all, time_all, extra_meas_all = rw.get_full_data(u_all, tester, step_back, origin=origin)


    # GP hypersurface model
    ###
    # Gaussian process for r
    ###
    num_active_gates = box_dim
    l_prior_mean = 0.4 * np.ones(num_active_gates)
    l_prior_var = 0.1*0.1 * np.ones(num_active_gates)
    r_min, r_max =  0.0, np.sqrt(num_active_gates)* 2000.
    v_prior_mean = ((r_max-r_min)/4.0)**2
    v_prior_var = v_prior_mean**2
    noise_var_r = np.square(d_r/2.0)

    gp_r = GP_util.create_GP(num_active_gates, 'Matern52', v_prior_mean, l_prior_mean, (r_max-r_min)/2.0, const_kernel=True)
    GP_util.set_GP_prior(gp_r, l_prior_mean, l_prior_var, v_prior_mean, v_prior_var) # do not set prior for kernel var
    GP_util.fix_hyperparams(gp_r, False, False)
    #GP_util.set_GP_prior(gp_r, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
    #GP_util.fix_hyperparams(gp_r, False, True)

    ###
    # Gaussian process classifiers for predicting each probability component
    ###
    l_prior_mean =  500. * np.ones(num_active_gates)
    l_prior_var = 100.**2 * np.ones(num_active_gates)
    v_prior_mean =  50.
    v_prior_var = 20.**2
    gpc_dict = dict()
    gpc_dict['valid'] = GP_util.create_GP(num_active_gates, 'Matern52', lengthscale=l_prior_mean, const_kernel=True, GP=GPyWrapper_Classifier)
    gpc_dict['peak'] = GP_util.create_GP(num_active_gates, 'Matern52', lengthscale=l_prior_mean, const_kernel=True, GP=GPyWrapper_Classifier) # when a point is valid
    gpc_dict['goodscore'] = GP_util.create_GP(num_active_gates, 'Matern52', lengthscale=l_prior_mean, const_kernel=True, GP=GPyWrapper_Classifier) # when a point is valid and has peaks

    GP_util.set_GP_prior(gpc_dict['valid'], l_prior_mean, l_prior_var, v_prior_mean, v_prior_var) # do not set prior for kernel var
    GP_util.set_GP_prior(gpc_dict['peak'], l_prior_mean, l_prior_var, v_prior_mean, v_prior_var) # do not set prior for kernel var
    GP_util.set_GP_prior(gpc_dict['goodscore'], l_prior_mean, l_prior_var, v_prior_mean, v_prior_var) # do not set prior for kernel var

    # GP with initial observations
    fig2a = partial(plot_fig2a, shape)
    num_samples = 20
    do_random_meas(box_dim, num_samples, tester, step_back, origin, lb_short, ub_short, gp_r, gpc_dict, do_extra_meas=do_extra_meas, save_dir=None, fig2a=fig2a)

def update_gpc(gpc_dict, points_extended, poff_extended, pks_all, lowres_score, M):
    # All points for valid/invalid
    poff_extended = np.array(poff_extended)
    gpc_dict['valid'].create_model(points_extended, poff_extended.astype(np.float))

    # Only valid points for peaks/no peak
    assert len(pks_all) == len(lowres_score)
    num_scored = len(pks_all)
    is_valid = poff_extended[:num_scored]
    points_valid = points_extended[:num_scored][is_valid].copy()
    pks_valid = np.array(pks_all)[is_valid].astype(np.float).copy()
    gpc_dict['peak'].create_model(points_valid, pks_valid)
    #print('Training data, peak')
    #print(points_valid)
    #print(pks_valid)

    # Only peak points for goodscore/not

def optimize_gpc(gpc_dict):
    for gpc in gpc_dict.values():
        if gpc.model is not None:
            gpc.optimize()

def predict_probs(points, gpc_dict):
    p1 = gpc_dict['valid'].predict_prob(points)[:,0]
    p2 = gpc_dict['peak'].predict_prob(points)[:,0]

    probs = [p1, p2]

    total_prob = np.prod(probs, axis=0)
    log_total_prob = np.sum(np.log(probs), axis=0)
    return total_prob, log_total_prob, probs

def choose_next(points_candidate, points_observed, gpc_dict, d_tooclose = 10.):
    points_observed = np.array(points_observed)
    if len(points_candidate) == 0: # No cadidate points
        return None, None, None

    # Exclude samples that are too close to observed points
    #nottooclose = np.all(
    #        np.fabs(points_candidate[:,np.newaxis,:] - points_observed[np.newaxis,...])
    #        > d_tooclose, axis=(1,2))
    tooclose = np.any(
            np.all(np.fabs(points_candidate[:,np.newaxis,:] - points_observed[np.newaxis,...]) <= d_tooclose, axis=2),
            axis=1)
    nottooclose = np.logical_not(tooclose)

    if np.sum(nottooclose) == 0: # All points are too close to observed points
        return None, None, None

    points_reduced = points_candidate[nottooclose]
    prob = predict_probs(points_reduced, gpc_dict)[0]


    #idx_max = np.argmax(prob)
    #point_best = points_reduced[idx_max]
    p = prob / np.sum(prob)
    idx = np.random.choice(len(points_reduced), p=p)
    point_best =  points_reduced[idx]

    #print('Possible locations')
    #print(points_reduced)
    #print('Best location')
    #print(point_best)

    return point_best

def do_random_meas(num_active_gates, num_samples, tester, step_back, origin, lb_box, ub_box, gp_r, gpc_dict, do_extra_meas=None, save_dir=None, fig2a=None):

    vols_poff_all = list()
    vols_poff_axes_all = list()
    d_all = list()
    poff_all = list()
    detected_all = list()
    time_all = list()
    u_all = list()
    r_all = list()
    extra_meas_all = list()
    time_removed_all = list()
    ub_history = list()
    changeindex_history = list()
    path_all = list()

    axes = list(range(num_active_gates))

    hardub = np.zeros(origin.size) # Hardbound of ub_samples
    ub_samples = origin.copy() # Random samples should be below this point
    origin_to_ub = True

    least_num_GP = 5 # minimum number of iterations for GP inference
    num_dvec = 0 # the number of iterations to collect d vector
    min_interval_GP_opt = 3 # minimum interval for GP inference
    steps_GP_inference = [least_num_GP]
    while steps_GP_inference[-1] < num_samples:
        prev = steps_GP_inference[-1]
        nextstep = np.maximum(min_interval_GP_opt, int(0.1*prev))
        steps_GP_inference += [prev + nextstep]

    num_particles = 200 # number of particles
    samples = None
    point_selected = None
    boundary_points = []
    gp_availble = False
    while len(r_all) < num_samples:
        t = time.time()
        # Pick one surface point on the estimated surface
        if point_selected is not None:
            v = point_selected
            v_origin = v - origin
            u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        elif len(boundary_points) != 0:
            v = rw.pick_from_boundary_points(boundary_points) 
            v_origin = v - origin
            u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        else:
            print('WARNING: no boundary point is sampled')
            u = random_hypersphere(num_active_gates, 1)[0]
        print(u)

        # Start sampling
        sampling_on = False
        if gp_availble:
            # Initial samples that are inside of the hypersurface
            if samples is None:
                samples = rw.random_points_inside(num_active_gates, num_particles, gp_r, origin, lb_box, ub_samples)
            sampler = rw.create_sampler(gp_r, origin, lb_box, ub_samples, sigma=50)
            stopper = multiprocessing.Value('i', 0)
            listener, sender = multiprocessing.Pipe(duplex=False)
            sampler.reset(samples, max_steps=100000, stopper=stopper, result_sender=sender)
            sampler.start()
            sampling_on = True

        # Estimation of r
        if gp_availble:
            r_mean, r_var = gp_r.predict_f(u[np.newaxis,:], full_cov=False)
            r_est = np.maximum(r_mean - 2.0*np.sqrt(r_var), 0.0)
            r_est = r_est[0,0]
        else:
            r_est = None

        # Get measurements
        r, vols_pinchoff, found, t_firstjump = tester.get_r(u, origin=origin, r_est=r_est) # Measure the distance
        t1 = time.time() - t
        path_all.append(((r_mean[0,0]*u + origin) if r_est else origin, vols_pinchoff))

        if gp_availble:
            time.sleep(5)

        if len(r_all) >= num_dvec: axes=[] # do not measure d vector
        d_vec, poff_vec, meas_each_axis, vols_each_axis = tester.measure_dvec(vols_pinchoff+step_back, axes=axes)
        t2 = time.time() - t

        print('vols: ', vols_pinchoff, 'd_all: ', d_vec, 'poff: ', poff_vec)
        print('end_points:', vols_each_axis)

        # Store the measurement
        vols_poff_all.append(vols_pinchoff)
        vols_poff_axes_all.append(vols_each_axis)
        u_all.append(u)
        r_all.append(r)
        d_all.append(d_vec)
        poff_all.append(poff_vec)
        detected_all.append(found)

        # Extra measurement
        extra = meas_each_axis
        if found and do_extra_meas is not None:
            extra += do_extra_meas(vols_pinchoff)
        extra_meas_all.append(extra)
        t3 = time.time() - t

        # Stop sampling
        if sampling_on:
            stopper.value = 1
            counter, samples, boundary_points = listener.recv()
            sampler.join()
            print('Steps={} ({})'.format(counter, time.time()-t))

        # Compute hardbound
        change_hardbound, new_hardbound = util.compute_hardbound(poff_vec, found, vols_pinchoff, step_back, axes, hardub)
        if change_hardbound:
            hardub = new_hardbound
            outofbound = ub_samples > hardub
            if np.any(outofbound):
                ub_samples[outofbound] = hardub[outofbound]
            print('New upperbound: ', ub_samples)
            ub_history.append(ub_samples.copy())
            changeindex_history.append(len(r_all))
            if save_dir is not None:
                ub_samples.dump(str(save_dir / 'upperbound.npy'))
                np.array(ub_history).dump(str(save_dir / 'ub_history.npy'))
                np.array(changeindex_history).dump(str(save_dir / 'changeindex_history.npy'))

            if origin_to_ub:
                origin = ub_samples
                u_all, r_all = util.ur_from_vols_origin(vols_poff_all, origin)
                if save_dir is not None:
                    ub_samples.dump(str(save_dir / 'origin.npy'))
                    np.array(ub_history).dump(str(save_dir / 'origin_history.npy'))


        # Merge data of boundary points
        if len(r_all) >= min(least_num_GP, num_dvec):
            points_extended, poff_extended = util.merge_data(vols_poff_all, detected_all, vols_poff_axes_all[:min(num_dvec,len(r_all))], poff_all[:min(num_dvec,len(r_all))])


        # Update GP
        if len(r_all) >= least_num_GP:
            # Choose data for gp_r
            points_poff = points_extended[poff_extended]
            inside = np.all(points_poff < origin, axis=1)
            u_all_gp, r_all_gp = util.ur_from_vols_origin(points_poff[inside], origin, returntype='array')

            gp_r.create_model(u_all_gp, r_all_gp[:,np.newaxis], (tester.d_r/2)**2, noise_prior='fixed')
            gp_availble = True
            if len(r_all) in steps_GP_inference or (change_hardbound and origin_to_ub):
                gp_r.optimize(num_restarts=5, opt_messages=False, print_result=True)

        if fig2a and gp_availble:
            n = num_active_gates # short notation
            pks_all = np.array([ext[3+n] >= 1 if len(ext)>=4+n else 0.0 for ext in extra_meas_all])
            lowres_score = np.array([ext[8+n] if len(ext)>=9+n else 0.0 for ext in extra_meas_all])
            fig2a(lb_box, ub_box, gp_r, gpc_dict, u_all, r_all, poff_extended[:len(pks_all)], pks_all, lowres_score, origin, str(len(r_all))+'a', samples, path_all)

        # Project old samples to inside of the new estimated surface
        #samples = np.minimum(samples, ub_samples[np.newaxis,:]-1.)
        if gp_availble and samples is not None:
            samples = rw.project_points_to_inside(samples, gp_r, origin, factor=0.99)
        #samples = np.maximum(samples, lb_box[np.newaxis,:]+1.)
            samples_outside = np.logical_or(np.any(samples>ub_samples), np.any(samples<lb_box))
            print(lb_box, ub_samples)
            print('samples outside: ', samples[samples_outside])
            samples[samples_outside] = ub_samples - 1.

        print('The number of collected samples: ', len(r_all))

        # Update the probabilities and choose the best sample
        if len(r_all) >= least_num_GP:
            # Data preparation
            n = num_active_gates # short notation
            pks_all = np.array([ext[3+n] >= 1 if len(ext)>=4+n else 0.0 for ext in extra_meas_all])
            lowres_score = np.array([ext[8+n] if len(ext)>=9+n else 0.0 for ext in extra_meas_all])
            #print(poff_extended[:len(pks_all)])
            #print(pks_all)

            # Update GPC
            M = 0.05
            update_gpc(gpc_dict, points_extended, poff_extended, pks_all, lowres_score, M)
            if len(r_all) in steps_GP_inference:
                optimize_gpc(gpc_dict)

            if len(boundary_points) > 0:
                # Calculate correct boundary_points
                points_candidate = rw.project_crosses_to_boundary(boundary_points, gp_r, origin)
                # Choose the best point in boundary_points
                point_selected = choose_next(points_candidate, vols_poff_all, gpc_dict, d_tooclose = 10.)
                print('Next point: ', point_selected)

                # Move the point to the safe bound, it is not going to happen, but just for safety
                if point_selected is not None:
                    # Move the point to the safe bound, it is not going to happen, but just for safety
                    point_selected = np.maximum(point_selected, lb_box+1.)

            else:
                point_selected = None

            # plot something
            fig2b(lb_box, ub_box, gp_r, gpc_dict, u_all, r_all, poff_extended[:len(pks_all)], pks_all, lowres_score, origin, str(len(r_all)), samples)

        t4 = time.time() - t

        print('Elapsed time: ', (t_firstjump, t1, t2, t3, t4))
        time_all.append((t_firstjump, t1, t2, t3, t4))
        if save_dir is not None:
            save(save_dir, np.array(vols_poff_all), np.array(u_all), [], np.array(r_all), np.array(d_all), np.array(poff_all), detected_all, tester.logger, extra_meas_all, np.array([]), time_all, [])


    # make u_all 2dim array
    vols_poff_all = np.array(vols_poff_all)
    u_all = np.array(u_all)
    r_all = np.array(r_all)
    d_all = np.array(d_all)
    poff_all = np.array(poff_all)

    return vols_poff_all, u_all, r_all, d_all, poff_all, detected_all, time_all, time_removed_all, extra_meas_all, origin, vols_poff_axes_all


if __name__ == '__main__':
    test()
