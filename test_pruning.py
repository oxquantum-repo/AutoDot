import multiprocessing
import threading
import time
import numpy as np

import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath

import config
from test_common import Tester
import pygor_dummy
import util
import GP_util
from test_BO_device2 import random_hypersphere
import random_walk as rw
from config_model import DummyExtMeas

from test_BO_device4 import eval_proposed_ub_pruning, merge_data
from test_BO_device3 import compute_hardbound

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
    plt.savefig(name+'.svg')
    plt.close()

def plot_probs(f, lb, ub, gp_r, u_all, r_all, origin, name):
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
    resol = 20
    v1_grid = np.linspace(lb[0], ub[0], resol)
    v2_grid = np.linspace(lb[1], ub[1], resol)
    V2, V1 = np.meshgrid(v2_grid, v1_grid)
    v_test = np.array([V1.ravel(), V2.ravel()]).transpose()

    prob_each = np.zeros(V1.shape+(4,))
    prob_final = np.zeros(V1.shape)
    for i, v in enumerate(v_test):
        unrav_idx = np.unravel_index([i],V1.shape)
        prob_final[unrav_idx], prob_each[unrav_idx], _ = f(v)

    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(prob_final, cmap='viridis', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    plt.plot(v_pred[:,1], v_pred[:,0], color='yellow')
    plt.plot(v_all[:,1], v_all[:,0], 'o', color='yellow', alpha=0.3)
    plt.xlim([-2000, 0])
    plt.ylim([-2000, 0])

    plt.subplot(2,3,2)
    plt.imshow(prob_each[...,0], cmap='viridis', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    plt.plot(v_pred[:,1], v_pred[:,0], color='yellow')
    plt.plot(v_all[:,1], v_all[:,0], 'o', color='yellow', alpha=0.3)
    plt.xlim([-2000, 0])
    plt.ylim([-2000, 0])

    plt.subplot(2,3,3)
    plt.imshow(prob_each[...,1], cmap='viridis', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    plt.plot(v_pred[:,1], v_pred[:,0], color='yellow')
    plt.plot(v_all[:,1], v_all[:,0], 'o', color='yellow', alpha=0.3)
    plt.xlim([-2000, 0])
    plt.ylim([-2000, 0])

    plt.subplot(2,3,4)
    plt.imshow(prob_each[...,2], cmap='viridis', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    plt.plot(v_pred[:,1], v_pred[:,0], color='yellow')
    plt.plot(v_all[:,1], v_all[:,0], 'o', color='yellow', alpha=0.3)
    plt.xlim([-2000, 0])
    plt.ylim([-2000, 0])

    plt.subplot(2,3,5)
    plt.imshow(prob_each[...,3], cmap='viridis', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    plt.plot(v_pred[:,1], v_pred[:,0], color='yellow')
    plt.plot(v_all[:,1], v_all[:,0], 'o', color='yellow', alpha=0.3)
    plt.xlim([-2000, 0])
    plt.ylim([-2000, 0])

    plt.tight_layout()
    plt.savefig(name+'.svg')
    plt.close()

def test():
    # Hypersurface shape and pygor
    box_dim = 2
    box_a = np.array([-1300., -1000.])
    box_b = 500. * np.ones(box_dim)
    shape = pygor_dummy.Box(ndim=box_dim, a=box_a, b=box_b)
    th_leak = -500.*np.ones(2)
    shape = pygor_dummy.Leakage(shape, th_leak)
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
    num_init = 10
    u_all = random_hypersphere(box_dim, num_init)
    u_all, r_all, d_all, poff_all, detected_all, time_all, extra_meas_all = rw.get_full_data(u_all, tester, step_back, origin=origin)


    # GP hypersurface model
    ###
    # Gaussian process for r
    ###
    num_active_gates = box_dim
    l_prior_mean = 0.2 * np.ones(num_active_gates)
    l_prior_var = 0.1*0.1 * np.ones(num_active_gates)
    r_min, r_max =  0.0, np.sqrt(num_active_gates)* 2000.
    v_prior_mean = ((r_max-r_min)/4.0)**2
    v_prior_var = v_prior_mean**2
    noise_var_r = np.square(d_r/2.0)

    gp_r = GP_util.create_GP(num_active_gates, 'Matern52', v_prior_mean, l_prior_mean, (r_max-r_min)/2.0)
    GP_util.set_GP_prior(gp_r, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
    GP_util.fix_hyperparams(gp_r, False, True)

    # GP with initial observations
    #gp_r.create_model(u_all, r_all[:,np.newaxis], noise_var_r, noise_prior='fixed')
    #gp_r.optimize(num_restarts=20, opt_messages=False, print_result=True)

    #plot_example(lb_short, ub_short, shape, gp_r, u_all, r_all, origin, 'initial')
    #print(np.array(sampler.history_all).shape) # (num_samples, num_iter, ndim)
    #plot_example(lb_short, ub_short, shape, gp_r, u_all, r_all, origin, 'after')
    num_samples = 100
    do_random_meas(box_dim, num_samples, tester, step_back, origin, lb_short, ub_short, gp_r, do_extra_meas=do_extra_meas, save_dir=None)

def do_random_meas(num_active_gates, num_samples, tester, step_back, origin, lb_box, ub_box, gp_r, do_extra_meas=None, save_dir=None):

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
    ub_pruning_history = list()
    ub_pruning = origin

    axes = list(range(num_active_gates))

    hardub = np.zeros(origin.size)

    least_num_GP = 10 # use random direction, not random points on the hypersurface until reaching this number of data
    num_dvec=30 # the number of iterations to collect d vector
    min_interval_GP_opt = 10 # minimum interval for GP inference
    steps_GP_inference = [least_num_GP]
    while steps_GP_inference[-1] < num_samples:
        prev = steps_GP_inference[-1]
        nextstep = np.maximum(min_interval_GP_opt, int(0.1*prev))
        steps_GP_inference += [prev + nextstep]
    ub_opt_steps = steps_GP_inference

    num_particles = 200 # number of particles
    # Initial samples that are inside of the hypersurface
    samples = rw.random_points_inside(num_active_gates, num_particles, gp_r, origin, factor=0.5)
    boundary_points = []
    num_valid = 0
    while len(r_all) < num_samples:
        t = time.time()
        # Pick one surface point on the estimated surface
        if len(boundary_points) != 0:
            v = rw.pick_from_boundary_points(boundary_points) 
            v_origin = v - origin
            u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        else:
            print('WARNING: no boundary point is sampled')
            u = random_hypersphere(num_active_gates, 1)[0]
        print(u)

        # Start sampling
        sampling_on = False
        if num_valid >= least_num_GP:
            sampler = rw.create_sampler(gp_r, origin, lb_box, origin, sigma=50)# use origin as the upper bound
            stopper = multiprocessing.Value('i', 0)
            listener, sender = multiprocessing.Pipe(duplex=False)
            sampler.reset(samples, max_steps=100000, stopper=stopper, result_sender=sender)
            sampler.start()
            sampling_on = True

        # Estimation of r
        if num_valid >= least_num_GP:
            r_mean, r_var = gp_r.predict_f(u[np.newaxis,:], full_cov=False)
            r_est = np.maximum(r_mean - 3.0*np.sqrt(r_var), 0.0)
            r_est = r_est[0,0]
        else:
            r_est = None

        # Get measurements
        r, vols_pinchoff, found, t_firstjump = tester.get_r(u, origin=origin, r_est=r_est) # Measure the distance
        t1 = time.time() - t

        if len(r_all) > num_dvec: axes=[] # do not measure d vector
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

        # Update GP
        if len(r_all) >= least_num_GP:
            u_all_gp = np.array(u_all)
            r_all_gp = np.array(r_all)[:,np.newaxis]
            gp_r.create_model(u_all_gp, r_all_gp, (tester.d_r/2)**2, noise_prior='fixed')
            if len(r_all) in steps_GP_inference:
                gp_r.optimize(num_restarts=5, opt_messages=False, print_result=True)

        ub_changed = False
        # Compute hardbound
        change_hardbound, new_hardbound = compute_hardbound(poff_vec, found, vols_pinchoff, step_back, axes, hardub)
        if change_hardbound:
            hardub = new_hardbound
            if np.any(origin > hardub):
                origin[origin > hardub] = hardub[origin > hardub]
                ub_changed = True
            print('New origin, hardbound: ', origin)

        # Condition for changing the origin
        set_ub = len(r_all) in  ub_opt_steps
        ub_changed = False
        if set_ub:
            # Prepare data for optimisationa (all points, scored points)
            n = num_active_gates # short notation
            assert len(r_all) >= num_init
            pks_all = np.array([ext[3+n] >= 1 if len(ext)>=4+n else 0.0 for ext in extra_meas_all])
            lowres_score = np.array([ext[8+n] if len(ext)>=9+n else 0.0 for ext in extra_meas_all])
            points_extended, valid_extended = merge_data(vols_poff_all, detected_all, vols_poff_axes_all[:num_init], poff_all[:num_init])
            print(pks_all)
            print(lowres_score)

            # Compute probability map for illustration purpose
            prior = {'V': (10, 10), 'P': (10, 10), 'S': (10, 100)}
            M = 0.05
            f = lambda new_ub: eval_proposed_ub_pruning(new_ub, gp_r, origin, points_extended, valid_extended, pks_all, lowres_score, M, prior)
            #f(np.array([1., 1.]))
            #return
            plot_probs(f, lb_box, ub_box, gp_r, u_all, r_all, origin, str(len(r_all)))
            return

            '''
            ub_changed = True
            origin = origin_new
            origin.dump(str(save_dir / 'origin.npy'))
            origin_history.append(origin.copy())
            if save_dir is not None:
                np.array(origin_history).dump(str(save_dir / 'origin_history.npy'))
            u_all, r_all = ur_from_vols_origin(vols_poff_all, origin)
            '''

        t4 = time.time() - t

        print('Elapsed time: ', (t_firstjump, t1, t2, t3, t4))
        time_all.append((t_firstjump, t1, t2, t3, t4))
        if save_dir is not None:
            save(save_dir, np.array(vols_poff_all), np.array(u_all), [], np.array(r_all), np.array(d_all), np.array(poff_all), detected_all, tester.logger, extra_meas_all, np.array([]), time_all, [])

        # Stop sampling
        if sampling_on:
            stopper.value = 1
            counter, samples, boundary_points = listener.recv()
            sampler.join()
            print('Steps={} ({})'.format(counter, time.time()-t))

        # Project old samples to inside of the new estimated surface
        samples = np.minimum(samples, origin[np.newaxis,:]-10)
        if len(r_all) >= num_init:
            samples = rw.project_points_to_inside(samples, gp_r, origin)
        print('The number of collected samples: ', len(r_all))

    # make u_all 2dim array
    vols_poff_all = np.array(vols_poff_all)
    u_all = np.array(u_all)
    r_all = np.array(r_all)
    d_all = np.array(d_all)
    poff_all = np.array(poff_all)

    return vols_poff_all, u_all, r_all, d_all, poff_all, detected_all, time_all, time_removed_all, extra_meas_all, origin, vols_poff_axes_all


if __name__ == '__main__':
    test()
