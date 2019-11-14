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
import curvature

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

    #Draw several tangent line
    v_selected = v_pred[::10]
    #g = GP_util.gradient_surface(gp_r, v_selected, origin)
    k, _, g = GP_util.curvature_surface(gp_r, v_selected, origin)
    print(g, k)
    #print(g)
    x = np.linspace(lb[0], ub[0], resol)
    for i, v in enumerate(v_selected):
        y = -g[i,1]/g[i,0]*(x-v[1]) + v[0]
        plt.plot(x,y,'m-')
        plt.plot(v[1],v[0], 'm.')


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

def plot_d_est(lb, ub, shape, gp_r, u_all, r_all, origin, name, func_more=None):
    if np.isscalar(origin):
        origin = origin * np.ones_like(lb)
    # Draw the True hypersurface
    resol = 20
    v1_grid = np.linspace(lb[0], ub[0], resol)
    v2_grid = np.linspace(lb[1], ub[1], resol)
    V2, V1 = np.meshgrid(v2_grid, v1_grid)
    v_test = np.array([V1.ravel(), V2.ravel()]).transpose()

    result_grid = np.zeros(V1.shape)
    for i, v in enumerate(v_test):
        f = lambda v_: gp_r.predict_f(v_)[0][0] # single argument expected
        d = curvature.compute_d(f, v, origin, lb, ub)
        unrav_idx = np.unravel_index([i],V1.shape)
        result_grid[unrav_idx] = np.sum(d)
        #print(unrav_idx, d)

    plt.figure()
    plt.imshow(result_grid, cmap='bwr', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])

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

    plt.xlim([-2000, 0])
    plt.ylim([-2000, 0])
    plt.savefig(name+'.svg')
    plt.close()

def test():
    # Hypersurface shape and pygor
    box_dim = 2
    box_a = np.array([-1750., -1500.])
    box_b = 500. * np.ones(box_dim)
    shape = pygor_dummy.Box(ndim=box_dim, a=box_a, b=box_b)
    #th_leak = -500.*np.ones(2)
    #shape = pygor_dummy.Leakage(shape, th_leak)
    origin = 0.

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
    u_all = np.zeros((num_init,2))
    u_all[:,0] = -np.linspace(0.,1.,num_init)
    u_all[:,1] = -np.sqrt(1.0 - np.square(u_all[:,0]))

    u_all, r_all, d_all, poff_all, detected_all, time_all, extra_meas_all = rw.get_full_data(u_all, tester, step_back, origin=origin)


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

    gp_r = GP_util.create_GP(num_active_gates, 'Matern52', v_prior_mean, l_prior_mean, (r_max-r_min)/2.0)
    GP_util.set_GP_prior(gp_r, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
    GP_util.fix_hyperparams(gp_r, False, True)

    # GP with initial observations
    gp_r.create_model(u_all, r_all[:,np.newaxis], noise_var_r, noise_prior='fixed')
    gp_r.optimize(num_restarts=20, opt_messages=False, print_result=True)

    # Initial samples that are inside of the hypersurface
    num_samples = 1
    samples = rw.random_points_inside(box_dim, num_samples, gp_r, origin, lb_short, ub_short)

    # Random walk settings
    sampler = rw.create_sampler(gp_r, origin, lb_short, ub_short, sigma=50, history=True)
    #sampler.reset(samples, max_steps=1000)
    # Run the sampler
    counter, samples, boundary_points = sampler(samples, max_steps=200)

    plot_example(lb_short, ub_short, shape, gp_r, u_all, r_all, origin, 'initial', func_more=plot_crosses(boundary_points), history_all = np.array(sampler.history_all))
    plot_d_est(lb_short, ub_short, shape, gp_r, u_all, r_all, origin, 'dest')

    print(np.array(sampler.history_all).shape)
    return

    '''
    vals_all = [item['val'] for item in tester.logger]
    for i, vals in enumerate(vals_all[:num_init]):
        plt.figure() 
        plt.plot(vals)
        plt.savefig(str(i))
        plt.close()
    '''

    print('More observations')

    # Take more samples on the expected edge
    num_more = 30
    for i in range(num_more):
        # Pick one surface point on the estimated surface
        if len(boundary_points) != 0:
            v = rw.pick_from_boundary_points(boundary_points) 
            v_origin = v-origin
            u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        else:
            print('WARNING: no boundary point is sampled')
            u = random_hypersphere(box_dim, 1)[0]

        # Start sampling
        tic = time.time()
        sampler = rw.create_sampler(gp_r, origin, lb_short, ub_short, sigma=50)
        #stopper = threading.Event()
        stopper = multiprocessing.Value('i', 0)
        #sampler.reset(samples, max_steps=10000, stopper=stopper)
        listener, sender = multiprocessing.Pipe(duplex=False)
        sampler.reset(samples, max_steps=100000, stopper=stopper, result_sender=sender)
        sampler.start()

        # measurement
        time.sleep(5)
        r, vols_pinchoff, found = tester.get_r(u, origin=origin) # Do measurement
        print(u, vols_pinchoff)

        # Terminate sampling
        #stopper.set()
        stopper.value = 1
        #counter, samples, boundary_points = sampler.get_result()
        counter, samples, boundary_points = listener.recv()
        sampler.join()
        print('Steps={} ({})'.format(counter, time.time()-tic))

        # Append the new data
        u_all = np.concatenate((u_all, u[np.newaxis,:]))
        r_all = np.append(r_all, r)

        # Update the estimated surface
        gp_r.create_model(u_all, r_all[:,np.newaxis], noise_var_r, noise_prior='fixed')
        gp_r.optimize(num_restarts=20, opt_messages=False, print_result=True)

        # Project old samples to inside of the new estimated surface
        samples = rw.project_points_to_inside(samples, gp_r, origin)


    plot_example(lb_short, ub_short, shape, gp_r, u_all, r_all, origin, 'after', func_more=plot_crosses(boundary_points))


if __name__ == '__main__':
    test()
