import numpy as np
import matplotlib.pyplot as plt

import util
from test_common import Tester, translate_window_inside_boundary, scan_cross
from test_BO_device import get_u_init, lhs_hypersphere
from GPy_wrapper import GPyWrapper as GP
import pygor_dummy

def L2_norm(x, keepdims=False):
    return np.sqrt(np.sum(np.square(x),axis=-1, keepdims=keepdims))

def GP_hypersurface(u_all, r_all, fname='samples', overlay=None, lb_overlay=None, ub_overlay=None):
    num_active_gates = u_all.shape[1]
    # stats of initial measurements for transforming the values
    r_all = np.array(r_all)
    sample_var = np.var(r_all, ddof=1)
    sample_mean = np.mean(r_all)
    trans_val = lambda x : (x - sample_mean)/np.sqrt(sample_var)
    rev_transform = lambda y : y * np.sqrt(sample_var) + sample_mean

    # prior for GP
    l_prior_mean = 0.5 * np.ones(num_active_gates)
    l_prior_var = 0.3*0.3 * np.ones(num_active_gates)
    v_prior_mean = 1.0
    v_prior_var = 1.0*1.0
    noise_var = np.square(50.0) / sample_var

    gp = GP() # initialize GP environment
    # GP kernels
    gp.create_kernel(num_active_gates, 'Matern52', var_f=1.0, lengthscale=np.ones(num_active_gates))
    gp.set_kernel_length_prior(l_prior_mean, l_prior_var)
    gp.set_kernel_var_prior(v_prior_mean, v_prior_var)

    # GP with initial observations
    transformed = trans_val(r_all[:,np.newaxis])
    gp.create_model(u_all, transformed, noise_var, noise_prior='fixed')
    gp.optimize(num_restarts=10)

    # use more information (d_all to r)
    u_test = lhs_hypersphere(num_active_gates, 50)
    u_test = np.concatenate((u_test, [[0., -1.], [-1., 0.]]))
    sort_idx = np.argsort(u_test[:,0])
    u_test = u_test[sort_idx]
    r_samples = gp.posterior_sample_f(u_test, 10)[:,0,:].transpose() # (samples x Nnew)
    r_samples = rev_transform(r_samples)

    lb, ub = lb_overlay, ub_overlay
    plt.figure()
    if overlay is not None:
        plt.imshow(overlay, cmap='RdBu_r', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]], alpha=0.5)

    # posterior lines
    for i in range(len(r_samples)):
        v_samples = r_samples[i,:,np.newaxis] * u_test
        plt.plot(v_samples[:,1], v_samples[:,0])

    # obsevations
    for i in range(len(r_all)):
        v= r_all[i] * u_all[i]
        plt.plot(v[1], v[0], marker='x', markersize=12, color='black')

    plt.xlim(lb[1], ub[1])
    plt.ylim(lb[0], ub[0])
    plt.savefig(fname + '_samples')
    plt.close()

    # 1D plot
    mean_post, var_post = gp.predict_f(u_test)
    plt.figure()
    plt.plot(u_test[:,0], mean_post[:,0], 'C0')
    plt.fill_between(u_test[:,0], mean_post[:,0] - 2*np.sqrt(var_post[:,0]),
                                  mean_post[:,0] + 2*np.sqrt(var_post[:,0]),
                                  color='C0', alpha=0.2)
    plt.scatter(u_all[:,0], trans_val(r_all), marker='x', color='black')
    plt.savefig(fname + '_1D_plot')
    plt.close()

def calc_ur_from_d(u_all, r_all, d_all, step_back):
    if u_all.ndim == 1:
        u_all = u_all[np.newaxis,:].copy()
    if d_all.ndim == 1:
        d_all = d_all[np.newaxis,:].copy()
    r_all = np.atleast_1d(r_all)

    num_active_gates = u_all.shape[1]

    # more data ( convert d to r information)
    u_ext_all = list()
    r_ext_all = list()
    for i in range(len(u_all)):
        u = u_all[i]
        v = u * r_all[i]
        v_ext = (v[np.newaxis,:] + step_back) + np.eye(num_active_gates) * -d_all[i]
        u_ext = v_ext / L2_norm(v_ext, keepdims=True)
        r_ext = L2_norm(v_ext, keepdims=False)
        u_ext_all.append(u_ext)
        r_ext_all.append(r_ext)
    u_ext_all = np.concatenate(u_ext_all)
    r_ext_all = np.concatenate(r_ext_all)

    return u_ext_all, r_ext_all

def main():
    # make a dummy pygor
    num_active_gates = 2
    lb = np.array([-1000., -1000.])
    ub = np.array([0., 0.])
    box = pygor_dummy.Box(ndim=num_active_gates, a=[-500., -500], b=[500., 500.])
    pg = pygor_dummy.PygorDummyBinary(lb,ub, box)

    img = pg.do2d('c1', -1000., 0., 128, 'c2', -1000, 0., 128)[0]

    plt.figure()
    plt.imshow(img, cmap='RdBu_r', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    plt.savefig('test_dummy')
    plt.close()


    # algorithm parameters
    threshold = 0.2
    d_r = 20
    step_back = 50
    num_lhs = 5
    # pichoff detector
    detector_pinchoff = util.PinchoffDetectorThreshold(threshold)
    # tester to get r and d information
    tester = Tester(pg, lb, ub, detector_pinchoff, d_r=d_r, logging=True)
    u_all, r_all, d_all, detected_all, time_all = get_u_init(num_active_gates, num_lhs, tester, step_back) 

    # simple GP
    GP_hypersurface(u_all, r_all, fname='basic', overlay=img, lb_overlay=lb, ub_overlay=ub)

    u_ext_all, r_ext_all = calc_ur_from_d(u_all, r_all, d_all, step_back)

    # simple GP
    u_more = np.concatenate((u_all, u_ext_all))
    r_more = np.concatenate((r_all, r_ext_all))
    GP_hypersurface(u_more, r_more, fname='moredata', overlay=img, lb_overlay=lb, ub_overlay=ub)

    # hypersquare model


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    main()
