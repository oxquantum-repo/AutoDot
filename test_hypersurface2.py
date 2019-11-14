import numpy as np
import matplotlib.pyplot as plt

import util
from test_common import Tester, translate_window_inside_boundary, scan_cross
from test_BO_device import get_u_init, lhs_hypersphere
import pygor_dummy

from hypersurface_GP import  Hypersurface_IndependentGP

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

def calc_v_from_urd(u_all, r_all, d_all, step_back):
    if u_all.ndim == 1:
        u_all = u_all[np.newaxis,:].copy()
    if d_all.ndim == 1:
        d_all = d_all[np.newaxis,:].copy()
    r_all = np.atleast_1d(r_all)

    num_active_gates = u_all.shape[1]

    v_ext_all = list()
    for i in range(len(u_all)):
        u = u_all[i]
        v = u * r_all[i]
        v_ext = (v[np.newaxis,:] + step_back) + np.eye(num_active_gates) * -d_all[i]
        v_ext_all.append(v_ext)
    v_ext_all = np.array(v_ext_all)

    v_ext_per_direction = np.zeros((num_active_gates,len(u_all), num_active_gates))
    for g in range(num_active_gates):
        v_ext_per_direction[g,...] = v_ext_all[:,g,:]
    return v_ext_per_direction

def calc_v_from_vd(v_all, d_all, step_back):
    if v_all.ndim == 1:
        v_all = v_all[np.newaxis,:].copy()
    if d_all.ndim == 1:
        d_all = d_all[np.newaxis,:].copy()
    num_active_gates = v_all.shape[1]

    v_ext_all = list()
    for i in range(len(v_all)):
        v = v_all[i]
        v_ext = (v[np.newaxis,:] + step_back) + np.eye(num_active_gates) * -d_all[i]
        v_ext_all.append(v_ext)
    v_ext_all = np.array(v_ext_all)

    v_ext_per_direction = np.zeros((num_active_gates,len(v_all), num_active_gates))
    for g in range(num_active_gates):
        v_ext_per_direction[g,...] = v_ext_all[:,g,:]

    return v_ext_per_direction

def main():
    # make a dummy pygor
    num_active_gates = 2
    lb = np.array([-1000., -1000.])
    ub = np.array([0., 0.])
    
    # cube
    box = pygor_dummy.Box(ndim=num_active_gates, a=[-500., -500], b=[500., 500.])
    pg = pygor_dummy.PygorDummyBinary(lb,ub, box)

    # circle
    #circle = pygor_dummy.Circle(r=500.,ndim=num_active_gates)
    #pg = pygor_dummy.PygorDummyBinary(lb,ub, circle)

    # convexhull
    conv_points = np.array([[-500., -100.],
                            [-700., 200.],
                            [1000., 200.],
                            [1000., -200.]])
    convexhull = pygor_dummy.Convexhull(conv_points)
    pg = pygor_dummy.PygorDummyBinary(lb,ub, convexhull)

    img = pg.do2d('c1', -1000., 0., 128, 'c2', -1000, 0., 128)[0]

    plt.figure()
    plt.imshow(img, cmap='RdBu_r', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    plt.savefig('test_dummy')
    plt.close()


    # algorithm parameters
    threshold = 0.2
    d_r = 10
    step_back = 50.
    num_lhs = 5
    # pichoff detector
    detector_pinchoff = util.PinchoffDetectorThreshold(threshold)
    # tester to get r and d information
    tester = Tester(pg, lb, ub, detector_pinchoff, d_r=d_r, logging=True)
    u_all, r_all, d_all, detected_all, time_all = get_u_init(num_active_gates, num_lhs, tester, step_back) 

    # simple GP
    #GP_hypersurface(u_all, r_all, fname='basic', overlay=img, lb_overlay=lb, ub_overlay=ub)

    #u_ext_all, r_ext_all = calc_ur_from_d(u_all, r_all, d_all, step_back)

    # intersection model
    h = Hypersurface_IndependentGP(num_active_gates)

    # data
    v_ext = calc_v_from_urd(u_all, r_all, d_all, step_back)
    h.set_data(v_ext, noise_var=(d_r/2.)**2)

    resol = 300
    num_samples = 10

    v1_grid = np.linspace(lb[0], ub[0], resol)
    v_test = np.hstack((v1_grid[:,np.newaxis],np.zeros(resol)[:,np.newaxis]))
    samples_v2 = -h.generate_samples_dg(v_test, 1, num_samples)

    v2_grid = np.linspace(lb[1], ub[1], resol)
    v_test = np.hstack((np.zeros(resol)[:,np.newaxis],v2_grid[:,np.newaxis]))
    samples_v1 = -h.generate_samples_dg(v_test, 0, num_samples)

    plt.figure()
    plt.imshow(img, cmap='RdBu_r', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]], alpha=0.3)
    for i in range(num_samples):
        line, = plt.plot(samples_v2[i],v1_grid,alpha=0.5)
        plt.plot(v2_grid, samples_v1[i], color=line.get_color(),alpha=0.5)
    # obsevations
    plt.scatter(v_ext[0,:,1], v_ext[0,:,0], marker='x', color='black')
    plt.scatter(v_ext[1,:,1], v_ext[1,:,0], marker='x', color='black')

    plt.xlim(lb[1], ub[1])
    plt.ylim(lb[0], ub[0])
    plt.savefig('temp_intersection')
    plt.close()

    # intersection likelihood
    V2, V1 = np.meshgrid(v2_grid, v1_grid)
    v_test = np.array([V1.ravel(), V2.ravel()]).transpose()
    L = h.boundary_likelihood(v_test)
    L = L.reshape(V1.shape)

    fig = plt.figure()
    ax = plt.gca()
    cs = ax.imshow(L, aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    fig.colorbar(cs)

    plt.xlim(lb[1], ub[1])
    plt.ylim(lb[0], ub[0])
    plt.savefig('temp_intersection_L')
    plt.close()

    # expected distance to the hypersurface
    d_all, _ = h.posterior_d(v_test, full_cov=False)
    ss_d = np.sqrt(np.sum(np.square(d_all),axis=0))
    ss_d = ss_d.reshape(V1.shape)


    fig = plt.figure()
    plt.imshow(img, cmap='RdBu_r', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]], alpha=0.5)
    ax = plt.gca()
    cs = ax.imshow(ss_d, aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]], alpha=0.5)
    fig.colorbar(cs)
    plt.xlim(lb[1], ub[1])
    plt.ylim(lb[0], ub[0])
    plt.savefig('temp_sum_sq_d')
    plt.close()

    # probability that v is inside of a hypersurface
    P = h.prob_inside_each(v_test)
    P = P.reshape(V1.shape)

    fig = plt.figure()
    ax = plt.gca()
    cs = ax.imshow(P, aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    fig.colorbar(cs)

    plt.xlim(lb[1], ub[1])
    plt.ylim(lb[0], ub[0])
    plt.savefig('temp_prob_inside')
    plt.close()

    # Expected improvement
    v = np.zeros(num_active_gates)
    d_best = tester.measure_dist_all_axis(v)
    print(d_best)

    EI = h.EI2(v_test, d_best)
    EI = EI.reshape(V1.shape)

    fig = plt.figure()
    ax = plt.gca()
    cs = ax.imshow(EI, aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    fig.colorbar(cs)

    plt.xlim(lb[1], ub[1])
    plt.ylim(lb[0], ub[0])
    plt.savefig('temp_EI')
    plt.close()

    # Expected improvement with stepback
    v = np.zeros(num_active_gates)
    d_best = tester.measure_dist_all_axis(v)
    print(d_best)

    EI = h.EI2(v_test, d_best, stepback=100.)
    EI = EI.reshape(V1.shape)

    fig = plt.figure()
    ax = plt.gca()
    cs = ax.imshow(EI, aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    fig.colorbar(cs)

    plt.xlim(lb[1], ub[1])
    plt.ylim(lb[0], ub[0])
    plt.savefig('temp_EI_stepback')
    plt.close()


    #algorithm
    stepback = 50.
    num_iter = 5

    v_init = -stepback * np.ones(num_active_gates) # initial point
    # measurement
    d = tester.measure_dist_all_axis(v_init + stepback)

    v_all = np.array([v_init])
    d_all = np.array([d])
    v_ext = calc_v_from_vd(v_all, d_all, step_back)
    # update the hypersurface model
    h = Hypersurface_IndependentGP(num_active_gates)
    h.set_data(v_ext, noise_var=(d_r/2.)**2)

    v_bestL_all = list()
    v_bestEI_all = list()

    for i in range(num_iter):
        # draw corner likelihood
        L = h.boundary_likelihood(v_test)
        draw_map(L.reshape((V1.shape)), lb, ub, 'Corner_{}'.format(i))
        best_L_idx = np.argmax(L)
        v_bestL_all.append(v_test[best_L_idx])

        # d at best v
        d_sum = np.sum(d_all,axis=1)
        d_best_idx = np.argmin(d_sum)
        d_best = d_all[d_best_idx]

        print('d_best:', d_best)

        # next point
        EI = h.EI2(v_test, d_best, stepback=stepback)
        draw_map(EI.reshape((V1.shape)), lb, ub, 'EI_{}'.format(i))
        best_idx = np.argmax(EI)
        v_bestEI = v_test[best_idx]
        v_bestEI_all.append(v_bestEI)

        # go the the next point
        d_step = 5.
        dist = L2_norm(v_all[-1] - v_bestEI)
        unit_vector = (v_bestEI - v_all[-1]) / dist
        vol_line, val_line, pinchoff_idx = tester.measure_dist_unit_vector(v_all[-1], unit_vector, d_r=d_step, max_r=dist)

        if pinchoff_idx != -1: # pinchoff detected
            v_next = vol_line[pinchoff_idx] - unit_vector*d_step # right before pinchoff
        else: # pinchoff not detected from the current v to v_bestEI
            v_next = vol_line[-1]

        # measurement
        v_all = np.concatenate((v_all,v_next[np.newaxis,:]), axis=0)
        d = tester.measure_dist_all_axis(v_all[-1] + stepback)
        d_all = np.concatenate((d_all, d[np.newaxis,:]), axis=0)

        # update the hypersurface model
        v_ext = calc_v_from_vd(v_all, d_all, step_back)
        h.set_data(v_ext, noise_var=(d_r/2.)**2)
    np.set_printoptions(precision=0)
    np.set_printoptions(suppress=True)
    print(v_all)
    v_bestL_all = np.array(v_bestL_all)
    print(v_bestL_all)
    v_bestEI_all = np.array(v_bestEI_all)
    print(v_bestEI_all)

    fig = plt.figure(dpi=300)
    ax = plt.gca()
    cs = ax.imshow(img, aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    fig.colorbar(cs)

    plt.scatter(v_all[:,1], v_all[:,0], marker='.', color='black')

    plt.xlim(lb[1], ub[1])
    plt.ylim(lb[0], ub[0])
    plt.savefig('temp_history')
    plt.close()

def draw_map(data, lb, ub, name):
    fig = plt.figure()
    ax = plt.gca()
    cs = ax.imshow(data, aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    fig.colorbar(cs)

    plt.xlim(lb[1], ub[1])
    plt.ylim(lb[0], ub[0])
    plt.savefig(name)
    plt.close()

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    main()
