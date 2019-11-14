from pathlib import Path
import os
import time
import sys

import numpy as np
from scipy.stats import norm, truncnorm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from test_common import Tester
import pygor_dummy
import util

def main():
    box_dim = 2
    # Box with leakage
    #box_a = -1000. * np.ones(box_dim)
    #box_b = 500. * np.ones(box_dim)
    #shape = pygor_dummy.Box(ndim=box_dim, a=box_a, b=box_b)
    #th_leak = -500.*np.ones(2)
    #shape = pygor_dummy.Leakage(shape, th_leak)

    # Box with a free lower vertex
    origin = -100.
    box_a = -1000. * np.ones(box_dim)
    box_a_prime = -700. * np.ones(box_dim)
    box_b = 500. * np.ones(box_dim)
    shape = pygor_dummy.Box_FreeLowerVertex(box_dim, box_a, box_b, box_a_prime)
    # Optional Leakage
    th_leak = -300.*np.ones(2)
    shape = pygor_dummy.Leakage(shape, th_leak)

    pg, active_gate_idxs, lb_short, ub_short, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor_dummy(shape)


    step_back = 100 # important param for measuring d
    len_after_pinchoff=100
    threshold_high = 0.8
    threshold_low = 0.2
    d_r = 10

    detector_pinchoff = util.PinchoffDetectorThreshold(threshold_low) # pichoff detector
    detector_conducting = util.ConductingDetectorThreshold(threshold_high) # reverse direction
    tester = Tester(pg, lb_short, ub_short, detector_pinchoff, d_r=d_r, len_after_pinchoff=len_after_pinchoff, logging=True, detector_conducting=detector_conducting, set_big_jump = set_big_jump, set_small_jump = set_small_jump)

    num_points = 100
    u_all = np.zeros((num_points,2))
    u_all[:,0] = -np.linspace(0.,1.,num_points)
    u_all[:,1] = -np.sqrt(1.0 - np.square(u_all[:,0]))
    
    u_all, r_all, d_all, poff_all, detected_all, time_all, extra_meas_all = get_u_init(u_all, tester, step_back, origin=origin)

    plt.figure()
    u_simp = -np.square(u_all[:,0])
    valid = np.logical_and(poff_all[:,0],poff_all[:,1])
    plt.plot(u_simp[valid], d_all[valid,0], color='C0')
    plt.plot(u_simp[valid], d_all[valid,1], color='C1')

    #plt.plot(u_simp[poff_all[:,0]], d_all[poff_all[:,0],0], color='C0')
    #plt.plot(u_simp[poff_all[:,1]], d_all[poff_all[:,1],1], color='C1')

    #plt.plot(u_simp, d_all[:,0], color='C0')
    #plt.plot(u_simp, d_all[:,1], color='C1')
    plt.savefig('dist')
    plt.close()


    lb = lb_short
    ub = ub_short
    resol = 100
    v1_grid = np.linspace(lb[0], ub[0], resol)
    v2_grid = np.linspace(lb[1], ub[1], resol)
    V2, V1 = np.meshgrid(v2_grid, v1_grid)
    v_test = np.array([V1.ravel(), V2.ravel()]).transpose()
    inside = shape(v_test)
    plt.figure()
    plt.imshow(inside.reshape(V1.shape), cmap='RdBu_r', aspect='equal', origin='lower', extent=[lb[1], ub[1], lb[0], ub[0]])
    plt.savefig('shape')
    plt.close()

def get_u_init(u_all, tester, step_back, origin=0.0, penalty_non_poff=0.0, do_extra_meas=None, save_dir=None):
    # Latin hypercube sampling (for initial measurement)

    r_all = list()
    d_all = list()
    poff_all = list()
    detected_all = list()
    time_all = list()
    extra_meas_all = list()
    for i, u in enumerate(u_all):
        print('Initial random iteration: ', i)
        print(u)

        t = time.time()
        r, vols_pinchoff, found = tester.get_r(u, origin=origin) # Measure the distance
        d_vec, poff_vec, meas_each_axis, vols_each_axis = tester.measure_dist_all_axis(vols_pinchoff+step_back, penalty_non_poff=penalty_non_poff)

        r_all.append(r)
        detected_all.append(found)
        d_all.append(d_vec)
        poff_all.append(poff_vec)
        print('vols: ', vols_pinchoff, 'd_all: ', d_all[-1])
        if do_extra_meas is not None:
            extra_meas_all.append(meas_each_axis+do_extra_meas(vols_pinchoff))
        if save_dir is not None:
            save(save_dir, np.array(u_all), [], np.array(r_all), np.array(d_all), np.array(poff_all), detected_all, tester.logger, extra_meas_all, np.array([]), time_all, [])
        
        elapsed = time.time() - t
        print('Elapsed time: ', elapsed)
        time_all.append(elapsed)

    r_all = np.array(r_all)
    d_all = np.array(d_all)
    poff_all = np.array(poff_all)
    return u_all, r_all, d_all, poff_all, detected_all, time_all, extra_meas_all
if __name__ == "__main__":
    main()
