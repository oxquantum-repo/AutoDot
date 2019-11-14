from pathlib import Path
import os
import time

import mkl
mkl.set_num_threads(8)

import numpy as np
from scipy.stats import norm, truncnorm

from sklearn.metrics import confusion_matrix,log_loss
import cma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import util
from test_common import Tester
from BO_common import random_hypersphere
from GPy_wrapper import GPyWrapper as GP
from GPy_wrapper import GPyWrapper_Classifier as GPC
import GP_util
#from GPflow_wrapper import GPflowWrapper as GP

import config
import config_model

import random_walk as rw

def main():
    conf_name = 'config1'
    if conf_name == 'config1':
        conf_func = config.config1
        ip = "http://129.67.86.107:8000/RPC2"
        save_dir = Path('./save_Dominic_redo_Basel2_correct')
        settletime, settletime_big = 0.01, 0.03 # settletime_big is for shuttling, 'None' disables shuttling
        # ['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
        #origin = -100.
        origin = 0.
        threshold_low = 0.0 # for pinchoff detector
        d_r = 10 # length of one step
    elif conf_name == 'config2':
        # ['c3', 'c4', 'c8', 'c10', 'c11', 'c12', 'c16']
        conf_func = config.config2
        ip = "http://129.67.85.235:8000/RPC2"
        save_dir = Path('./save_Florian_redo')
        settletime, settletime_big = 0.01, 0.03 # settletime_big is for shuttling, 'None' disables shuttling
        #origin = -100.
        origin = 0.
        settletime = 0.01
        threshold_low = 2.e-11
        d_r = 10 # length of one step
    elif conf_name == 'dummy':
        import pygor_dummy
        box_dim = 5
        box_a = -1000. * np.ones(box_dim)
        box_b = 1000. * np.ones(box_dim)
        shape = pygor_dummy.Box(ndim=box_dim, a=box_a, b=box_b)
        th_leak = np.array([-500., -400., -300., 0., 0.])
        shape = pygor_dummy.Leakage(shape, th_leak)
        save_dir = Path('./save_dummy')
        origin = -100.
        threshold_low = 0.2
        d_r = 10 # length of one step
    else:
        raise ValueError('Unsupported setup')
    save_dir.mkdir(exist_ok=True)

    if conf_name != 'dummy':
        pg, active_gate_idxs, lb_short, ub_short, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor(conf_func, ip, settletime, settletime_big)
    else:
        pg, active_gate_idxs, lb_short, ub_short, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor_dummy(shape)
    threshold_high = 0.8 * max_current

    active_gate_names = ["c{}".format(i+1) for i in active_gate_idxs]
    num_active_gates = len(active_gate_names)
    print(active_gate_names)

    # choose the origin
    if np.isscalar(origin):
        origin = origin*np.ones(num_active_gates)
    else:
        if len(origin) != num_active_gates:
            raise ValueError('Wrong array shape of origin.')

    # important algorithm parameters
    len_after_pinchoff=100

    detector_pinchoff = util.PinchoffDetectorThreshold(threshold_low) # pichoff detector
    detector_conducting = util.ConductingDetectorThreshold(threshold_high) # reverse direction
    # create a Callable object, input: unit_vector, output: distance between the boundary and the origin
    tester = Tester(pg, lb_short, ub_short, detector_pinchoff, d_r=d_r, len_after_pinchoff=len_after_pinchoff, logging=True, detector_conducting=detector_conducting, set_big_jump = set_big_jump, set_small_jump = set_small_jump)

    #do_extra_meas = lambda vols: config_model.do_extra_meas(pg, vols)
    do_extra_meas = None


    # load voltages of a previous experiment
    load_dir = Path('./save_Florian_randompoints')
    #load_dir = Path('./save_Dominic_hs')
    #load_dir = Path('./save_Dominic_randomHS_origin0_fixed')
    vols_prev = np.load(load_dir/'vols_poff.npy', allow_pickle=True)

    num_points = len(vols_prev)
    vols_poff_all = list()
    detected_all = list()
    for i, v in enumerate(vols_prev):
        v = convert_Basel2_to_Basel1(v)
        v_origin = v - origin
        u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        # Get measurements
        r, vols_pinchoff, found, t_firstjump = tester.get_r(u, origin=origin) # Measure the distance
        vols_poff_all.append(vols_pinchoff)
        detected_all.append(found)
        print(i)
        print(v)
        print(vols_pinchoff)

        np.array(vols_poff_all).dump(str(save_dir / 'vols_poff_after.npy'))
        np.array(vols_prev).dump(str(save_dir / 'vols_poff_prev.npy'))
        np.array(detected_all).dump(str(save_dir / 'detected_after.npy'))

def convert_Basel2_to_Basel1(v):
    #Basel1: [nose, r barrier, r wall, r plunger, m plunger, l plunger, l wall, l barrier]
    #Basel2: [nose, r barrier, r wall, r plunger, m plunger, l wall,  l barrier]
    return np.append( v[:5], [0., v[5], v[6]] )

if __name__ == "__main__":
    main()
