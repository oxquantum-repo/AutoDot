from pathlib import Path
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import util
from test_common import Tester
from BO_common import random_hypersphere

import config
import config_model

import measure_2d

def main():
    conf_name = 'config2'
    if conf_name == 'config1':
        conf_func = config.config1
        ip = "http://129.67.86.107:8000/RPC2"
        save_dir = Path('./save_Dominic_ABL_group_optparam_run2/135_2')
        settletime, settletime_big = 0.01, 0.03 # settletime_big is for shuttling, 'None' disables shuttling
        # ['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
        threshold_low = 0.0 # for pinchoff detector
        d_r = 10 # length of one step
    elif conf_name == 'config2':
        # ['c3', 'c4', 'c8', 'c10', 'c11', 'c12', 'c16']
        conf_func = config.config2
        #ip = "http://129.67.85.235:8000/RPC2"
        ip = 'http://129.67.85.38:8000/RPC2'
        save_dir = Path('./save_Basel2_group/105')
        settletime, settletime_big = 0.02, 0.02 # settletime_big is for shuttling, 'None' disables shuttling
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


    # important algorithm parameters
    len_after_pinchoff=100

    detector_pinchoff = util.PinchoffDetectorThreshold(threshold_low) # pichoff detector
    detector_conducting = util.ConductingDetectorThreshold(threshold_high) # reverse direction
    # create a Callable object, input: unit_vector, output: distance between the boundary and the origin
    tester = Tester(pg, lb_short, ub_short, detector_pinchoff, d_r=d_r, len_after_pinchoff=len_after_pinchoff, logging=True, detector_conducting=detector_conducting, set_big_jump = set_big_jump, set_small_jump = set_small_jump)

    #do_extra_meas = lambda vols, th: config_model.do_extra_meas(pg, vols, th)
    do_extra_meas = None

    #v_target = np.array([-1877.11448379,  -315.09271412,  -588.03713859, -1743.6191362 , -587.8001855 ,  -693.38667762,  -832.91995001,  -186.99101894])
    #new_wires = [[1, 0, 0, 0, 0, 0, 0, 0],
         #[0, 1, 0, 0, 0, 0, 0, 1],
         #[0, 0, 1, 0, 0, 0, 1, 0],
         #[0, 0, 0, 1, 1, 1, 0, 0]]
    new_wires = [[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 1, 1, 0, 0]]
    new_wires = np.array(new_wires)

    v_target = np.array([-1290.10731323, -1521.90819125,  -813.40589252, -1069.5955569 ])
    v_target = np.matmul(new_wires.T, v_target)
    origin = np.minimum(v_target + 200., 0.)
    print('Origin: ', origin)

    np.array(v_target).dump(str(save_dir/'v_local.npy'))
    np.array(origin).dump(str(save_dir/'origin_local.npy'))

    num_trials = 500
    sigma = 100.
    #name_g1, name_g2 = 'c5', 'c9'
    name_g1, name_g2 = 'c8', 'c12'
    resol = 64
    w1 = w2 = 75 
    idx_g1, idx_g2 = active_gate_names.index(name_g1), active_gate_names.index(name_g2)
    lb_g1, lb_g2 = lb_short[idx_g1], lb_short[idx_g2]
    ub_g1, ub_g2 = ub_short[idx_g1], ub_short[idx_g2]

    v_all = list()
    meas_all = list()
    time_all = list()
    for i in range(num_trials):
        '''
        delta = np.random.normal(scale=sigma, size=num_active_gates)
        v_origin = v_target + delta - origin
        print('Random vol:', v_origin+origin)
        u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        '''
        u = random_hypersphere(num_active_gates, 1)[0]
        # Get measurements
        t = time.time()
        r, vols_pinchoff, found, t_firstjump = tester.get_r(u, origin=origin) # Measure the distance
        t1 = time.time() - t
        print('Poff at ', vols_pinchoff)
        if found:
            # measurement
            print('Found')
            meas = measure_2d.scan2d(pg, name_g1, name_g2, resol, vols_pinchoff, w1, w2, lb_g1, lb_g2, ub_g1, ub_g2, str(i), pic_dpi=None, mult1=1., mult2=1., save_dir=save_dir)
            '''
            meas = do_extra_meas(vols_pinchoff, 0.0)
            # highres = meas[-6] if len(meas) > 10 else None
            # lowres = meas[4] if len(meas) > 4 else None
            if lowres:
                plt.imsave(str(save_dir/(str(i)+'lowres.png')), lowres, origin='lower', cmap='viridis')
            if highres:
                plt.imsave(str(save_dir/(str(i)+'highres.png')), highres, origin='lower', cmap='viridis')

            '''
        else:
            print('Not found')
            meas = []
        t2 = time.time() - t

        time_all.append((t1, t2))

        v_all.append(vols_pinchoff)
        meas_all.append(meas)
        np.array(v_all).dump(str(save_dir/'vols_poff.npy'))
        #np.array(meas_all).dump(str(save_dir/'meas.npy'))
        np.save(str(save_dir / 'meas'), meas_all)
        np.array(time_all).dump(str(save_dir/'time.npy'))

if __name__ == "__main__":
    main()
