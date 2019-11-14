from pathlib import Path
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Pygor_new import Experiment
import pygor_fixvol
import util, plot_util
from test_common import Tester, translate_window_inside_boundary
import BO_common

import plot_util

import config

def scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb_g1, lb_g2, ub_g1, ub_g2, name_append, pic_dpi=None, mult1=1., mult2=1., save_dir=None):
    pg.set_params(vols.tolist())
    #idx_c5 = active_gate_names.index("c5")
    #idx_c9 = active_gate_names.index("c9")

    y_center = pg.getval(name_g1)
    x_center = pg.getval(name_g2)
    w_from = np.array([y_center-w1, x_center-w2])
    w_to = np.array([y_center+w1, x_center+w2])
    lb_w = np.array([lb_g1, lb_g2])
    ub_w = np.array([ub_g1, ub_g2])
    w_from, w_to = translate_window_inside_boundary(w_from, w_to, lb_w, ub_w)
    t = time.time()
    img=pg.do2d(name_g1,float(w_from[0]),float(w_to[0]),int(resol),
                name_g2,float(w_from[1]),float(w_to[1]),int(resol))
    print('Elapsed time: ', time.time()-t)
    img = np.array(img.data) # array
    print(img.shape)
    fname = 'scan{}{}_{}'.format(name_g1,name_g2,name_append)
    np.savez(save_dir/fname, w_from=w_from, w_to=w_to, img=img)
    fig=plt.figure(dpi=pic_dpi)
    ax_img = plt.imshow(img.reshape((resol,resol)), cmap='RdBu_r', extent=[w_from[1]*mult2, w_to[1]*mult2, w_from[0]*mult1, w_to[0]*mult1], origin='lower',vmin=None,vmax=None)
    fig.colorbar(ax_img)
    plt.savefig(str(save_dir/fname))
    plt.close()
    return img

def scan2d_fromto(pg, vols, name_g1, name_g2, resol, g1_from, g2_from, g1_to, g2_to, name_append, pic_dpi=None):
    t = time.time()
    pg.set_params(vols.tolist())
    img=pg.do2d(name_g1,float(g1_from),float(g1_to),int(resol),
                name_g2,float(g2_from),float(g2_to),int(resol))
    print('Elapsed time: ', time.time()-t)
    img = img.data # array
    fname = 'scan{}{}_{}'.format(name_g1,name_g2,name_append)
    w_from = [g1_from, g2_from]
    w_to = [g1_to, g2_to]
    np.savez(save_dir/fname, w_from=w_from, w_to=w_to, img=img)
    plot_util.plot_2d_map(img[0], str(save_dir/fname), [w_from[1], w_to[1], w_from[0], w_to[0]], colorbar=True, dpi=pic_dpi)

def main():
    conf_name = 'config1'
    if conf_name == 'config1':
        conf_func = config.config1
        ip = "http://129.67.86.107:8000/RPC2"
        save_dir = Path('./save_Dominic_ABL_group_optparam_run2/135_2')
        settletime, settletime_big = 0.01, 0.03 # settletime_big is for shuttling, 'None' disables shuttling
        # active gates = ['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
        threshold_low = 0.0 # for pinchoff detector
        d_r = 10 # length of one step
    elif conf_name == 'config2':
        conf_func = config.config2
        ip = "http://129.67.85.235:8000/RPC2"
        save_dir = Path('./save_Florian_gpc2_run7')
        settletime, settletime_big = 0.01, 0.03 # settletime_big is for shuttling, 'None' disables shuttling
        threshold_low = 2.e-11
        d_r = 10 # length of one step
    elif conf_name == 'dummy':
        import pygor_dummy
        box_dim = 5
        box_a = -500. * np.ones(box_dim)
        box_b = 500. * np.ones(box_dim)
        shape = pygor_dummy.Box(ndim=box_dim, a=box_a, b=box_b)
        save_dir = Path('./save_dummy')
        threshold_low = 0.2
        d_r = 10 # length of one step
    else:
        raise ValueError('Unsupported setup')
    # pygor
    if conf_name != 'dummy':
        pg, active_gate_idxs, lb, ub, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor(conf_func, ip, settletime, settletime_big)
    else:
        pg, active_gate_idxs, lb, ub, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor_dummy(shape)
    active_gate_names = ["c{}".format(i+1) for i in active_gate_idxs]
    num_active_gates = len(active_gate_names)
    print(active_gate_names)

    #idx_good = [17, 84, 208, 213, 229, 336, 380, 401, 442, 510, 520]
    #idx_good = [108, 117, 161, 263, 273, 301, 405, 428, 461]
    #idx_good = [115, 129, 145, 162, 172, 194, 226, 261, 275, 281, 333, 343, 368, 403]
    #idx_good = [115, 129, 145, 162, 172, 194, 226, 261, 275, 281, 333, 343, 368, 403]
    #idx_good = [25, 61, 110, 120, 135, 228, 368, 415, 434, 446, 491]
    #idx_good = [63, 110, 118, 130, 142, 231, 254, 301, 324, 373, 375, 384, 419, 435]
    #idx_good = [74, 81, 145, 161, 498, 547, 722, 760, 843]
    idx_good = [2, 36, 58, 95, 105, 109, 110, 115, 135, 145, 203, 271]
    vols = np.load(save_dir/'vols_poff.npy', allow_pickle=True)

    vol_list = list()
    name_list = list()

    #name_list.append('28_1')
    #vol_list.append(np.array([ -686.98787658, -1068.10040637,  -420, -1814.13833536, -1782.99333658, -1842.60789763,  -500, -1173.39034208]))
    #name_list.append('142_2')
    #vol_list.append(np.array([ -763.44723453, -1079.39689593,  -420, -1881.75335181, -1846.69798366, -1732.91877179,  -420, -1036.52360957]))
    #name_list.append('142_2')
    #vol_list.append(np.array([ -763.44723453, -1079.39689593,  -480, -1881.75335181, -1846.69798366, -1732.91877179,  -420, -1036.52360957]))
    #name_list.append('36_1')
    #vol_list.append(np.array([ -900.86410779,  -865.64664275,  -380, -1630.33211249, -1850.53930075, -1973.40024467,  -420,  -937.07388057]))
    #name_list.append('1_1')
    #vol_list.append(np.array([ -856.04412475,  -937.16103294,  -520, -1648.0176585 , -1688.79801685, -1658.93619951,  -410,  -932.04616637]))
    #name_list.append('1_2')
    #vol_list.append(np.array([ -856.04412475,  -937.16103294,  -500, -1648.0176585 , -1688.79801685, -1658.93619951,  -480,  -932.04616637]))
    #name_list.append('13_1')
    #vol_list.append(np.array([ -852.91749095,  -989.38046736,  -460, -1719.22163011, -1703.85973708, -1860.66172595,  -350,  -986.43953877]))
    #name_list.append('1_3')
    #vol_list.append(np.array([ -856.04412475,  -937.16103294,  -480, -1648.0176585 , -1688.79801685, -1658.93619951,  -540,  -932.04616637]))
    #name_list.append('13_2')
    #vol_list.append(np.array([ -852.91749095,  -989.38046736,  -450, -1719.22163011, -1703.85973708, -1860.66172595,  -420,  -986.43953877]))
    #name_list.append('1_4')
    #vol_list.append(np.array([ -856.04412475,  -937.16103294,  -460, -1648.0176585 , -1688.79801685, -1658.93619951,  -610,  -932.04616637]))
    #name_list.append('13_3')
    #vol_list.append(np.array([ -852.91749095,  -989.38046736,  -440, -1719.22163011, -1703.85973708, -1860.66172595,  -490,  -986.43953877]))
    #name_list.append('1_5')
    #vol_list.append(np.array([ -856.04412475,  -937.16103294,  -450, -1648.0176585 , -1688.79801685, -1658.93619951,  -690,  -932.04616637]))
    #name_list.append('13_4')
    #vol_list.append(np.array([ -852.91749095,  -989.38046736,  -420, -1719.22163011, -1703.85973708, -1860.66172595,  -560,  -986.43953877]))
    name_list.append('13_5')
    vol_list.append(np.array([ -852.91749095,  -989.38046736,  -380, -1719.22163011, -1703.85973708, -1860.66172595,  -620,  -986.43953877]))

    '''
    new_wires = [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 1, 1, 0, 0]]
    new_wires = np.array(new_wires)
    for idx in idx_good:
        name_list.append(str(idx) + '_')
        #vol_list.append(vols[idx])
        vol_list.append(np.matmul(new_wires.T, vols[idx]))
    '''

    '''
    resol = 128
    w1 = w2 = 50
    #name_g1, name_g2 = 'c8', 'c12'
    #name_g1, name_g2 = 'c5', 'c9'
    name_g1, name_g2 = 'c5', 'c9'

    idx_g1, idx_g2 = active_gate_names.index(name_g1), active_gate_names.index(name_g2)
    lb_g1, lb_g2 = lb[idx_g1], lb[idx_g2]
    ub_g1, ub_g2 = ub[idx_g1], ub[idx_g2]
    vols= np.array([-1150.9539658 ,  -808.79439154,  -632.5, -1815.3845539 , -1127.79332466, -1406.53212482,  -985.,  -682.98769082])
    scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb_g1, lb_g2, ub_g1, ub_g2, '1_43_bigwindow', pic_dpi=None, mult1=1., mult2=1., save_dir=save_dir)
    '''

    resol = 128
    w1, w2 = 75, 75
    name_g1, name_g2 = 'c5', 'c9'
    idx_g1, idx_g2 = active_gate_names.index(name_g1), active_gate_names.index(name_g2)
    lb_g1, lb_g2 = lb[idx_g1], lb[idx_g2]
    ub_g1, ub_g2 = ub[idx_g1], ub[idx_g2]

    for v, name in zip(vol_list,name_list):
        scan2d(pg, name_g1, name_g2, resol, v, w1, w2, lb_g1, lb_g2, ub_g1, ub_g2, name, pic_dpi=None, mult1=1., mult2=1., save_dir=save_dir)

    '''
    #vols= np.array([-1150.9539658 ,  -808.79439154,  -630., -1815.3845539 , -1127.79332466, -1406.53212482,  -983.5,  -682.98769082])
    vols= np.array([-1150.9539658 ,  -808,  -630., -1815.3845539 , -1127.79332466, -1406.53212482,  -983.5,  -682.98769082])
    scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb_g1, lb_g2, ub_g1, ub_g2, '1_43_c4_808', pic_dpi=None, mult1=1., mult2=1., save_dir=save_dir)

    vols= np.array([-1150.9539658 ,  -805,  -630., -1815.3845539 , -1127.79332466, -1406.53212482,  -983.5,  -682.98769082])
    scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb_g1, lb_g2, ub_g1, ub_g2, '1_43_c4_805', pic_dpi=None, mult1=1., mult2=1., save_dir=save_dir)

    vols= np.array([-1150.9539658 ,  -802,  -630., -1815.3845539 , -1127.79332466, -1406.53212482,  -983.5,  -682.98769082])
    scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb_g1, lb_g2, ub_g1, ub_g2, '1_43_c4_802', pic_dpi=None, mult1=1., mult2=1., save_dir=save_dir)
    '''


if __name__  == "__main__":
    main()
