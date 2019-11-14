from pathlib import Path
import os, sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Pygor_new import Experiment
import pygor_fixvol
from test_common import translate_window_inside_boundary
import util
from test_common import Tester

import config

conf_name = 'config1'
if conf_name == 'config1':
    conf_func = config.config1
    ip = "http://129.67.86.107:8000/RPC2"
    save_dir = Path('./save_Dominic14')
    settletime, settletime_big = 0.01, 0.03 # settletime_big is for shuttling, 'None' disables shuttling
    # active gates = ['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
    #origin = np.array([-100., -850., -850., -100., -100., -100., -100., -100.])
    origin = -100.
    threshold_low = 0.0 # for pinchoff detector
    d_r = 10 # length of one step
elif conf_name == 'config2':
    conf_func = config.config2
    ip = "http://129.67.85.235:8000/RPC2"
    save_dir = Path('./save_YutianFlorian8')
    settletime, settletime_big = 0.01, None # settletime_big is for shuttling, 'None' disables shuttling
    origin = -100.
    settletime = 0.01
    threshold_low = 5.e-11
    d_r = 10 # length of one step
elif conf_name == 'dummy':
    import pygor_dummy
    box_dim = 5
    box_a = -500. * np.ones(box_dim)
    box_b = 500. * np.ones(box_dim)
    shape = pygor_dummy.Box(ndim=box_dim, a=box_a, b=box_b)
    save_dir = Path('./save_dummy')
    origin = -100.
    threshold_low = 0.2
    d_r = 10 # length of one step
else:
    raise ValueError('Unsupported setup')
save_dir.mkdir(exist_ok=True)

origin = np.load(save_dir / 'origin.npy')
#origin = np.zeros(8)
vols_all = np.load(save_dir / 'vols.npy')
d_all = np.load(save_dir / 'dist_surface.npy')
pinchoff_idx = np.load(save_dir / 'pinchoff_idx.npy')

d_tot = np.array(np.load(save_dir/'obj_val.npy'))
#d_tot = np.sqrt(np.sum(np.square(d_all), axis=1)) #np.sum(d_all,axis=1)
sort_idxs = np.argsort(d_tot)

vols_all = vols_all[sort_idxs]
d_all = d_all[sort_idxs]
d_tot = d_tot[sort_idxs]
pinchoff_idx = pinchoff_idx[sort_idxs]

vols_pinchoff_all = list()
for i in range(sort_idxs.size):
    print(d_tot[i])
    vols_pinchoff_all.append(vols_all[i][pinchoff_idx[i]])
np.savetxt(save_dir/'vols_pinchoff_sorted.csv', np.array(vols_pinchoff_all), delimiter=',', fmt='%4.2f')
np.savetxt(save_dir/'d_sorted.csv', d_all, delimiter=',', fmt='%4.2f')
np.savetxt(save_dir/'dtot_sorted.csv', d_tot[:,np.newaxis], delimiter=',', fmt='%4.2f')

#sys.exit()


# pygor
if conf_name != 'dummy':
    pg, active_gate_idxs, lb, ub, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor(conf_func, ip, settletime, settletime_big)
else:
    pg, active_gate_idxs, lb, ub, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor_dummy(shape)
active_gate_names = ["c{}".format(i+1) for i in active_gate_idxs]
num_active_gates = len(active_gate_names)
print(active_gate_names)
# choose the origin
if np.isscalar(origin):
    origin = origin*np.ones(num_active_gates)
else:
    if len(origin) != num_active_gates:
        raise ValueError('Wrong array shape of origin.')


# pinchoff detector
find_pinchoff_agian = False
if find_pinchoff_agian:
    threshold = 0.2 * (max_current - min_current) + min_current
    d_r = 2.5 # length of one step
    len_after_pinchoff=400
    detector_pinchoff = util.PinchoffDetectorThreshold(threshold)
    tester = Tester(pg, lb, ub, detector_pinchoff, d_r=d_r, len_after_pinchoff=len_after_pinchoff, logging=False)

def scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb_g1, lb_g2, ub_g1, ub_g2, name_append, pic_dpi=None, mult1=1., mult2=1.):
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
    img = img.data # array
    fname = 'scan{}{}_{}'.format(name_g1,name_g2,name_append)
    np.savez(save_dir/fname, w_from=w_from, w_to=w_to, img=img)
    fig=plt.figure(dpi=pic_dpi)
    ax_img = plt.imshow(img[0], cmap='RdBu_r', extent=[w_from[1]*mult2, w_to[1]*mult2, w_from[0]*mult1, w_to[0]*mult1], origin='lower',vmin=None,vmax=None)
    fig.colorbar(ax_img)
    plt.savefig(str(save_dir/fname))
    plt.close()


counter = 0
vols_measured = list()
for i in range(sort_idxs.size):
    print(d_tot[i])
    #if d_tot[i] > 2000:
    #    break

    vols = vols_all[i][pinchoff_idx[i]]

    if find_pinchoff_agian:
        u = vols / np.sqrt(np.sum(np.square(vols)))
        r, vols_pinchoff, found = tester.get_r(u)
        vols = vols_pinchoff

    too_close = False
    for vols_ in vols_measured:
        if np.all(np.fabs(np.array(vols) - np.array(vols_))<100.0):
            too_close = True
    if too_close:
        continue
    if d_tot[i] < 100.:
        continue

    print(vols)
    vols_measured.append(vols)

    #w = 150
    #resol = 128
    #w = 100
    #resol = 128
    #name_g1, name_g2 = 'c5', 'c11'
    #idx_g1 = active_gate_names.index(name_g1)
    #idx_g2 = active_gate_names.index(name_g2)

    #scan2d(pg, name_g1, name_g2, resol, vols, w, lb[idx_g1], lb[idx_g2], ub[idx_g1], ub[idx_g2], str(i)+'_'+str(resol),pic_dpi=150)


    w1, w2 = 100., 100.
    mult1, mult2 = 1., 1.
    resol = 128
    #name_g1, name_g2 = 'c8', 'c12'
    #name_g1, name_g2 = 'c4', 'c16'
    name_g1, name_g2 = 'c5', 'c9'
    idx_g1 = active_gate_names.index(name_g1)
    idx_g2 = active_gate_names.index(name_g2)

    scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb[idx_g1], lb[idx_g2], ub[idx_g1], ub[idx_g2], str(i)+'_'+str(resol), pic_dpi=200, mult1=mult1, mult2=mult2)


    '''
    w1, w2 = 200., 200.
    mult1, mult2 = 1., 1.
    name_g1, name_g2 = 'c7', 'c9'
    idx_g1 = active_gate_names.index(name_g1)
    idx_g2 = active_gate_names.index(name_g2)

    scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb[idx_g1], lb[idx_g2], ub[idx_g1], ub[idx_g2], str(i)+'_'+str(resol), pic_dpi=150, mult1=mult1, mult2=mult2)

    w1, w2 = 40., 200.
    mult1, mult2 = 5., 1.
    name_g1, name_g2 = 'c6', 'c9'
    idx_g1 = active_gate_names.index(name_g1)
    idx_g2 = active_gate_names.index(name_g2)

    scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb[idx_g1], lb[idx_g2], ub[idx_g1], ub[idx_g2], str(i)+'_'+str(resol), pic_dpi=150, mult1=mult1, mult2=mult2)

    w1, w2 = 200., 200.
    mult1, mult2 = 1., 1.
    name_g1, name_g2 = 'c7', 'c10'
    idx_g1 = active_gate_names.index(name_g1)
    idx_g2 = active_gate_names.index(name_g2)

    scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb[idx_g1], lb[idx_g2], ub[idx_g1], ub[idx_g2], str(i)+'_'+str(resol), pic_dpi=150, mult1=mult1, mult2=mult2)
    '''

print(len(vols_measured))
print(vols_measured)
