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

save_dir = Path('./savexxxx')

# pygor
pygor = Experiment(savedata=False, plot=False, fast=True, xmlip="http://129.67.85.235:8000/RPC2")
print('After initialization')

print(pygor.server.system.listMethods())
#print(pygor.server.hello())

print(pygor.get_params()) # return the current voltages
# get server configurations
print(pygor.server.config_control('dac',{}))
print(pygor.server.config_aquisition('delft1',{}))
pygor.server.config_aquisition('delft1',{'set_active':[True,False,False]})
print(pygor.server.config_aquisition('delft1',{}))

pygor.server.config_control('dac',{'set_settletime':0.1})
# current when all voltages = 0.0
pygor.set_params([0.0]*16)

sys.exit(0)


time.sleep(1)
print(pygor.do0d())
noise_current = pygor.do0d()[0] # first channel
print('Current when voltages are 0.0: ', noise_current)

#gates = ['c{}'.format(i) for i in range(3,13)]
gates = ['c10']
print(gates)

for gate in gates:
    pygor.set_params([0.0]*16)
    pygor.setval('c1', 200)
    temp = pygor.do1d(gate, -2000, 0, 100)
    plt.figure()
    plt.plot(temp.data[0])
    plt.savefig('line_{}'.format(gate))
    plt.close()

sys.exit(0)

temp = pygor.do1d('c9', 170, 0, 100)
print(temp.data[0])
plt.figure()
plt.plot(temp.data[0])
plt.savefig('line')
plt.close()
sys.exit(0)


pg = pygor_fixvol.PygorFixVol(pygor)
fixed_gate_idxs = [0,1,10,11,12,13,14,15]
active_gate_idxs = [i for i in range(16) if i not in fixed_gate_idxs]
fixed_gate_names = ["c{}".format(i+1) for i in fixed_gate_idxs]
fixed_vals = [18.0] + [0.0]*(len(fixed_gate_names) -1)
fixed_dict = dict(zip(fixed_gate_names,fixed_vals))
pg.fix_vols(fixed_dict)

active_gate_names = ["c{}".format(i+1) for i in active_gate_idxs]

# get safe ranges
lims = pygor.server.config_control('dac',{})['set_lims']
ub, lb = np.array(lims[0]), np.array(lims[1])
ub_short, lb_short = ub[active_gate_idxs], lb[active_gate_idxs]

# max current
num_active_gates = len(active_gate_idxs)
pg.set_params([0.0]*num_active_gates)
time.sleep(1)
max_current = pg.do0d()[0] # first channel
print(max_current)

def scan2d(pg, name_g1, name_g2, resol, vols, w, lb_g1, lb_g2, ub_g1, ub_g2, name_append, pic_dpi=None):
    pg.set_params(vols.tolist())
    #idx_c5 = active_gate_names.index("c5")
    #idx_c9 = active_gate_names.index("c9")

    y_center = pg.getval(name_g1)
    x_center = pg.getval(name_g2)
    w_from = np.array([y_center-w, x_center-w])
    w_to = np.array([y_center+w, x_center+w])
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
    plt.figure(dpi=pic_dpi)
    plt.imshow(img[0], cmap='RdBu_r', extent=[w_from[1], w_to[1], w_from[0], w_to[0]], origin='lower',vmin=0.0,vmax=max_current)
    plt.savefig(str(save_dir/fname))
    plt.close()

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
    plt.figure(dpi=pic_dpi)
    ax = plt.imshow(img[0], cmap='RdBu_r', extent=[w_from[1], w_to[1], w_from[0], w_to[0]], origin='lower',vmin=0.0,vmax=max_current, aspect='equal')
    plt.savefig(str(save_dir/fname))
    plt.close()

vols = np.array([-1241.72400121,  -447.09580435,  -447.09580435, -1765.40770536,
                 -1959.98350845,  -447.09595741,  -610.28109534,  -447.09580435])
# reset the center
vols[2] = (-425.0 -375.0)/2.0 # c5 (right wall)
vols[6] = (-710.0 -660.0)/2.0 # c9 (left wall)

w = 25
resol = 256
pg.setval('c1', 3.0)
scan2d_fromto(pg, vols, 'c5', 'c9', resol, -430.0, -710.0, -380.0, -680.0, 'smallwindow_bias3_'+str(resol), pic_dpi=150)
