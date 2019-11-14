import sys
import time
import numpy as np
import pygor_fixvol
from Pygor_new import Experiment

def fixvols(pg, fixed_gate_idxs, fixed_vals = None):
    if not isinstance(pg, pygor_fixvol.PygorFixVol):
        raise ValueError('pg should be an instance of PygorFixVol')
    fixed_gate_names = ["c{}".format(i+1) for i in fixed_gate_idxs]
    if fixed_vals is None:
        fixed_vals = [0.0]*len(fixed_gate_names)
    else:
        if len(fixed_vals) != len(fixed_gate_names):
            raise ValueError('The number of gate names and values mismatch.')
        if isinstance(fixed_vals, np.ndarray):
            fixed_vals = fixed_vals.tolist()
    fixed_dict = dict(zip(fixed_gate_names,fixed_vals))
    pg.fix_vols(fixed_dict)

def config(pg, fixed_gate_idxs, ub_arr, lb_arr):
    fixvols(pg, fixed_gate_idxs)

    active_gate_idxs = [i for i in range(16) if i not in fixed_gate_idxs]

    dac_settings = {'set_lims': [ub_arr.tolist(), lb_arr.tolist()]}
    if isinstance(pg, pygor_fixvol.PygorFixVol): # wrapper
        server = pg.pygor.server
    else: # original pygor
        server = pg.server
    server.config_control('dac', dac_settings) # set
    lims = server.config_control('dac',{})['set_lims'] # get

    print('Bounds:', lims)
    ub, lb = np.array(lims[0]), np.array(lims[1])
    ub_short, lb_short = ub[active_gate_idxs], lb[active_gate_idxs]

    return active_gate_idxs, lb_short, ub_short

def do_nothing(*args):
    return 
def jump_mode(pygor, settletime, shuttle=False):
    #pygor.server.config_control('dac',{'set_settletime':settletime, 'set_shuttle':shuttle})
    pygor.server.config_control('dac',{'set_settletime':settletime, 'set_shuttle':True})
    return

def config1(pg):
    # fix some gates and set lower and upper bounds
    fixed_gate_idxs = [0,1,10,11,12,13,14,15]

    ub_arr = np.zeros(16)
    ub_arr[0:2] = 500.0
    lb_arr = -2000.0*np.ones(16)
    lb_arr[0:2] = -500.0

    #default_vols = {'c1':14.179}
    default_vols = {'c1':30.}

    active_gate_idxs, lb_short, ub_short = config(pg, fixed_gate_idxs, ub_arr, lb_arr)

    return active_gate_idxs, lb_short, ub_short, default_vols

def config2(pg):
    # fix some gates and set lower and upper bounds
    #fixed_gate_idxs = [0,1,8,13]
    fixed_gate_idxs = [0,1,4,5,6,8,12,13,14]
    fixvols(pg, fixed_gate_idxs)

    ub_arr = 0. * np.ones(16) # upper bound for the voltages
    ub_arr[0] = 500.
    ub_arr[8], ub_arr[13] = 0., 0.

    lb_arr = -1700.0*np.ones(16) # lower bound
    lb_arr[0] = -500
    lb_arr[8], lb_arr[13] = 0., 0.

    default_vols = {'c1':40}
    '''
    for i in range(2,16):
        if i not in fixed_gate_idxs:
            default_vols['c'+str(i+1)] = 0. # origin
    '''

    active_gate_idxs, lb_short, ub_short = config(pg, fixed_gate_idxs, ub_arr, lb_arr)

    return active_gate_idxs, lb_short, ub_short, default_vols

def setup_pygor(conf_func, ip, settletime, settletime_big=None):
    # Pygor
    print('Pygor setup')
    pygor = Experiment(mode='none', xmlip=ip)
    print(pygor.server.system.listMethods())
    print(pygor.get_params()) # return the current voltages
    # get server configurations
    print(pygor.server.config_control('dac',{}))
    #print(pygor.server.config_aquisition('delft1',{}))
    #print(pygor.server.config_aquisition('delft1',{'set_active': [True,False,False]}))
    print(pygor.server.config_aquisition('alazar1',{}))
    print(pygor.server.config_aquisition('alazar1',{'set_active': [False,True]}))
    pygor.server.config_control('dac',{'set_settletime':settletime})
    #pygor.set_params([0.0]*16)
    #sys.exit(0)

    # thin pygor wrapper to hide constant voltages
    pg = pygor_fixvol.PygorFixVol(pygor)
    active_gate_idxs, lb_short, ub_short, default_vols = conf_func(pg)

    # set active voltages to 0.0
    num_active_gates = len(active_gate_idxs)
    print('the number of active gates: ', num_active_gates)
    #pg.set_params([0.0]*num_active_gates)

    #pg.setval('c1', 0.0)
    #time.sleep(1)
    #min_current = pg.do0d()[0][0] # [0]: first channel
    #print(pygor.get_params()) # check the full-length voltages
    #print('Current when voltages including bias are 0.0: ', min_current)

    for gate, val in default_vols.items():
        pg.setval(gate, val)

    pg.setval('c1', 0.0)
    time.sleep(1)
    min_current = pg.do0d()[0][0] # [0]: first channel
    print(pygor.get_params()) # check the full-length voltages
    print('Current when bias=0.0: ', min_current)
    #sys.exit(0)

    print(default_vols['c1'])
    pg.setval('c1', default_vols['c1'])
    time.sleep(1)
    max_current = pg.do0d()[0][0] # [0]: first channel
    print(pygor.get_params()) # check the full-length voltages
    print('Current: ', max_current)

    if settletime_big is not None:
        #set_big_jump = lambda: jump_mode(pygor, settletime_big, True)
        #set_small_jump = lambda: jump_mode(pygor, settletime, False)
        set_big_jump = lambda: jump_mode(pygor, settletime_big, True) # shuttling for large jumps
        set_small_jump = lambda: jump_mode(pygor, settletime, True) # shuttling for small jumps
    else:
        set_big_jump = lambda: 0 # do nothing
        set_small_jump = lambda: 0 # do nothing

    return pg, active_gate_idxs, lb_short, ub_short, max_current, min_current, set_big_jump, set_small_jump

def setup_pygor_dummy(shape):
    import pygor_dummy
    ndim = shape.ndim
    lb = -2000. * np.ones(ndim)
    ub = np.zeros(ndim)
    pg = pygor_dummy.PygorDummyBinary(lb, ub, shape)

    # set active voltages to 0.0
    num_active_gates = ndim
    active_gate_idxs = list(range(ndim))
    print('the number of active gates: ', num_active_gates)

    set_big_jump = lambda: 0 # do nothing
    set_small_jump = lambda: 0 # do nothing

    return pg, active_gate_idxs, lb, ub, 1.0, 0.0, set_big_jump, set_small_jump
