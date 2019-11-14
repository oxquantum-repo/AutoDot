import numpy as np
from gym import spaces
import config
from test_common import Tester
import pygor_dummy
import util

class Env(object):
    def __init__(self):
        self.counter = 0
        self.done = False # boolean
        self.info = dict()

    def step(self, action):
        raise NotImplementedError()
        #return [observation, reward, done, info]

    def reset(self):
        raise NotImplementedError()
        #return self.state

    def render(self, mode='human', close=False):
        raise NotImplementedError()

def to_vector(val, ndim):
    if np.isscalar(val):
        val = np.ones(ndim)*val
    else:
        if len(val) != ndim: raise ValueError('Dimension mismatch.')
    return val

class Action_grid(object):
    def __init__(self, ndim, unit_move):
        self.action_list = list()
        for i in range(self.ndim):
            change_vol = np.zeros(ndim)
            change_vol[i] = unit_move
            self.action_list.append(change_vols[i])
            change_vol[i] = -unit_move
            self.action_list.append(change_vols[i])
        self.type = 'Discrete'
        self.num_actions = len(self.action_list)
        self.unit_move = unit_move

    def __call__(self, state, action):
        return state + self.action_list[action]

    def __str__(self):
        return 'step{}'.format(int(self.unit_move))

class Reward_singlecorner(object):
    def __init__(self, th_corner):
        self.th_corner = 500.
    def __call__(self, vols, d_vec):
        return None
    def __str(self):
        return 'singlecorner'

class Env_simple(object):
    '''
    state: relative voltages + d vector
    '''
    def __init__(self, shape, len_d = None, params_poff=None, action_def=None, start=-100, params_reward=None):
        self.shape = shape
        self.ndim = shape.ndim
        self.unit_move = unit_move
        self.counter = 0
        self.done = 0
        self.start = to_vector(start, self.ndim)

        # Define actions
        if action_def is None:
            self.action_def = Action_grid(self.ndim, 10.)
        self.action_space = spaces.Discrete(self.action_def.num_actions)
        self.observation_space = None

        # Create the tester
        pg, active_gate_idxs, lb_short, ub_short, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor_dummy(shape)
        if params_poff is None:
            params_poff = {'step_back':100, 'len_after_pinchoff':100, 'th_high':0.8, 'th_low':0.2, 'd_r':10}
        if params_reward is None:
            self.params_reward = {'OoB':-1., 'th_d'=500.}
        self.step_back = params_poff['step_back']
        detector_pinchoff = util.PinchoffDetectorThreshold(params_poff['th_low']) # pichoff detector
        detector_conducting = util.ConductingDetectorThreshold(params_poff['th_high']) # reverse direction
        tester = Tester(pg, lb_short, ub_short, detector_pinchoff, d_r=params_poff['d_r'], len_after_pinchoff=params_poff['len_after_pinchoff'], logging=False, detector_conducting=detector_conducting, set_big_jump = set_big_jump, set_small_jump = set_small_jump)

        self.pg = pg
        self.tester = tester

        self.reset()

    def reset(self, start=None):
        if start is None:
            self.vols = self.start
        else:
            self.vols = to_vector(start, self.ndim)

        self.vols = np.zeros(self.ndim) # state
        d_vec, poff_vec, meas_each_axis, vols_each_axis = self.tester.measure_dist_all_axis(self.vols+self.step_back)
        self.d_vec = d_vec
        print(self.vols, self.d_vec)

    def step(self, action):
        vols_next = self.action_def(self.vols, action)
        if np.any(self.shape.lb > vols_next) or np.nay(self.shape.ub < vols_next):
            # out of bound
            reward1 = self.params_reward['OoB'] # punish OoB, state remains
            measure_dvec = False
        else:
            # valid action
            self.vols = vols_next
            reward1 = 0.0
            measure_dvec = True

        if measure_dvec:
            d_vec, poff_vec, meas_each_axis, vols_each_axis = self.tester.measure_dist_all_axis(self.vols+self.step_back)
            d_vec = 


        observation = np.append(self.vols, self.d_vec)
        #return [observation, reward, done, info]

def test():
    box_dim = 2
    box_a = -1000. * np.ones(box_dim)
    box_b = 500. * np.ones(box_dim)
    shape = pygor_dummy.Box(ndim=box_dim, a=box_a, b=box_b)
    th_leak = -500.*np.ones(2)
    shape = pygor_dummy.Leakage(shape, th_leak)

    params_poff = {'step_back':100, 'len_after_pinchoff':100, 'th_high':0.8, 'th_low':0.2, 'd_r':10}
    env = Env_simple(shape, box_dim, params_poff)

if __name__ == '__main__':
    test()
    
    
