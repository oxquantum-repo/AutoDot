import time
import numpy as np
from .gp import util

def L2_norm(vector):
    return np.sqrt(np.sum(np.square(vector)))

def check_inside_boundary(voltages, lb, ub):
    if np.any(voltages<lb) or np.any(voltages>ub):
        return False
    return True

def project_inside_boundary(voltages, lb, ub):
    vol = voltages.copy()
    vol[vol<lb] = lb[vol<lb]
    vol[vol>ub] = ub[vol>ub]
    return vol

def truncate_distance(v_in, unit_vector, d, lb, ub):
    v = v_in + d * unit_vector
    if check_inside_boundary(v, lb, ub):
        return d
    else:
        d_candidates = np.linspace(0.,d,100)
        inside = np.array([check_inside_boundary(v_in + d_*unit_vector, lb, ub) for d_ in d_candidates])
        last_idx = np.where(inside)[0][-1] # [0]
        return d_candidates[last_idx]

def translate_window_inside_boundary(w_from, w_to, lb, ub):
    w_size = w_to - w_from
    
    exceed_lb = w_from<lb
    w_from[exceed_lb] = lb[exceed_lb]
    w_to[exceed_lb] = w_from[exceed_lb] + w_size[exceed_lb]

    exceed_ub = w_to>ub
    w_to[exceed_ub] = ub[exceed_ub]
    w_from[exceed_ub] = w_to[exceed_ub] - w_size[exceed_ub]

    return w_from, w_to

def adjust_window(a, b, lb, ub):
    if a >= b :
        raise ValueError('Incorrect voltage window')
    if a < lb:
        diff = lb - a
        a =lb
        b += diff
    if b > ub:
        diff = b - ub
        a = a - diff
        b = ub
    return a, b

class Tester(object):
    def __init__(self, jump, measure, real_lb, real_ub, detector_pinchoff, d_r=5, len_after_pinchoff=200, channel=0, logging=False, detector_conducting=None, set_big_jump=None, set_small_jump=None):
        self.jump = jump # connection
        self.measure = measure
        self.lb = np.array(real_lb) # lower bound
        self.ub = np.array(real_ub) # upper bound
        self.d_r = d_r # step size (default)
        self.detector_pinchoff = detector_pinchoff
        self.len_after_pinchoff = len_after_pinchoff
        self.channel = channel
        self.logging = logging
        if self.logging:
            self.logger = list()
        self.detector_conducting = detector_conducting

        self.set_big_jump = set_big_jump
        self.set_small_jump = set_small_jump

    def get_r(self, unit_vector, r_est=None, d_r=None, origin=0.0):
        if np.sum(unit_vector) == 0.0:
            raise ValueError('Unit vector cannot be all zero.')
        # ensure the length of unit_vector is 1.0
        unit_vector = unit_vector / np.sqrt(np.sum(np.square(unit_vector)))
        # set the origin
        if np.isscalar(origin):
            origin = origin*np.ones_like(unit_vector)
        else:
            if len(origin) != len(unit_vector):
                raise ValueError('Wrong shape of origin.')

        voltages_all, measurement_all, pinchoff_idx, t_firstjump = self.measure_until_poff(np.array(origin), unit_vector, r_est, d_r=d_r)
        if self.logging:
            self.logger.append({'vols':voltages_all, 'val':measurement_all, 'pinchoff_idx': pinchoff_idx})
            
            
        trace_data = zip(voltages_all,measurement_all)

        # note: pinch-off = -1 when pinch-off is not detected
        found = pinchoff_idx != -1
        if len(voltages_all) == 0:
            raise RuntimeError('Voltages out of bound')
        voltages_pinchoff = voltages_all[pinchoff_idx]
        r = L2_norm(voltages_all[pinchoff_idx]-origin)
        return r, voltages_pinchoff, found, t_firstjump, trace_data

    def measure_until_poff(self, voltages, unit_vector, dist_est=None, d_r=None, max_r = np.infty):
        d_r = d_r or self.d_r # d_r = self.d_r if d_r is None

        t_firstjump = 0.
        do_est = False
        # if an estimated distance is given, go there and trace back until reach high current
        if dist_est is not None and self.detector_conducting is not None:
            if dist_est > 0.0:
                do_est = True
                dist_est = truncate_distance(voltages, unit_vector, dist_est, self.lb, self.ub)
                voltages_ = voltages + unit_vector * dist_est
                voltages_all, measurement_all, detected_idx, t_firstjump =  \
                    self.search_line(voltages_, self.detector_conducting, -unit_vector, d_r, [-1], 0.0)
                if detected_idx is not -1: # jump only conducting point is detected
                    voltages = voltages_all[detected_idx]

        voltages_all, measurement_all, pinchoff_idx, t_firstjump_ =  \
            self.search_line(voltages, self.detector_pinchoff, unit_vector, d_r, [-1, 0], self.len_after_pinchoff, max_r)
        if do_est is False: t_firstjump = t_firstjump_

        #if pinchoff_idx == 0:
        #    print('WARNING: pinch off from index 0')

        return voltages_all, measurement_all, pinchoff_idx, t_firstjump

    def search_line(self, voltages, detector, unit_vector, step_size, ignore_idxs, len_after=0.0, max_dist = np.infty):
        '''
        Args:
            detector: function that returns a scalar integer of a detected point
            ignore_idxs: list of integer indice, search won't stop if pinchoff idx is in the list
        Returns:
            voltages_all: list of 1D array (num of points for a line trace, num of gates)
            measurement_all: list of scalar vals (num of points for a line trace)
            found_idx: scalar, -1 indicates nothing detected across the whole line, or the voltages are out of bound
                                0 indicates detected at the starting point
            
        '''
        voltages_from = voltages.copy()
        voltages_all = list()
        measurement_all = list()
        found_idx = -1

        # There are two lines that terminate the while loop
        # 1. checking the voltages is still in the boundary, and distance < max_dist, break if not
        # 2. break if poff detected, but check some signal afterwards within 'len_after'
        first_iter = True
        while check_inside_boundary(voltages, self.lb, self.ub) and L2_norm(voltages-voltages_from) < max_dist:
            
            if first_iter:
                t = time.time()
                # big jump expected, swith on the big jump mode
                if self.set_big_jump is not None:
                    self.set_big_jump()
                self.jump(voltages.tolist()) # set voltages for measurement
                # switch off the big jump mode
                if self.set_small_jump is not None:
                    self.set_small_jump()
                first_iter = False
                t_firstjump = time.time() - t
            else:
                self.jump(voltages.tolist()) # set voltages for measurement

            current = self.measure() # current measurement
            voltages_all.append(voltages.copy()) # location of the measurement
            measurement_all.append(current)
            found_idx = detector(np.array(measurement_all))
            if found_idx not in ignore_idxs and L2_norm(voltages_all[-1]-voltages_all[found_idx]) >= len_after:
                # pinchoff found, not 0, and measured enough length after the found pinchoff
                break
            voltages = voltages + step_size*unit_vector # go futher to theta direction
        return voltages_all, measurement_all, found_idx, t_firstjump

    # OLD, use 'measure_dvec'
    def measure_dist_all_axis(self, voltages, d_r = None):
        return self.measure_dvec(voltages, d_r)

    # NEW
    def measure_dvec(self, voltages, d_r = None, axes= None):
        assert voltages.ndim == 1
        if axes == None:
            axes = list(range(len(voltages)))
        ndim = len(voltages)

        d_all = np.zeros(ndim)
        poff_all = np.full(ndim, True)

        voltages = project_inside_boundary(voltages, self.lb, self.ub)
        meas_each_axis = list()
        vols_each_axis = list()

        for i in range(len(voltages)):
            if i not in axes:
                d_all[i] = np.nan
                poff_all[i] = False
                meas_each_axis.append(None)
                vols_each_axis.append(None)
                continue
            unit_vector = np.zeros_like(voltages)
            unit_vector[i] = -1.0
            voltages_all, measurement_all, pinchoff_idx, _ = self.measure_until_poff(voltages, unit_vector, d_r=d_r)
            meas_each_axis.append(measurement_all)
            if len(voltages_all) == 0: # Out of Bound
                d_all[i] = np.nan
                vols_each_axis.append([])
            else:
                d_all[i] = np.max(np.fabs(voltages_all[pinchoff_idx] - voltages)) # all vols are same except i th dim.
                if pinchoff_idx == -1: # penalty for non-pinchoff boundary
                    poff_all[i] = False
                vols_each_axis.append(voltages_all[pinchoff_idx])

        return d_all, poff_all, meas_each_axis, vols_each_axis
