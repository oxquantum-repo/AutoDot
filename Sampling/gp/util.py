import numpy as np

class PinchoffDetectorThreshold(object):
    def __init__(self, th_low):
        self.th_low = th_low
    def __call__(self, trace, reverse_direction=False):
        if reverse_direction is True: #reverse search
            trace = trace[::-1]

        low = trace < self.th_low
        # change_points is 1 when ~low -> low, or low -> ~low (~: not)
        #change_points = np.logical_xor(low[:-1], np.logical_not(low[1:]))
        change_points = np.logical_xor(low[:-1], low[1:])
        change_points = np.concatenate( ([True], change_points), axis=0 ) # change point is true for the first point

        possible_points = np.logical_and(low,change_points)

        # high_prev indicates whether it was above the thresholdshold at least once
        # (not high enough -> low is not pinchoff)
        #high_prev = np.concatenate( ([False], (trace>self.th_high)[:-1]), axis=0 ) 
        #for i in range(1,trace.size):
        #    high_prev[i] = np.logical_or(high_prev[i-1], high_prev[i])
        #possible_points = np.logical_and(possible_points,high_prev)

        # keep_low indicates whether it keeps low afterwards
        keep_low = np.concatenate( (low[1:], [True]), axis=0 ) 
        for i in range(trace.size-2,-1,-1):
            keep_low[i] = np.logical_and(keep_low[i], keep_low[i+1])

        idxs = np.where(np.logical_and(possible_points,keep_low))[0]
        if np.size(idxs) == 0:
            return -1

        idx = idxs[0]
        if reverse_direction is True: #reverse search
            idx = np.size(trace)-idx-1
        return idx

class ConductingDetectorThreshold(object):
    def __init__(self, th_high):
        self.th_high = th_high
    def __call__(self, trace):
        high = trace > self.th_high
        idxs = np.where(high)[0]
        if np.size(idxs) == 0:
            return -1
        return idxs[0] # return the first index satistying the condition

def merge_data(vols_poff_all, detected_all, vols_poff_axes_all, poff_all):
    #print(detected_all,poff_all)
    vols_poff_all = np.array(vols_poff_all)
    detected_all = np.array(detected_all)
    vols_poff_axes_all = np.array(vols_poff_axes_all)
    poff_all = np.array(poff_all,dtype=np.float)

    if vols_poff_axes_all.size == 0 and poff_all.size == 0:
        return vols_poff_all, vols_poff_all[detected_all] , detected_all
    else:
        num_gates = vols_poff_all.shape[-1]
        
        vols_poff_axes_all = vols_poff_axes_all.reshape((-1, num_gates))
        poff_all = poff_all.ravel()

        vols_all = np.concatenate([vols_poff_all, vols_poff_axes_all], axis=0)
        
        #print(detected_all,poff_all)
        found_all = np.concatenate([detected_all, poff_all], axis=0)
        
        return vols_all, vols_all[found_all], found_all

def L1_norm(arr, axis=None, keepdims=False):
    return np.sum(np.fabs(arr), axis=axis, keepdims=keepdims)

def L2_norm(arr, axis=None, keepdims=False):
    return np.sqrt(np.sum(np.square(arr), axis=axis, keepdims=keepdims))

def ur_from_vols_origin(vols, origin, returntype='list'):
    vols = np.array(vols)
    num_data = vols.shape[0]
    if num_data == 0:
        return [], []
    ndim = vols.shape[1]

    if np.isscalar(origin):
        origin = origin * np.ones(ndim)

    diff = vols - origin[np.newaxis, :]
    r_all = L2_norm(diff, axis=1)
    u_all = diff / r_all[:,np.newaxis]

    if returntype == 'list':
        return [u for u in u_all], r_all.tolist()
    else:
        return u_all, r_all

def compute_hardbound(poff_vec, found, vols_pinchoff, step_back, origin, bound):
    do_change = False
    
    print("dvec pinches: ",poff_vec)

    # Only one gate can pinchoff
    if np.sum(poff_vec) == 1:
        do_change = True
        new_origin = np.array(origin).copy()
        # if only one gate can pinchoff the current, move the origin
        idx = np.nonzero(poff_vec)
        if found:
            new_origin[idx] = np.array(vols_pinchoff)[idx] + np.array(step_back)
        if not found:
            new_origin[idx] = np.array(vols_pinchoff)[idx]
        origin = new_origin
        print("New origin: ",origin)
    
    real_ub = np.maximum(origin,bound)
    real_lb = np.minimum(origin,bound)
    
    return {'changed_origin':do_change, 'origin':origin, 'real_ub':real_ub, 'real_lb':real_lb}
