import numpy as np
from scipy.optimize import brentq

# Compute distances from a function u -> r given v_0
def compute_di(f, v0, origin, i, lb, ub):
    '''
    Args:
        f: callable, input is a unit vector, output is a distance from the origin
        v0: the point where the distances to be computed
        origin: 1D vector, origin
        i: axis number
        lb: lower bound
        ub: upper bound
    '''
    # TODO: Check whether v0 is inside of the volume

    v_end = v0.copy()
    v_end[i] = lb[i]

    v = lambda t : (1.-t) * v0 + t*v_end
    r = lambda t : np.sqrt(np.sum(np.square(v(t)-origin)))
    u = lambda t : (v(t)-origin)[np.newaxis,:] / r(t)

    result = np.nan
    if r(0.) > f(u(0.)):
        # v_0 is outside of the volume
        result = 0.
    if r(1.) < f(u(1.)):
        # v_1 is inside of the volume
        result = v0[i] - lb[i]

    if np.isnan(result):
        f_root = lambda t : r(t) - f(u(t))
        t_root = brentq(f_root, 0., 1., xtol=0.01)
        result = v(t_root)
    return result

def compute_d(f, v0, origin, lb, ub, axes=None):
    if len(v0) != len(origin):
        raise ValueError('Length of v0 is different from the origin.')
    if axes is None:
        axes = np.arange(len(origin))
    d = list()
    for ax in axes:
        d.append(compute_di(f, v0, origin, ax, lb, ub))
    return np.array(d)







    
    
