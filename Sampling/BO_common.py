from functools import partial
from multiprocessing import Pool
import numpy as np 
from scipy.stats import norm # for calculating normpdf, normcdf
from scipy import optimize # for optimisation
from pyDOE import lhs # Latin hypercube sampling
#import scipydirect

# for minimisation
def EI( best_prev, mean_x, std_x, min_obj=True , dmdx=None, dsdx=None ):
    #diff = best_prev - mean_x
    #return diff*norm.cdf(diff/std_x) + std_x*norm.pdf(diff/std_x)
    diff = best_prev - mean_x
    if min_obj is False:
        diff = -diff # max problem
    z = (best_prev - mean_x)/std_x
    phi, Phi = norm.pdf(z), norm.cdf(z)

    if dmdx is not None and dsdx is not None:
        #https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/acquisitions/EI.py
        if min_obj:
            dEIdx = dsdx * phi - Phi * dmdx
        else:
            dEIdx = dsdx * phi + Phi * dmdx
    else:
        dEIdx = None
    return std_x*(z*Phi + phi), dEIdx

def augmented_EI( best_prev, mean_x, std_x, std_noise, min_obj=True, dmdx=None, dsdx=None ):
    var_sum = np.square(std_noise) + np.square(std_x)
    EI, _ = EI( best_prev, mean_x, std_x, min_obj, dmdx, dsdx) 
    aug_EI = EI* (1.0 - std_noise/np.sqrt(var_sum))
    # WARNING: gradient is not impledmented yet
    return aug_EI, None
    
def sample_lhs_basic(ndim, num_samples):
    # returns a num_samples x ndim array
    lhd = lhs(ndim, samples=num_samples) 
    return lhd

def sample_lhs_bounds(lb, ub, num_samples):
    # returns a num_samples x ndim array
    if lb.ndim != 1 or ub.ndim != 1:
        raise ValueError('Bounds should be 1-dim. vectors.')
    if lb.size != ub.size:
        raise ValueError('Length of lb should be same with ub.')
    if np.any(lb > ub):
        raise ValueError('lb cannot be larger than ub.')

    ndim = ub.size
    diff = ub - lb
    lhd = sample_lhs_basic(ndim, num_samples)
    lhd = lhd * diff + lb

    return lhd

# TODO: check trust-constr params
def optimize_trust_constr(x0, f, lb, ub, const_func=None, maxiter=200):
    dim = lb.size
    bounds = [(lb[i],ub[i]) for i in range(dim)]
    # constraint: const_func(x) == 0
    const = optimize.NonlinearConstraint(const_func, 0.0, 0.0)
    res = optimize.minimize(f, x0=x0, method='trust-constr', jac='3-point', hess='3-point', bounds=bounds, constraints=const)
    result_x = np.atleast_1d(res.x)
    result_fx = np.atleast_1d(res.fun)
    return result_x, result_fx

def optimize_lbfgs(x0, f, lb, ub, const_func=None, maxiter=200):
    if const_func is not None:
        f_augmented = lambda x : f(x) + 10000.*const_func(x)
    else:
        f_augmented = f
    dim = lb.size
    bounds = [(lb[i],ub[i]) for i in range(dim)]
    res = optimize.fmin_l_bfgs_b(f,x0=x0,bounds=bounds,approx_grad=True, maxiter=maxiter)

    d = res[2]
    if d['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
        result_x  = np.atleast_1d(x0)
    else:
        result_x = np.atleast_1d(res[0])
    result_fx = f(result_x)
    const_val = const_func(result_x)

    converged = True
    if d['warnflag'] != 0:
        converged = False
    disp = True
    if converged is False and disp is True:
        if d['warnflag'] == 1:
            print('Too many function evaluations or iterations')
        elif d['warnflag'] == 2:
            print('Stopped for another reason')
        print('x: ', result_x, ', fx: ', result_fx)
        print('gradient: ', d['grad'], ', constraint: ', const_val )

    return result_x, result_fx, {'converged':converged, 'const_val':const_val}

def optimize_Largrange(x0, f, lb, ub, const_func, maxiter=200):
    dim = lb.size
    bounds = [(lb[i],ub[i]) for i in range(dim)]  + [(0.0, np.inf)]
    f_augmented = lambda x : f(x[:-1]) + x[-1]*const_func(x[:-1])

    x0 = np.append(x0, 1.0) # initial lambda
    res = optimize.fmin_l_bfgs_b(f_augmented,x0=x0,bounds=bounds,approx_grad=True, maxiter=maxiter)

    d = res[2]
    if d['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
        result_x  = np.atleast_1d(x0)[:-1]
    else:
        result_x = np.atleast_1d(res[0])[:-1]
    result_fx = f(result_x)
    const_val = const_func(result_x)

    converged = True
    if d['warnflag'] != 0:
        converged = False
    disp = True
    if converged is False and disp is True:
        if d['warnflag'] == 1:
            print('Too many function evaluations or iterations')
        elif d['warnflag'] == 2:
            print('Stopped for another reason')
        print('x: ', result_x, ', lambda: ', res[0][-1], ', fx: ', result_fx)
        print('gradient: ', d['grad'], ', constraint: ', const_val )

    return result_x, result_fx, {'converged':converged, 'const_val':const_val}

def optimize_trust_constr(x0, f, lb, ub, const_func, maxiter=200):
    bounds = optimize.Bounds(lb, ub)
    nonlin_const = optimize.NonlinearConstraint(const_func, 0.0, 0.0, jac=const_func.J, hess=const_func.H)
    res = optimize.minimize(f, x0, method='trust-constr', constraints=[nonlin_const], bounds=bounds)
    converged = res.status is 1 or res.status is 2
    return res.x, res.fun, {'converged':converged, 'const_val':res.constr[0]}

def optimize_SLSQP(x0, f, lb, ub, const_func, maxiter=200):
    bounds = optimize.Bounds(lb, ub)
    eq_cons = {'type': 'eq',
            'fun': const_func,
            'jac': const_func.J}
    res = optimize.minimize(f, x0, method='SLSQP', constraints=[eq_cons], bounds=bounds, options={'ftol': 1e-9})
    converged = res.status is 0
    const_val = const_func(res.x)
    return res.x, res.fun, {'converged':converged, 'const_val':const_val}

def filter_results(result_filter, x_all, fx_all, stat_all):
    if callable(result_filter):
        filtered_all = [result_filter(stat) for stat in stat_all]
        if any(filtered_all):
            x_all = [x for (x,filtered) in zip(x_all, filtered_all) if filtered]
            fx_all = [fx for (fx,filtered) in zip(fx_all, filtered_all) if filtered]
            stat_all = [stat for (stat,filtered) in zip(stat_all, filtered_all) if filtered]
        else:
            print('WARNING: No result can satisfy the result filter')

    return x_all, fx_all, stat_all


def optimize_multi_x0(opt_func, x0_all, f, lb, ub, const_func, maxiter=200, result_filter=None):
    num_x0 = len(x0_all)

    # run the optimizer with multiple restarts
    x_found_all = list()
    fx_found_all = list()
    stat_all = list()
    for idx_x0 in range(len(x0_all)):
        x0 = x0_all[idx_x0]
        result_x, result_fx, stat = opt_func(x0, f, lb, ub, const_func=const_func, maxiter=maxiter)
        x_found_all.append(result_x)
        fx_found_all.append(result_fx) 
        stat_all.append(stat)

    x_found_all, fx_found_all, stat_all = filter_results(result_filter, x_found_all, fx_found_all, stat_all)
    idx_min = np.argmin(fx_found_all) # index of max EI
    x_min = x_found_all[idx_min]
    fx_min = fx_found_all[idx_min]
    return x_min, fx_min

def optimize_multi_x0_parallel(opt_func, x0_all, f, lb, ub, const_func, maxiter=200, result_filter=None, num_proc=4):
    #f_x0 = lambda x0 : opt_func(x0, f, lb, ub, const_func=const_func, maxiter=maxiter)
    f_x0 = partial(opt_func, f=f, lb=lb, ub=ub, const_func=const_func, maxiter=maxiter)
    pool = Pool(processes=num_proc)
    x0_all = [x0 for x0 in x0_all]
    list_tuples = pool.map(f_x0, x0_all)
    x_found_all, fx_found_all, stat_all = zip(*list_tuples) # list of tuples to multiple lists

    x_found_all, fx_found_all, stat_all = filter_results(result_filter, x_found_all, fx_found_all, stat_all)
    idx_min = np.argmin(fx_found_all) # index of max EI
    x_min = x_found_all[idx_min]
    fx_min = fx_found_all[idx_min]
    return x_min, fx_min

def optimize_DIRECT(f, lb, ub, const_func, maxiter=200):
    dim = lb.size
    bounds = [(lb[i],ub[i]) for i in range(dim)]
    f_augmented = lambda x : f(x) + 10.*const_func(x)

    res = scipydirect.minimize(f_augmented, bounds=bounds)
    print(res)
    x = res['x']
    print(const_func(x))
    print(f(x))
    return res['x'], res['fun']

class Constraint_SS(object):
    '''
    Sum of squared values
    '''
    def __call__(self, x):
        return np.sum(np.square(x)) -1.0  # constraint to make length 1.0
    def J(self, x):
        #print(x)
        return [2.0 * x]
    def H(self, x, v):
        #print(x,v)
        return v * 2.0 * np.eye(x.size)

class Constraint_Sum(object):
    '''
    Sum of values
    '''
    def __call__(self, x):
        return np.sum(x) - 1.0 # constraint to make length 1.0
    def J(self, x):
        return [x]
    def H(self, x, v):
        return np.zeros((x.size,x.size))

def uniform_to_hypersphere(samples):
    samples = norm.ppf(samples) # change to normally distributed samples
    samples = np.fabs(samples) # make to one-sided samples
    samples = -samples/np.sqrt(np.sum(np.square(samples),axis=1,keepdims=True)) # length to 1, direction to negative side
    return samples

def random_hypersphere(dim, num_samples):
    samples = np.random.uniform(size=(num_samples,dim))
    #return (-directions[np.newaxis,:])*uniform_to_hypersphere(samples)
    return uniform_to_hypersphere(samples)
def lhs_hypersphere(dim, num_samples):
    samples = sample_lhs_basic(dim, num_samples) # could be confusing with param. of np.random.uniform
    return uniform_to_hypersphere(samples)

def random_hypercube(point_lb, point_ub, num_samples):
    assert len(point_lb) == len(point_ub)
    ndim = len(point_lb)

    interval = point_ub - point_lb
    offset = point_lb
    samples = np.random.uniform(size=(num_samples,ndim))*interval[np.newaxis,:] + offset
    #print(samples)
    return samples

'''
def random_hypercube(point_lb, point_ub, num_samples):
    assert len(point_lb) == len(point_ub)
    ndim = len(point_lb)
    # choose a face
    face_idx = np.random.randint(ndim) # coordinate of this index is point_lb[face_idx]
    face_lb = np.append(point_lb[:face_idx], point_lb[face_idx+1:])
    face_ub = np.append(point_ub[:face_idx], point_ub[face_idx+1:])

    interval = point_ub - point_lb
    interval[face_idx] = 0.
    offset = point_lb
    samples = np.random.uniform(size=(num_samples,ndim))*interval[np.newaxis,:] + offset
    #print(samples)
    return samples

    # convert samples to unit vectors
    #u_samples = (samples-point_ub[np.newaxis,:])/np.sqrt(np.sum(np.square(samples),axis=1,keepdims=True)) # length to 1, direction to negative side
    #return u_samples
'''

# for test lhd
def main():
    lb = np.array([1.0, 2.0, 3.0])
    ub = np.array([2.0, 3.0, 4.0])
    lhd = sample_lhs_bounds(lb,ub,10)
    print(lhd)

if __name__ == "__main__":
    main()
