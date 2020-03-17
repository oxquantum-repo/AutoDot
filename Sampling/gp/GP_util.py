import numpy as np
from .GPy_wrapper import GPyWrapper
from .util import L2_norm


def create_GP(num_active_gates, k_name='Matern52', var_f=1.0, lengthscale=1.0, center=0.0, const_kernel=False, GP = GPyWrapper):
    if np.isscalar(lengthscale):
        lengthscale = lengthscale * np.ones(num_active_gates)
    gp = GP() # initialize GP environment
    gp.center = center
    # GP kernels
    gp.create_kernel(num_active_gates, k_name, var_f, lengthscale, const_kernel=const_kernel)
    return gp

def create_GP_multiout(indim, outdim, k_name='Matern52', var_f=1.0, lengthscale=1.0, center=0.0, GP = GPyWrapper):
    if np.isscalar(lengthscale):
        lengthscale = np.ones(indim)
    gp = GP() # initialize GP environment
    gp.center = center
    # GP kernels
    gp.create_kernel(indim, outdim, k_name, var_f, lengthscale)
    return gp

def set_GP_prior(gp, l_prior_mean=None, l_prior_var=None, v_prior_mean=None, v_prior_var=None):
    if l_prior_mean is not None and l_prior_var is not None:
        gp.set_kernel_length_prior(l_prior_mean, l_prior_var)
    if v_prior_mean is not None and v_prior_var is not None:
        gp.set_kernel_var_prior(v_prior_mean, v_prior_var)

def fix_hyperparams(gp, fix_lengthscale=True, fix_variance=True):
    if fix_lengthscale:
        gp.fix_kernel_lengthscale()
    if fix_variance:
        gp.fix_kernel_var()

def gradient_surface(gp_r, v, origin, projection=True):
    '''
    compute the gradient of F(v) = ||v|| - ||r(u)||
    where u = v / ||v||
    F(v) = 0 is the implicit surface equation

    Args:
        v: 2D array (num_points, ndim)
        origin: scalar or 1D array (ndim)
        projection: project 'v' to the mean surface if True
    '''
    if v.ndim != 2: raise ValueError('v should be 2-dim.')

    v_origin = v - origin
    r = L2_norm(v_origin, axis=1, keepdims=False)
    u = v_origin / r[:, np.newaxis]
    ru, _, drdu, _ = gp_r.predict_withGradients(u)

    if projection:
        v_origin = ru * u
        r = ru[:,0]

    n, ndim = v.shape
    '''
    dfdv = np.zeros(v.shape)
    for i in range(n): 
        uut = np.outer(u[i], u[i])
        rdudv = np.eye(ndim) - uut
        dfdv[i] = u[i]-np.dot(rdudv,drdu[i])/r[i]
    '''
    uut = np.matmul(u[:,:,np.newaxis], u[:,np.newaxis,:])
    rdudv = np.eye(ndim)[np.newaxis,...] - uut
    temp = np.matmul(rdudv,drdu[...,np.newaxis])[...,0]
    dfdv = u - temp/r[:,np.newaxis]

    return dfdv

def hessian_surface(gp_r, v, origin, projection=True, delta=1.):
    '''
    compute the hessian of F(v) = ||v|| - ||r(u)||
    where u = v / ||v||
    F(v) = 0 is the implicit surface equation

    Args:
        v: 2D array (num_points, ndim)
        origin: scalar or 1D array (ndim)
        projection: project 'v' to the mean surface if True
    '''
    if v.ndim != 2: raise ValueError('v should be 2-dim.')

    v_origin = v - origin
    r = L2_norm(v_origin, axis=1, keepdims=False)
    u = v_origin / r[:, np.newaxis]
    ru, _, drdu, _ = gp_r.predict_withGradients(u)

    if projection:
        v_origin = ru * u
        r = ru[:,0]

    n, ndim = v.shape
    result = np.zeros((n, ndim, ndim))
    for i in range(ndim):
        #central difference
        d = np.zeros(ndim)
        d[i] = delta/2.
        grad_1 = gradient_surface(gp_r, v+d, origin, projection=False)
        grad_2 = gradient_surface(gp_r, v-d, origin, projection=False)
        result[:,i,:] = (grad_1 - grad_2) / delta
    result = 0.5 * (result + np.swapaxes(result, 1,2))
    return result, gradient_surface(gp_r, v, origin, projection=False)

#TODO: Incorrect implementation
def curvature_surface(gp_r, v, origin, axes = None, projection=True, delta=1.):
    '''
    Args:
        axes: None or list of ints
    '''
    if v.ndim != 2: raise ValueError('v should be 2-dim.')

    v_origin = v - origin
    r = L2_norm(v_origin, axis=1, keepdims=False)
    u = v_origin / r[:, np.newaxis]
    ru, _, drdu, _ = gp_r.predict_withGradients(u)

    if projection:
        v_origin = ru * u
        r = ru[:,0]

    n, ndim = v.shape
    if axes == None:
        axes = np.arange(ndim)
    elif len(axes == 1):
        raise ValueError('More than 2 axes should be given to calculate the curvature.')

    result = np.zeros(n)
    H, grad = hessian_surface(gp_r, v, origin, projection=False, delta=1.)
    #H_ = H[:,np.ix_(axes, axes)]
    axes = np.array(axes)
    H_ = H[:, axes.reshape((-1,1)), axes.reshape((1,-1))]
    grad_ = grad[:, axes]
    norm_grad = L2_norm(grad_, axis=1, keepdims=True)
    grad_unit = grad_ / norm_grad
    print(grad_unit,H)

    temp = np.matmul(H_, grad_unit[...,np.newaxis])[...,0]

    k = np.sum(temp*grad_unit,axis=1)/norm_grad[:,0]

    return k, H, grad
