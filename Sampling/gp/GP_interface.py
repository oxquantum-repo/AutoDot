import numpy as np

class GPInterface(object):
    def __init__(self):
        self.kernel = None
        self.ndim = None
        self.model = None
        self.outdim = 1

    def create_kernel(self, ndim, kernel_name, var_f=1.0, lengthscale=1.0):
        pass

    def create_model(self, x, y, noise_var, noise_prior):
        pass

    def predict_f(self, x, full_cov=False):
        pass

    def optimize(self, num_restarts=30, opt_messages=False, print_result=False):
        pass

def convert_lengthscale(ndim, lengthscale):
    if np.isscalar(lengthscale):
        l = lengthscale * np.ones(ndim)
    else:
        l = lengthscale
    return l

def convert_2D_format(arr):
    if not isinstance(arr, np.ndarray):
        raise ValueError('The array is not a numpy array.')
    if arr.ndim == 1:
        return arr[:, np.newaxis] # asuumes arr is single dimensional data
    if arr.ndim == 2:
        return arr
    else:
        raise ValueError('The array cannot be more than 2 dimensional')
    

