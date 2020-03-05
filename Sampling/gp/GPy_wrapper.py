import numpy as np

import GPy

from .GP_interface import GPInterface, convert_lengthscale, convert_2D_format

class GPyWrapper(GPInterface):
    def __init__(self):
        # GPy settings
        GPy.plotting.change_plotting_library("matplotlib") # use matpoltlib for drawing
        super().__init__()
        self.center = 0.0

    def create_kernel(self, ndim, kernel_name, var_f=1.0, lengthscale=1.0, const_kernel=False):
        if kernel_name == 'Matern52':
            l = convert_lengthscale(ndim, lengthscale)
            kernel = GPy.kern.Matern52(input_dim=ndim, ARD=True, variance=var_f, lengthscale=l, name='basic')
        elif kernel_name == 'RBF':
            l = convert_lengthscale(ndim, lengthscale)
            kernel = GPy.kern.RBF(input_dim=ndim, ARD=True, variance=var_f, lengthscale=l, name='basic')
        else:
            raise ValueError('Unsupported kernel: '+ kernel_name)

        self.ndim = ndim
        self.kernel = kernel

        if const_kernel:
            self.kernel += GPy.kern.Bias(1.0)
            self.stat_kernel = self.kernel.basic
        else:
            self.stat_kernel = self.kernel

    def set_kernel_length_prior(self, prior_mean, prior_var):
        if self.ndim != len(prior_mean) or self.ndim != len(prior_var):
            raise ValueError('Incorrect kernel prior parameters.')
        if self.kernel is None:
            raise ValueError('Kernel should be defined first.')

        for i in range(self.ndim):
            self.stat_kernel.lengthscale[[i]].set_prior(GPy.priors.Gamma.from_EV(prior_mean[i],prior_var[i])) # don't know why, but [i] does not work

    def set_kernel_var_prior(self, prior_mean, prior_var):
        self.stat_kernel.variance.set_prior(GPy.priors.Gamma.from_EV(prior_mean,prior_var))

    def fix_kernel_lengthscale(self):
        self.stat_kernel.lengthscale.fix()
    def fix_kernel_var(self):
        self.stat_kernel.variance.fix()

    def create_model(self, x, y, noise_var, noise_prior='fixed'):
        x = convert_2D_format(x)
        y = convert_2D_format(y) - self.center
        self.outdim = y.shape[1]
        noise_var = np.array(noise_var)
        if noise_var.ndim == 0:
            self.model = GPy.models.GPRegression(x, y, self.kernel, noise_var=noise_var)
            noise = self.model.Gaussian_noise
        else:
            assert noise_var.shape == y.shape
            self.model = GPy.models.GPHeteroscedasticRegression(x, y, self.kernel)
            self.model['.*het_Gauss.variance'] = noise_var
            noise = self.model.het_Gauss.variance

        if noise_prior == 'fixed':
            noise.fix()
        else:
            raise ValueError('Not Implemented yet.')

    def predict_f(self, x, full_cov=False):
        '''
        Returns:
            posterior mean, posterior variance
        '''
        x = convert_2D_format(x)
        post_mean, post_var = self.model.predict_noiseless(x, full_cov=full_cov)
        if self.outdim > 1:
            post_var = np.concatenate([post_var]*self.outdim, axis=-1)
        return post_mean + self.center, post_var

    def predict_withGradients(self, x):
        '''
        Borrowed from https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/models/gpmodel.py
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        '''
        x = convert_2D_format(x)
        m, v = self.model.predict(x)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(x)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))
        return m + self.center, np.sqrt(v), dmdx, dsdx

    def posterior_sample_f(self, x, size = 10):
        '''
        Parameters
        x: (Nnew x input_dim)
        Returns
        (Nnew x output_dim x samples)

        '''
        return self.model.posterior_samples_f(x, size) + self.center

    def optimize(self, num_restarts=30, opt_messages=False, print_result=True, parallel=False):
        self.model.optimize_restarts(num_restarts=num_restarts, robust=True, parallel=False, messages=opt_messages)
        if print_result:
            print(self.kernel)
            print(self.stat_kernel.lengthscale)
            print(self.stat_kernel.variance)

class GPyWrapper_Classifier(GPyWrapper):
    def create_model(self, x, y):
        assert self.center == 0.0
        x = convert_2D_format(x)
        y = convert_2D_format(y)
        self.outdim = y.shape[1]
        self.model = GPy.models.GPClassification(x, y, self.kernel)

    def predict_prob(self, x):
        x = convert_2D_format(x)
        prob = self.model.predict(x, full_cov=False)[0]
        return prob

    def optimize(self, maxiter=1000, opt_messages=False, print_result=True):
        for i in range(5):
            self.model.optimize(max_iters=int(maxiter/5), messages=opt_messages)
        if print_result:
            print(self.kernel)
            print(self.stat_kernel.lengthscale)

class GPyWrapper_MultiSeparate(object):
    def create_kernel(self, ndim, outdim, kernel_name, var_f=1.0, lengthscale=1.0, const_kernel=False):
        if isinstance(kernel_name, str):
            kernel_name = [kernel_name]*outdim
        if np.isscalar(var_f):
            var_f = np.ones(outdim) * var_f
        if np.isscalar(lengthscale):
            var_f = np.ones(outdim) * lengthscale
        if isinstance(const_kernel, bool):
            const_kernel = [const_kernel]*outdim

        self.gp_list = list()
        for i in range(outdim):
            gp = GPyWrapper()
            gp.create_kernel(ndim, kernel_name[i], var_f[i], lengthscale[i], const_kernel[i])
            self.gp_list.append(gp)
        self.outdim = outdim

    def set_kernel_length_prior(self, prior_mean, prior_var):
        # Apply same prior for all outputs
        for i in range(self.outdim):
            self.gp_list[i].set_kernel_length_prior(prior_mean, prior_var)

    def set_kernel_var_prior(self, prior_mean, prior_var):
        # Apply same prior for all outputs
        for i in range(self.outdim):
            self.gp_list[i].set_kernel_var_prior(prior_mean, prior_var)

    def fix_kernel_lengthscale(self):
        for i in range(self.outdim):
            self.gp_list[i].fix_kernel_lengthscale()

    def fix_kernel_var(self):
        for i in range(self.outdim):
            self.gp_list[i].fix_kernel_var()

    def create_model(self, x, y, noise_var, noise_prior='fixed'):
        if not (y.ndim == 2 and y.shape[1] == self.outdim):
            raise ValueError('Incorrect data shape.')

        noise_var = np.array(noise_var)
        for i in range(self.outdim):
            if noise_var.ndim == 2 and noise_var.shape[1] == self.outdim:
                noise_var_i = noise_var[:, i:i+1]
            else:
                noise_var_i = noise_var
            gp = self.gp_list[i]
            gp.create_model(x, y[:,i:i+1], noise_var_i, noise_prior)

    def predict_f(self, x, full_cov=False):
        post_mean_all = list()
        post_var_all = list()
        for i in range(self.outdim):
            post_mean, post_var = self.gp_list[i].predict_f(x, full_cov)
            post_mean_all.append(post_mean)
            post_var_all.append(post_var)

        return np.concatenate(post_mean_all,axis=-1), np.concatenate(post_var_all,axis=-1)

    def posterior_sample_f(self, x, size = 10):
        post_samples_all = list()
        for i in range(self.outdim):
            post_samples = self.gp_list[i].predict_f(x, full_cov)
            post_samples_all.append(post_samples)
        return np.concatenate(post_samples_all,axis=1)

    def optimize(self, num_restarts=30, opt_messages=False, print_result=False):
        for i in range(self.outdim):
            self.gp_list[i].optimize(num_restarts, opt_messages, print_result)

    def predict_withGradients(self, x):
        '''
        Borrowed from https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/models/gpmodel.py
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
            m_all: (num_x, outdim)  
            std_all: (num_x, outdim)  
            dmdx_all: (num_x, outdim, n_dim)
            dsdx_all: (num_x, outdim, n_dim)
        '''
        m_all, std_all, dmdx_all, dsdx_all = [], [], [], []
        for i in range(self.outdim):
            m, std, dmdx, dsdx = self.gp_list[i].predict_withGradients(x)
            m_all.append(m)
            std_all.append(std)
            dmdx_all.append(dmdx)
            dsdx_all.append(dsdx)
        return np.concatenate(m_all,axis=-1), np.concatenate(std_all,axis=-1), np.stack(dmdx_all,axis=1), np.stack(dsdx_all,axis=1)

class GPyWrapper_MultiIndep(GPyWrapper):
    def create_kernel(self, ndim, outdim, kernel_name, var_f=1.0, lengthscale=1.0, const_kernel=False):
        super().create_kernel(ndim, kernel_name, var_f, lengthscale, const_kernel)

        k_multi = GPy.kern.IndependentOutputs([self.kernel, self.kernel.copy()])
        #icm = GPy.util.multioutput.ICM(input_dim=ndim, num_outputs=outdim, kernel=self.kernel)
        #icm.B.W.constrain_fixed(0) # fix W matrix to 0

        if const_kernel:
            self.stat_kernel = k_multi.sum.basic
        else:
            self.stat_kernel = k_multi.basic
        self.kernel = k_multi
        print(self.kernel)

    def create_model(self, x, y, noise_var, noise_prior='fixed'):
        x = convert_2D_format(x)
        y = convert_2D_format(y) - self.center

        numdata = x.shape[0]
        outdim = y.shape[1]
        indim = x.shape[1]

        yy = y.transpose().ravel()
        ind = np.concatenate([ o*np.ones(numdata) for o in range(outdim)])
        xx = np.concatenate([x]*outdim)
        xx = np.concatenate((xx,ind[:,np.newaxis]), axis=1)

        print(xx.shape, yy.shape)

        self.model = GPy.models.GPRegression(x, y, self.kernel, noise_var=noise_var)
        if noise_prior == 'fixed':
            self.model.Gaussian_noise.fix()
        else:
            raise ValueError('Not Implemented yet.')

def create_GP(num_active_gates, outdim, k_name='Matern52', var_f=1.0, lengthscale=1.0, center=0.0):
    if np.isscalar(lengthscale):
        lengthscale = np.ones(num_active_gates)
    gp = GPyWrapper() # initialize GP environment
    #gp = GPyWrapper_MultiIndep() # initialize GP environment
    gp.center = center
    # GP kernels
    gp.create_kernel(num_active_gates, k_name, var_f, lengthscale)
    #gp.create_kernel(num_active_gates, outdim, k_name, var_f, lengthscale)
    return gp

def main():
    X = np.arange(1,6).reshape((5,1))
    f = lambda x : np.square(x-4.0)
    #Y = np.concatenate([f(X), -f(X)], axis=1)
    Y = np.concatenate([f(X)], axis=1)
    #noise_var = 0.01**2
    #noise_var = np.concatenate([np.square(X / 10.)]*2, axis=1)
    noise_var = np.square(X / 10.)
    print(X.shape, Y.shape)
    gp = create_GP(1, 2, 'Matern52', 2.0, 1.0, 0.0)
    gp.create_model(X, Y, noise_var, noise_prior='fixed')

    gp.optimize()

    X_pred = np.linspace(1.,5.,10).reshape((-1,1))
    mean, cov = gp.predict_f(X_pred)
    print(mean)
    #print(cov)

    '''
    ###
    # GP Classification test
    ###
    X = np.arange(1,6).reshape((5,1))
    Y = np.array([1.0, 1.0, 1.0, 0.0, 0.0]).reshape((5,1))

    gpc = GPyWrapper_Classifier()
    gpc.create_kernel(1, 'RBF', 1.0, 1.0)
    gpc.create_model(X, Y)

    X_pred = np.linspace(1.,5.,10).reshape((-1,1))
    print(gpc.predict_prob(X_pred))
    print(gpc.model)
    gpc.optimize()
    print(gpc.predict_prob(X_pred))
    print(gpc.model)
    '''


if __name__ == '__main__':
    main()
