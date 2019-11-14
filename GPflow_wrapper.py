import numpy as np
import tensorflow as tf

import gpflow

from GP_interface import GPInterface, convert_lengthscale, convert_2D_format

class GPflowWrapper(GPInterface):
    def __init__(self):
        super().__init__()
        self.opt = gpflow.train.AdamOptimizer()
        self.optimizer_tensor = None
        self.center = 0.0

    @gpflow.defer_build()
    def create_kernel(self, ndim, kernel_name, var_f=1.0, lengthscale=1.0, const_kernel=False):
        if kernel_name == 'Matern52':
            l = convert_lengthscale(ndim, lengthscale)
            kernel = gpflow.kernels.Matern52(input_dim=ndim, ARD=True, variance=var_f, lengthscales=l, name='basic')
        else:
            raise ValueError('Unsupported kernel.')

        self.ndim = ndim
        self.kernel = kernel

        if const_kernel:
            self.kernel += gpflow.kernels.Constant(1.0)
            self.stat_kernel = self.kernel.kernels[0]
        else:
            self.stat_kernel = self.kernel

    @gpflow.defer_build()
    def set_kernel_length_prior(self, prior_mean, prior_var):
        if self.ndim != len(prior_mean) or self.ndim != len(prior_var):
            raise ValueError('Incorrect kernel prior parameters.')

        shape_all, scale_all = mv_to_shape_scale(prior_mean, prior_var)
        self.stat_kernel.lengthscales.prior = gpflow.priors.Gamma(shape_all, scale_all)

    @gpflow.defer_build()
    def set_kernel_var_prior(self, prior_mean, prior_var):
        shape, scale = mv_to_shape_scale(prior_mean, prior_var)
        self.stat_kernel.variance.prior = gpflow.priors.Gamma(prior_mean, prior_var)

    def fix_kernel_lengthscale(self):
        self.stat_kernel.lengthscales.set_trainable(False)
    def fix_kernel_var(self):
        self.stat_kernel.variance.set_trainable(False)

    def create_model(self, x, y, noise_var, noise_prior='fixed'):
        x = convert_2D_format(x)
        y = convert_2D_format(y) - self.center
        if self.model is None:
            self.model = gpflow.models.GPR(x, y, self.kernel)
            if noise_prior == 'fixed':
                self.model.likelihood.variance = noise_var
                self.model.likelihood.variance.set_trainable(False)
            else:
                raise ValueError('Not Implemented yet.')
            self.model.compile()

            #shape = [None, x.shape[1]]
            #self.x_pred = tf.placeholder(tf.float32, shape=shape, name='x_pred')
            #self.mean_full, self.cov_full = self.model.predict_f_full_cov(self.x_pred)
            #self.mean, self.cov = self.model.predict_f(self.x_pred)
            #print(self.mean)
            #print(self.cov)
        else:
            self.model.X = x
            self.model.Y = y

    def predict_f(self, x, full_cov=False):
        x = convert_2D_format(x)
        if full_cov:
            mean, cov = self.model.predict_f_full_cov(x)
        else:
            mean, cov = self.model.predict_f(x)
        return mean + self.center, cov

    def posterior_sample_f(self, x, size = 10):
        '''
        Parameters
        x: (Nnew x input_dim)
        Returns
        (Nnew x output_dim x samples)

        '''
        samples = self.model.predict_f_samples(x, size) + self.center
        return np.swapaxes(samples, 0, 2) # follow GPy format
    
    def optimize(self, num_iter=2000, num_restarts=30, opt_messages=False, print_result=False):
        # TODO multiple restarts from the samples of the prior distribution
        # https://github.com/GPflow/GPflow/issues/797
        if self.optimizer_tensor is None:
            self.optimizer_tensor = self.opt.make_optimize_tensor(self.model)
        session = gpflow.get_default_session()
        for i in range(num_iter):
            session.run(self.optimizer_tensor)
        self.model.anchor(session)
        print(self.model)

def mv_to_shape_scale(mu, var):
    theta = var / mu
    k = np.square(mu)/var
    return k, theta

def main():
    from GPy_wrapper import GPyWrapper
    N = 12
    X = np.random.rand(N,1)
    Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.1 + 3

    print(X)

    ndim = 1
    l_prior_mean = 0.2 * np.ones(ndim)
    l_prior_var = 0.1*0.1 * np.ones(ndim)
    v_prior_mean = 3.0**2
    v_prior_var = 3.0**4
    noise_var = 0.1

    #with gpflow.defer_build():
    gp = GPflowWrapper()
    #gp = GPyWrapper()
    gp.create_kernel(ndim, 'Matern52', var_f=1.0, lengthscale=np.ones(ndim))
    print(gp.kernel)
    gp.set_kernel_length_prior(l_prior_mean, l_prior_var)
    gp.set_kernel_var_prior(v_prior_mean, v_prior_var)
    gp.create_model(X , Y, noise_var, noise_prior='fixed')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    XX = np.linspace(0,1,100).reshape(100,1)
    mean_post, var_post = gp.predict_f(XX, full_cov=False)
    samples =  gp.posterior_sample_f(XX, size = 10)
    print(samples.shape)

    plt.figure()
    plt.scatter(X[:,0], Y[:,0], marker='x')
    plt.plot(XX[:,0], mean_post[:,0], 'C0')
    print(XX.shape, mean_post.shape, var_post.shape)
    plt.fill_between(XX[:,0], mean_post[:,0] - 2*np.sqrt(var_post[:,0]),
                                  mean_post[:,0] + 2*np.sqrt(var_post[:,0]),
                                  color='C0', alpha=0.2)
    plt.savefig('GPflow_test')
    plt.close()

    print(gp.model)
    gp.optimize()
    mean_post, var_post = gp.predict_f(XX, full_cov=False)

    plt.figure()
    plt.scatter(X[:,0], Y[:,0], marker='x')
    plt.plot(XX[:,0], mean_post[:,0], 'C0')
    print(XX.shape, mean_post.shape, var_post.shape)
    plt.fill_between(XX[:,0], mean_post[:,0] - 2*np.sqrt(var_post[:,0]),
                                  mean_post[:,0] + 2*np.sqrt(var_post[:,0]),
                                  color='C0', alpha=0.2)
    plt.savefig('GPflow_test_opt')
    plt.close()

if __name__ == '__main__':
    main()
