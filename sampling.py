# https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
# https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
import numpy as np
#import tensorflow as tf

# for testing
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


# TODO: resampling (effective sample size), MCMC move (proposal distribution, acceptance rate), keeping 'z'

# z: num_z X dim_z
# weights: a vector with the size num_z
# weighted samples -> equally weighted samples
def resample( z, weights, num_samples=None):
    assert weights.size == z.size

    num_z = z.shape[0]
    if num_samples is None:
        num_samples = num_z
    acc_weight = np.cumsum(weights) / np.sum(weights)

    # samples from 0 to 1
    U = np.ones(num_samples) / np.float(num_samples)
    U[0] = np.random.uniform(0.0,1.0/np.float(num_samples)) #first sample
    U = np.cumsum(U)

    # calculate the number of samples per each interval
    num_per_interval = np.zeros(acc_weight.size, dtype=np.int32)
    for i in range(acc_weight.size):
        lb = acc_weight[i-1] if i is not 0 else 0.0
        ub = acc_weight[i]
        num_per_interval[i] = np.sum(np.logical_and(U >= lb, U< ub).astype(np.int32))
    new_z = []
    for copied in [[z[i]]*num_per_interval[i] for i in range(num_z) ]:
        new_z += copied
    new_z = np.stack(new_z, axis=0)

    new_weights = np.ones(num_samples) / np.float(num_samples)

    return new_z, new_weights

# Effective sample size
def calculate_ESS(weights):
    weights_normalized = weights / np.sum(weights)
    ess = 1.0/np.sum(np.square(weights_normalized))
    return ess

# PDF of normal distributions
def pdf_standard(z_std):
    return np.exp(-np.square(z_std)/2.0)/(np.sqrt(2)*np.pi)
def norm_pdf(z, mu, sigma):
    return pdf_standard((z-mu)/sigma) / sigma

# 
class Proposal_move(object):
    pass
class Gaussian_proposal_move(Proposal_move):
    def __init__(self, cov=1.0E-2):
        if np.isscalar(cov):
            self.cov = cov
        elif cov.ndim is 1:
            self.cov = np.diag(cov)
        elif cov.ndim is 2:
            self.cov = cov
        else:
            raise ValueError
    # z: num_samples 
    def __call__(self, z):
        num_z, dim_z = z.shape
        if np.isscalar(self.cov):
            new_z = z + np.random.normal(scale=np.sqrt(self.cov), size=z.shape)
        else:
            new_z = z + np.random.multivariate_normal(np.zeros(dim_z), self.cov, size=num_z)
        return new_z

class Likelihood(object):
    pass
# for testing, mixture of two Gaussian distributions
class Likelihood_test(Likelihood):
    def __call__(self, z):
        assert z.shape[1] == 1

        mean1 = 0.5
        std1 = np.square(0.5)
        mean2 = 1.5
        std2 = np.square(0.5)

        L = norm_pdf(z,mean1,std1) + norm_pdf(z,mean2,std2)
        return L.ravel() # returns 1D vector

class MH_MCMC(object):
    def __init__(self, move, L_func, log_mode=False):
        assert isinstance(move, Proposal_move)
        #assert isinstance(L_func, Likelihood)
        self.move = move
        self.L_func = L_func
        self.log_mode = log_mode
    def __call__(self, z, L_current = None, num_step = 10):
        z_old = z
        if L_current is None:
            L_current = self.L_func(z, normalize=False, log_mode=self.log_mode)
        time_list = list()
        for i in range(num_step):
            time_start = time.time()
            z_proposal = self.move(z_old) # random move
            L_new = self.L_func(z_proposal, normalize=False, log_mode=self.log_mode) # evaluate the likelihood at the proposed positions
            if self.log_mode is True:
                acceptance_prob = np.clip(np.exp(L_new-L_current),0.0,1.0)
            else:
                acceptance_prob = np.clip(L_new/(L_current+1.0E-10),0.0,1.0)
            accept = np.random.uniform(size=acceptance_prob.shape) < acceptance_prob
            z_new = np.copy(z_old)
            #import ipdb; ipdb.set_trace() # Start debugger
            z_new[accept] = z_proposal[accept]
            z_old = z_new
            L_current[accept] = L_new[accept] #update L_current
            time_list.append(time.time()-time_start)
        #print('Average time for one iteration of sampling:{}'.format(np.mean(time_list)))
        return z_new

def test():
    ## test data
    #z_test = np.array([[1.0], [2.0], [3.0]])
    #likelihood_test = np.array([1.0, 3.0, 2.0])

    ## test resampling
    #new_z = resample(z_test, likelihood_test, num_samples=10)
    #print(new_z)

    ## test ESS
    #print(calculate_ESS(likelihood_test))

    # test the likelihood function
    z = np.linspace(-1.0, 3.0, 100).reshape(100,1)
    L_function = Likelihood_test()
    L = L_function(z)
    plt.subplot(4,1,1)
    plt.plot(z,L)
    plt.xlim(-1.0,3.0)

    # initial samples
    mean_init = 1.0
    std_init = 0.3
    num_samples = 300
    z_init =  np.random.normal(mean_init, std_init, size=num_samples)
    z_init = z_init.reshape(num_samples,1)

    ax1 = plt.subplot(4,1,2)
    L_init = norm_pdf(z_init,mean_init,std_init).ravel()
    weights = L_function(z_init) / L_init 
    print(weights.shape)
    plt.plot(z_init,L_init, 'bo')
    ax1.set_ylabel('sampled distribution')
    ax2 = ax1.twinx()
    plt.plot(z_init,weights, 'ro')
    ax2.set_ylabel('weights')
    plt.xlim(-1.0,3.0)

    # test ESS
    print(calculate_ESS(weights))

    # resample
    z_resampled,_ = resample(z_init, weights, num_samples)
    plt.subplot(4,1,3)
    plt.hist(z_resampled.ravel(), bins=100, range=(-1.0,3.0))
    plt.ylabel('after resampling')

    # MCMC move
    move = Gaussian_proposal_move(np.square(0.2))
    mcmc = MH_MCMC(move, L_function)
    z_after_mcmc = mcmc(z_resampled, num_step=20 )
    plt.subplot(4,1,4)
    plt.hist(z_after_mcmc.ravel(), bins=100, range=(-1.0,3.0))
    plt.ylabel('after mcmc')
    plt.show()

if __name__ == "__main__":
    test()
