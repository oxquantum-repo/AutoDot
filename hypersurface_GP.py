import numpy as np
from scipy.stats import multivariate_normal as mvnnorm
from scipy.stats import norm, truncnorm

from hypersurface import Hypersurface
from GPy_wrapper import GPyWrapper as GP
from BO_common import EI

def drop_onedim(arr, idx):
    return np.concatenate((arr[...,:idx], arr[...,(idx+1):]), axis=-1)

class Hypersurface_IndependentGP(Hypersurface):
    def __init__(self, ndim, l_prior_mean=400., l_prior_var=200.**2,
                                        v_prior_mean=200.**2, v_prior_var=100.**(4)):
        '''
        Args:
            ndim: integer, dimension of the voltage space
            l_prior_mean: scalar or 1D array, mean of lengthscale parameters
            l_prior_var: scalar or 1D array, var. of lenthscale parameters
            v_prior_mean: scalar or 1D array, mean of variance parameters
            v_prior_var: scalar or 1D array, var. of variance parameters
        Notes
            Lengthscale determines how quickly other voltages changes w.r.t. a given voltage 
            Variance determines how big voltages vary
        '''

        self.ndim = ndim
        if np.isscalar(l_prior_mean):
            l_prior_mean = l_prior_mean * np.ones(ndim)
        if np.isscalar(l_prior_var):
            l_prior_var = l_prior_var * np.ones(ndim)
        if np.isscalar(v_prior_mean):
            v_prior_mean = v_prior_mean * np.ones(ndim)
        if np.isscalar(v_prior_var):
            v_prior_var = v_prior_var * np.ones(ndim)

        # create a GP for each dimension
        gp_list = list()
        for i in range(ndim):
            gp = GP()
            # output is ith voltage and input is other voltages
            gp.create_kernel(ndim-1, 'Matern52', 
                    var_f=v_prior_mean[i], lengthscale=drop_onedim(l_prior_mean,i)) 
            gp.set_kernel_length_prior(drop_onedim(l_prior_mean,i), drop_onedim(l_prior_var,i))
            gp.set_kernel_var_prior(v_prior_mean[i], v_prior_var[i])
            gp_list.append(gp)
        self.gp_list = gp_list

        self.mean_vec = -500 * np.ones(ndim) # dummy mean vector for testing

    def set_data(self, vs, **kargs):
        '''
        Args:
            vs: 3D array (ndim, num_data, ndim)
            kargs:
                noise_var: scalar, variance of noise
        '''
        self.data = vs
        noise_var = kargs['noise_var']

        self.mean_vec = np.zeros(self.ndim)
        for i in range(self.ndim):
            self.mean_vec[i] = np.mean(vs[i,:,i])
        for i in range(self.ndim):
            gp = self.gp_list[i]
            gp.create_model(drop_onedim(vs[i],i), vs[i,:,i:i+1]-self.mean_vec[i], noise_var, noise_prior='fixed')
            #gp.optimize_restarts(num_restarts=10)

    def is_inside(self, v, **kargs):
        '''
        Args:
            v: 1D or 2D array, (num_points, ndim)
        Returns: 
            prob: scalar, probability that all of 'v' are inside of the hypersurface
        TODO: not tested
        '''
        # prob_inside =  product of P(d_i > 0.0) for all i
        _, cdf_vec = self.prob_d(v, np.zeros(self.ndim), cdf_on=True)
        return np.prod(1.-cdf_vec)

    def posterior_d(self, vs, full_cov=True):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
        Returns: 
            d_mean_all: 2D array (ndim, num_vs), mean of 'd_g' at 'vs' per each 'g'
            d_cov_all: 2D array (ndim, num_vs) if full_cov=False, or 3D array (ndim, num_vs, num_vs), cov of 'd_g' at 'vs' per each 'g'
        '''
        vs = np.atleast_2d(vs)
        mean_cov_all = [self.posterior_dg(vs,i,full_cov) for i in range(self.ndim)] # list of tuple (mean, cov)
        d_mean_all, d_cov_all = zip(*mean_cov_all) # two lists of mean and cov
        return np.array(d_mean_all), np.array(d_cov_all)

    def posterior_dg(self, vs, g_idx, full_cov=True):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            g_idx: integer index of an axis
        Returns: 
            dg_mean: 1D array (num_vs), mean of 'd_g' for all vs
            cov: 1D array (num_vs) if full_cov=False, or 2D array (num_vs, num_vs), covariance of 'd_g' for all vs
        TODO: not tested
        '''
        vs = np.atleast_2d(vs)
        if vs.shape[-1] != self.ndim:
            raise ValueError('vs has a wrong shape. ')

        gp = self.gp_list[g_idx]
        v_mean, cov = gp.predict_f(drop_onedim(vs,g_idx), full_cov=full_cov)
        # v_mean: 2D array (num_vs,1)
        dg_mean = -(v_mean[:,0] + self.mean_vec[g_idx] - vs[:,g_idx])
        #print(v_mean[:,0])
        #print(self.mean_vec[g_idx])
        #print(vs[:,g_idx])
        if full_cov is False:
            cov = cov[:,0] #[:,0] for removing the singleton dim.
        return dg_mean, cov

    def trunc_moment_dg(self, vs, g_idx, lb=-np.infty, ub=np.infty, order=1):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            g_idx: integer index of an axis
            lb: scalar, for truncate normal distributions
            ub: sacalr, for truncte normal distributions
            order: integer, order of the moment
        Returns:
            vector, moment for each 'v' in 'vs'
        '''
        dg_mean, dg_var = self.posterior_dg(vs, g_idx, full_cov=False)
        dg_std = np.sqrt(dg_var)
        lb_std, ub_std = (lb - dg_mean)/dg_std, (ub - dg_mean)/dg_std

        num_vs = vs.shape[0]
        result = [truncnorm.moment(order, lb_std[i], ub_std[i], dg_mean[i], dg_std[i]) for i in range(num_vs)]

        return np.array(result)
        #return truncnorm.moment(order, lb_std, ub_std, dg_mean, dg_std)

    def prob_d(self, vs, xs, pdf_on=False, cdf_on=False, **kargs):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            xs: scalar or 1D array (ndim or num_vs) or 2D array (ndim, num_vs), location to evaluate pdf or cdf
        Returns:
            pdf: None or 1D, p(dg = xg) for each g
            cdf: None or 1D, P(dg < xg) for each g
        TODO: not tested
        '''
        vs = np.atleast_2d(vs)
        num_vs = vs.shape[0]
        xs = np.atleast_2d(xs)
        if xs.size == 1:
            pass # xs.shape[0] == 1, broadcast expected in prob_dg
        elif xs.size == num_vs:
            pass # xs.shape[0] == 1
        elif xs.size == self.ndim:
            xs = xs.szie # shape to (self.ndim, 1), broadcast expected in prob_dg
        else:
            raise ValueError('Unsupproted shape of xs.')

        if xs.shape[0] == 1:
            pdf_cdf = [self.prob_dg(vs,i,xs[0], pdf_on, cdf_on) for i in range(self.ndim)] 
        elif xs.shape[0] == self.ndim:
            pdf_cdf = [self.prob_dg(vs,i,xs[i], pdf_on, cdf_on) for i in range(self.ndim)] 
        else:
            raise ValueError('Wrong shape of xs.')
        pdf_vec, cdf_vec = zip(*pdf_cdf) # two lists of pdf and cdf
        return np.array(pdf_vec), np.array(cdf_vec)


    def prob_dg(self, vs, g_idx, xs, pdf_on=False, cdf_on=False, **kargs):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            g_idx: integer index of an axis
            xs: scalar or 1D array(num_vs) or 2D array (num_xs, num_vs)
        Returns:
            pd: None or scalar or 1D array (num_xs), p(dg(v) = x(vi))
            prob: None or scalar or 1D array (num_xs), P(dg(v) < x(vi))
        '''
        vs = np.atleast_2d(vs)
        dg_mean, dg_cov =  self.posterior_dg(vs, g_idx)

        pd = mvnnorm.pdf(xs, dg_mean, dg_cov) if pdf_on else None
        prob = mvnnorm.cdf(xs, dg_mean, dg_cov) if pdf_on else None

        return pd, prob 

    def prob_inside_each(self, vs, logscale=False, offset=0.0):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            g_idx: integer index of an axis
            logscale: return logpdf if True
        Returns:
            pd: 1D array (num_vs), P('v' is inside) for each 'v' in 'vs'
        '''
        vs = np.atleast_2d(vs)

        logSF_list = [self.prob_inside_dg_each(vs, i, logscale=True, offset=offset) for i in range(self.ndim)]
        logSF = np.sum(logSF_list,axis=0)
        if logscale:
            return logSF
        else:
            return np.exp(logSF)

    def prob_inside_dg_each(self, vs, g_idx, logscale=False, dg_mean=None, dg_var=None, offset=0.0):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            g_idx: integer index of an axis
            logscale: return logpdf if True
        Returns:
            pd: 1D array (num_vs), P('v_g' is inside) for each 'v' in 'vs'
        '''
        vs = np.atleast_2d(vs)
        if dg_mean is None or dg_var is None:
            dg_mean, dg_var =  self.posterior_dg(vs, g_idx, full_cov=False)
        if logscale:
            return norm.logsf(offset, dg_mean, np.sqrt(dg_var))
        else:
            # sf: survival function, 1 - cdf, numerically more stable
            return norm.sf(offset, dg_mean, np.sqrt(dg_var))

    def boundary_likelihood(self, vs, logscale=False):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            logscale: return logpdf if True
        Returns:
            pd: 1D array (num_vs), probability density that 'v' is boundary
        '''
        vs = np.atleast_2d(vs)

        logL_list = [self.boundary_likelihood_dg(vs, i, logscale=True) for i in range(self.ndim)]
        logL = np.sum(logL_list,axis=0)
        if logscale:
            return logL
        else:
            return np.exp(logL)

    def boundary_likelihood_dg(self, vs, g_idx, logscale=False):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            g_idx: integer index of an axis
            logscale: return logpdf if True
        Returns:
            pd: 1D array (num_vs), probability density that 'v' is boundary along 'g'th axis
        '''
        vs = np.atleast_2d(vs)
        dg_mean, dg_var =  self.posterior_dg(vs, g_idx, full_cov=False)
        if logscale:
            return norm.logpdf(0.0, dg_mean, np.sqrt(dg_var))
        else:
            return norm.pdf(0.0, dg_mean, np.sqrt(dg_var))

    
    def generate_samples_d(self, v, num_samples=1, **kargs):
        '''
        Return: returns 2D array (num_samples x ndim) of vector 'd' given 'v'
        '''
        return None

    def generate_samples_dg(self, v, g_idx, num_samples=1, **kargs):
        '''
        Return: returns 2D array (num_samples x ndim) of vector 'd' given 'v'
        '''

        d_mean, cov = self.posterior_dg(v, g_idx, full_cov=True)
        samples = mvnnorm.rvs(mean=d_mean, cov=cov, size=num_samples)
        return samples
    
    def generate_samples_hypersurface(self, num_points, num_samples):
        '''
        Args:
            num_points: the number of points for a single hypersurface
            num_samples: the number of samples of hypersurface to draw
        Returns:
            samples: 3D array (num_samples x num_points x ndim)
        '''
        return None

    def EI2(self, vs, d_best, stepback=0.0):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            d_best: 1D array (ndim), 'd' at v_{best}
            stepback: scalar, step back length for each gate
        Returns:
            1D array (num_vs), sum of E[max(d_g(v)-d_g(v_{vest}),0)| v is inside] for all g
        '''
        vs = np.atleast_2d(vs)

        val_best = np.sum(d_best)
        #temp = self.trunc_moment_dg(vs, 0, lb=stepback)
        moment_alldim = [self.trunc_moment_dg(vs, i, lb=stepback) for i in range(self.ndim)]

        EI_sum = val_best - np.sum(moment_alldim, axis=0)
        prob_inside = self.prob_inside_each(vs, logscale=False)
        EI = np.maximum(EI_sum * prob_inside, 0.0)
        return EI

    def EI(self, vs, d_best, stepback=0.0):
        '''
        Args:
            vs: 2D array (num_vs, ndim)
            d_best: 1D array (ndim), 'd' at v_{best}
            stepback: scalar, step back length for each gate
        Returns:
            1D array (num_vs), sum of E[max(d_g(v)-d_g(v_{vest}),0)| v is inside] for all g
        '''
        vs = np.atleast_2d(vs)


        if stepback != 0.0:
            EI_alldim = [self.EI_dg(d_best[i], stepback, vs=vs, g_idx=i) for i in range(self.ndim)]
            logprob_inside = self.prob_inside_each(vs, logscale=True)

        else: # to reuse and save computations
            d_mean, d_var = self.posterior_d(vs, full_cov=False)
            logprob_inside_alldim = [self.prob_inside_dg_each(vs, i, logscale=True, dg_mean=d_mean[i], dg_var=d_var[i]) for i in range(self.ndim)]
            #prob_inside_alldim: (ndim, num_vs)
            EI_alldim = [self.EI_dg(d_best[i], stepback, logprob_inside_dg=logprob_inside_alldim[i], dg_mean=d_mean[i], dg_var=d_var[i]) for i in range(self.ndim)]
            logprob_inside = np.sum(logprob_inside_alldim, axis=0)

        EI_sum = np.sum(EI_alldim, axis=0)
        EI = EI_sum * np.exp(logprob_inside)
        return EI

    def EI_dg(self, dg_best, stepback=0.0, **kargs):
        '''
        Args:
            dg_best: scalar, d_g at v_{best}
            stepback: scalar, step back length for each gate
        Kargs (when step_back == 0.0, to reduce computations):
            logprob_inside_dg: 1D array
            dg_mean: 1D array
            dg_var: 1D array
        Kargs (when step_back != 0.0):
            vs: 2D array (num_vs, ndim)
            g_idx: gate index
        Returns:
            1D array (num_vs), E[max(d_g(v)-d_g(v_{vest}),0)| v is inside]
        '''
        if stepback == 0.0:
            logprob_inside_dg = kargs['logprob_inside_dg']
            dg_mean = kargs['dg_mean']
            dg_var = kargs['dg_var']
            num_vs = len(dg_mean)
        else:
            vs = kargs['vs'] + stepback
            g_idx = kargs['g_idx']
            dg_mean, dg_var = self.posterior_dg(vs, g_idx, full_cov=False)
            logprob_inside_dg = self.prob_inside_dg_each(vs, g_idx, logscale=True, dg_mean=dg_mean, dg_var=dg_var, offset=stepback)
            num_vs = vs.shape[0]

        if stepback > dg_best:
            return np.zeros(num_vs)

        term1 = EI(dg_best, dg_mean, np.sqrt(dg_var))
        term2 = EI(stepback, dg_mean, np.sqrt(dg_var))
        term3 = (dg_best-stepback) * (1. - np.exp(logprob_inside_dg))
        term_all = term1 - term2 - term3
        print(term_all)
        return (term1 - term2 - term3) / (np.exp(logprob_inside_dg) + 1.0e-12)

