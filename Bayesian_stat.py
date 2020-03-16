import numpy as np
from scipy.stats import invgamma, beta, gamma
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar

eps = 1.E-13

def calc_estimates(n, t):
    # posterior rate (lambda) is a Gamma distributiona (with Jeffreys prior of Poisson observations)
    a = n + 0.5 # 
    b = t

    p10 = invgamma.ppf(0.1, a=a, scale=b)
    p90 = invgamma.ppf(0.9, a=a, scale=b)
    mean = invgamma.mean(a=a, scale=b)
    return mean, p10, p90

def ci_mu_multi(n_all, t_all, lb=0.1, ub=0.9, labeller_idxs=None):
    n_all = np.array(n_all)

    if labeller_idxs is not None:
        n_all = n_all[:,labeller_idxs]
    t_sum = np.sum(t_all)
    n_sum = np.sum(n_all, axis=0)
    gamma_all = [gamma(a=0.5+n, scale=1./t_sum) for n in n_sum]
    P = len(gamma_all)
    cdf_lambda_mix = lambda x: np.sum([g.cdf(x) for g in gamma_all]) / P
    sf_lambda_mix = lambda x: np.sum([g.sf(x) for g in gamma_all]) / P

    cdf_mu = lambda x: sf_lambda_mix(1./x)
    pdf_mu = lambda x: (cdf_mu(x+0.5*eps) - cdf_mu(x-0.5*eps))/eps

    ci_lower = root_scalar(lambda x: cdf_mu(x) - lb, bracket=(1.E-10, 1.E3)).root
    ci_upper = root_scalar(lambda x: cdf_mu(x) - ub, bracket=(1.E-10, 1.E3)).root
    MAP = minimize_scalar(lambda x: -pdf_mu(x), bracket=(1.E-10, 1.E2), method='Golden').x
    MED = root_scalar(lambda x: cdf_mu(x) - 0.5, bracket=(1.E-10, 1.E3)).root
    
    # MC estimate of mean_mu
    # It needs random samples of lambda
    samples_lambda = [g.rvs(size=1000000) for g in gamma_all]
    samples_mu = 1/np.array(samples_lambda)
    mean = np.mean(samples_mu)

    #return mean, ci_lower, ci_upper
    return MED, ci_lower, ci_upper

def ci_p_multi(n_all, N_all, lb=0.1, ub=0.9, labeller_idxs=None):
    n_all = np.array(n_all)

    if labeller_idxs is not None:
        n_all = n_all[:,labeller_idxs]
    N_sum = np.sum(N_all)
    n_sum = np.sum(n_all, axis=0)

    beta_all = [beta(0.5+n, 0.5+(N_sum-n)) for n in n_sum]
    P = len(beta_all)
    cdf_mix = lambda x : np.sum([b.cdf(x) for b in beta_all]) / P
    #pdf_mix = lambda x: (cdf_mix(x+0.5*eps) - cdf_mix(x-0.5*eps))/eps
    pdf_mix = lambda x: np.sum([b.pdf(x) for b in beta_all]) / P

    f_lb = lambda x: cdf_mix(x) - lb
    f_ub = lambda x: cdf_mix(x) - ub

    ci_lower = root_scalar(lambda x: cdf_mix(x) - lb, bracket=(0., 1.)).root
    ci_upper = root_scalar(lambda x: cdf_mix(x) - ub, bracket=(0., 1.)).root
    MAP = minimize_scalar(lambda x: -pdf_mix(x), bracket=(eps, 1.-eps), method='Golden').x
    MED = root_scalar(lambda x: cdf_mix(x) - 0.5, bracket=(0., 1.)).root

    mean = np.sum([b.mean() for b in beta_all]) / P

    #return mean, ci_lower, ci_upper
    return MED, ci_lower, ci_upper

labeller_idxs = [0, 2, 3]

print('--------------------------------')
print('Ablation------------------------')
print('--------------------------------')
print('Pure random------------------------')
time_pr = [68.89280066, 6.889280066]
N_pr = [10000, 1000]
peaks_pr = [49, 6]
success_pr = [[1,2,3,1],
              [0,0,0,0]]
print(ci_p_multi(np.array(peaks_pr)[:,np.newaxis], N_pr))
print(ci_p_multi(success_pr, peaks_pr, labeller_idxs=labeller_idxs))
print(ci_mu_multi(success_pr, time_pr, labeller_idxs=labeller_idxs))


print('Uniform surface------------------------')
time_us = [11.4628116184472] 
N_us = [500]
peaks_us = [78]
success_us = [[0, 1, 1, 1]]
print(ci_p_multi(np.array(peaks_us)[:,np.newaxis], N_us))
print(ci_p_multi(success_us, peaks_us, labeller_idxs=labeller_idxs))
print(ci_mu_multi(success_us, time_us, labeller_idxs=labeller_idxs))


print('High-res always------------------------')
time_highres = [35.2738895172556]
N_highres = [500]
peaks_highres = [367]
success_highres = [[4,4,8,2]]
print(ci_p_multi(np.array(peaks_highres)[:,np.newaxis], N_highres))
print(ci_p_multi(success_highres, peaks_highres, labeller_idxs=labeller_idxs))
print(ci_mu_multi(success_highres, time_highres, labeller_idxs=labeller_idxs))


print('Full method------------------------')
time_gpc = [15.7, 16.2] 
N_gpc = [500, 459]
peaks_gpc = [363, 340]
success_gpc = [[2,6,9,5],
               [5,3,7,4]]
print(ci_p_multi(np.array(peaks_gpc)[:, np.newaxis], N_gpc))
print(ci_p_multi(success_gpc, peaks_gpc, labeller_idxs=labeller_idxs))
print(ci_mu_multi(success_gpc, time_gpc, labeller_idxs=labeller_idxs))


print('Gropued gates------------------------')
time_group = [14.8139497190056]
N_group = [500]
peaks_group = [389]
success_group = [[7,15,13,11]]
print(ci_p_multi(np.array(peaks_group)[:, np.newaxis], N_group))
print(ci_p_multi(success_group, peaks_group, labeller_idxs=labeller_idxs))
print(ci_mu_multi(success_group, time_group, labeller_idxs=labeller_idxs))


print('--------------------------------')
print('Tuning------------------------')
print('--------------------------------')

print('Basel1------------------------')
time_B1 = [11.4302536, 11.2369793884621, 10.5331366785367, 11.3588183663288, 11.2068004349205]
N_B1 = [500, 500, 500, 500, 500]
success_B1 = [[3,9,5,7],
              [2,3,3,1],
              [3,4,8,5],
              [1,6,9,6],
              [0,4,4,2]]
print(ci_mu_multi(success_B1, time_B1, labeller_idxs=labeller_idxs))

print('Basel2------------------------')
time_B2 = [9.087031979, 8.208526366, 8.23819523, 7.686700076, 8.380222799]
N_B2 = [500, 500, 500, 500, 500]
success_B2 = [[7,8,15,6],
              [5,4,5,6],
              [7,8,14,7],
              [4,8,8,4],
              [5,11,12,19]]
print(ci_mu_multi(success_B2, time_B2, labeller_idxs=labeller_idxs))

print('Grouped gates------------------------')
time_GG = [11.09331678 ]
N_GG = [500]
success_GG = [[12,19,25,20]]
print(ci_mu_multi(success_GG, time_GG, labeller_idxs=labeller_idxs))
