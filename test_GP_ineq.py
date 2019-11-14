import numpy as np
import GPy
from scipy.stats import norm
from scipy import optimize
import BO_common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def true_func(x):
    return np.maximum(-x, 0)

def obs_func(x):
    cut = 0.2
    true_val = true_func(x)
    exceed = true_val > cut
    obs_val = true_val.copy()
    obs_val[exceed] = cut
    return obs_val, exceed

def calc_g_Hinv(K, f, mu, y_b, f_b_idxs, E, std_noise):
    b, C = calc_b_C(y_b, f[f_b_idxs], E, std_noise)
    g = -np.linalg.solve(K, f-mu) + b
    print('f-mu: ', f-mu)
    print(-np.linalg.solve(K, f-mu),b)
    #H = inv(K) + C
    #Hinv = inv(H), updated covariance from H
    # assume that C is diagonal matrix
    diag_C = np.diag(C)
    Hinv = K.copy()
    for i in range(diag_C.size):
        c = diag_C[i]
        if c == 0.0:
            continue
        Hinv -= (c*Hinv[:,i:i+1]*Hinv[i:i+1,:]) / (1. + c*Hinv[i,i])
    return g, Hinv

def calc_b_C(y_b, f_b, E, std_noise, calc_b=True, calc_C=True):
    z = (f_b - y_b) / std_noise

    '''
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ratio = phi_z/Phi_z
    '''
    ratio = np.exp(norm.logpdf(z) - norm.logcdf(z))

    print('z: ', z)
    print('ratio: ', ratio)
    print('ratio: ', norm.pdf(z)/norm.cdf(z))

    b = C = None
    if calc_b:
        b = np.dot(E, ratio) / std_noise
    if calc_C:
        terms = (np.square(ratio) + ratio*z) / np.square(std_noise)
        EE = E[np.newaxis,...] * E[:,np.newaxis,...] # (i,j,k) component is 1 when E[i,k] == 1 and E[j,k] == 1
        C = np.dot(EE,terms)
    return b, C

def calc_E(x_a, x_b):
    '''
    Returns
        comp: 2D array(len(x_a), len(x_b)), whose (i,j) component is 1.0 when x_a[i] == x_b[j]
    '''
    x_a = np.atleast_2d(x_a)
    x_b = np.atleast_2d(x_b)

    comp = np.prod(x_a[:,np.newaxis,...] == x_b[np.newaxis,...], axis=-1)
    return comp

def main():
    x = np.linspace(-3,3,10)
    y, ineq = obs_func(x)

    # (x1, y1) : exact observation
    x1 = x[np.logical_not(ineq)]
    y1 = y[np.logical_not(ineq)]

    # (x2, y2): inequality observation
    x2 = x[ineq]
    y2 = y[ineq]

    plt.figure()
    plt.scatter(x1, y1, color='b')
    plt.scatter(x2, y2, color='r')
    plt.savefig('observation')

    # GP
    kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=2.0)
    model = GPy.models.GPRegression(x1[:,np.newaxis], y1[:,np.newaxis], kernel, noise_var=0.01**2)
    post_mean, post_cov = model.predict_noiseless(x[:,np.newaxis], full_cov=True)
    post_var = np.diag(post_cov)

    plt.figure()
    plt.scatter(x1, y1, color='b')
    plt.scatter(x2, y2, color='r')

    plt.plot(x, post_mean[:,0], 'C0')
    plt.fill_between(x, post_mean[:,0] - 2*np.sqrt(post_var),
                                  post_mean[:,0] + 2*np.sqrt(post_var),
                                  color='C0', alpha=0.2)
    plt.savefig('GP_simple')

    # Apply the inequality information
    f = post_mean[:,0].copy() # took a day to fix the bug (.copy())
    K = post_cov
    f_b_idxs = np.where(ineq)
    E = calc_E(x[:,np.newaxis], x2[:,np.newaxis])

    for i in range(1000):
        g, Hinv = calc_g_Hinv(K, f, post_mean[:,0], y2, f_b_idxs, E, 0.1)
        f += np.dot(Hinv,g)
    g, Hinv = calc_g_Hinv(K, f, post_mean[:,0], y2, f_b_idxs, E, 0.1)
    '''
    obj = lambda x : np.sqrt(np.sum(np.square(calc_g_Hinv(K, x, post_mean[:,0], y2, f_b_idxs, E, 0.1)[0])))
    lb = -1. * np.ones(len(f))
    ub = 1. * np.ones(len(f))
    result_x, result_fx = BO_common.optimize_lbfgs(f, obj, lb, ub, maxiter=10000)

    f = result_x
    g, Hinv = calc_g_Hinv(K, f, post_mean[:,0], y2, f_b_idxs, E, 0.1)
    #print(f, result_fx)
    #print(g)
    #print(Hinv)
    '''
    print(g)
    print(np.diag(Hinv))

    plt.figure()
    plt.scatter(x1, y1, color='b')
    plt.scatter(x2, y2, color='r')

    plt.plot(x, f, 'C0')
    plt.fill_between(x, f - 2*np.sqrt(np.diag(Hinv)),
                                  f + 2*np.sqrt(np.diag(Hinv)),
                                  color='C0', alpha=0.2)
    plt.savefig('GP_adjusted')

if __name__ == '__main__':
    main()
