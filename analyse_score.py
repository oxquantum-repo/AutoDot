import time
from multiprocessing import Pool
import numpy as np
from  scipy.stats import pareto, uniform, exponweib
from scipy.optimize import fmin

from pathlib import Path
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

class Weibull(object):
    def __init__(self, k, lam):
        self.k = k
        self.lam =lam
    def rvs(self, size=None):
        return np.random.weibull(self.k, size=size) * self.lam

class Empirical(object):
    def __init(self, samples):
        self.samples = samples
    def rvs(self, size=None):
        return np.random.choice(self.samples, size)

class Env_threshold(object):
    def __init__(self, params):
        self.params = params
        self.th = params['th']
        self.rv_lowres = Weibull(params['theta'][0], params['theta'][1])

    def get_lowres_score(self):
        return self.rv_lowres.rvs()

    def get_highres_score(self, lowres_score):
        return np.random.rand()*lowres_score

    def simulate(self):
        T = self.params['T']
        cl = self.params['cl']
        ch = self.params['ch']

        t = 0
        history = list()
        sH_max = 0.
        while t < T:
            # Random sampling & low-res scan
            if t+cl >= T: break
            t += cl
            sL = self.get_lowres_score()
            if sL > self.th:
                if t+ch >= T: break
                t += ch
                sH = self.get_highres_score(sL)
            else:
                sH = np.nan
            history.append((t, sL, sH))
            if sH > sH_max:
                sH_max = sH
        return sH_max, history

    def average_reward(self, num_trial):
        result = list()
        for i in range(num_trial):
            result.append(self.simulate()[0])
        return np.mean(result)

class Objective_Best(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.current_best = 0.
    def update(self, newval):
        #print(self.current_best, newval)
        # newval: scalar expected
        self.current_best = max(self.current_best, newval)
        return self.current_best
    def without_update(self, newval):
        # newval: could be a vector
        return np.maximum(self.current_best, newval)

class Objective_NthBest(object):
    def __init__(self, n=2):
        assert n >= 2
        self.n = n
        self.reset()
    def reset(self):
        self.best_n = np.zeros(n)
    def update(self, newval):
        # newval: scalar expected
        if newval > self.best_n[-1]:
            self.best_n[-1] = newval
            self.best_n = np.sort(self.best_n)
        return self.best_n[-1]
    def without_update(self, newval):
        # newval: vector expected
        result = np.zeros(newval.size)
        # if newval is worse than best_n[-1], then the new nth best does not change
        worse = newval <= self.best_n[-1]
        result[worse] = self.best_n[-1]

        # if newval is better than best_n[-1], there are two cases: better than best_n[-2] or not
        better_2 = newval > self.best_n[-2]
        result[better_2] = best_n[-2]

        newval_nthbest = np.logical_and(np.logical_not(better_2), np.logical_not(worse))
        result[newval_nthbest] = newval[newval_nthbest]

        return newval_nthbest

class Env_standard(object):
    def __init__(self, params, obj=None, rv_lowres=None):
        #self.params = params
        self.T = params['T']
        self.ct = params['ct']
        self.cl = params['cl']
        self.ch = params['ch']
        self.th = params['th']
        self.prob_peaks = params['prob_peaks']

        if obj==None:
            self.obj = Objective_Best()
        else:
            self.obj = obj

        if rv_lowres is None:
            raise ValueError('rv_lowres should be given.')
        #self.rv_lowres = Weibull(params['theta'][0], params['theta'][1])

        if 'current_best' in self.params.keys():
            self.initial_current_best = self.params['current_best']
        else:
            self.initial_current_best = 0.
        self.reset()

    def get_lowres_score(self, num_samples=None):
        if num_samples is not None:
            return (np.random.rand(num_samples) < self.prob_peaks) * self.rv_lowres.rvs(size=num_samples)
        else:
            return (np.random.rand() < self.prob_peaks) * self.rv_lowres.rvs()


    def get_highres_score(self, lowres_score):
        return np.random.rand(np.size(lowres_score))*lowres_score

    def reset(self):
        self.t = self.T
        self.current_best = self.initial_current_best
        self.current_sL = 0.
        self.history_sH = list()
        self.obj.reset()

    def get_state(self):
        #return (self.t, self.current_best)
        return (self.t, self.current_best, self.current_sL)

    def step(self, action):
        info = None
        self.t -= self.ct # time for moving to the next point and taking a trace
        if self.t-self.cl < 0: # not sufficent time to get a low-res scan
            done = True
            reward = 0.
            self.current_sL = 0.

        else:
            done = False
            self.t -= self.cl # time for measuring a low-res scan
            self.current_sL = self.get_lowres_score() # low-res score

            if self.current_sL < action or self.t - self.ch < 0:
                # do not measure high-res scan if low-res score is less than the threshold, or there is no time to do it
                #done = True
                reward = 0.
            else:
                sH = self.get_highres_score(self.current_sL)
                self.history_sH.append(sH)

                new_best = self.obj.update(np.asscalar(sH))
                reward = new_best - self.current_best
                self.current_best = new_best
                self.t -= self.ch # time for measuring a high-res scan

        return [self.get_state(), reward, done, info]

    def simulate(self):
        self.reset()
        total_reward = 0.
        history = list()
        while True:
            state, reward, done, info = self.step(self.th)
            total_reward += reward
            history.append(state)
            if done: break
        #return total_reward, history
        return self.current_best, history

    def average_reward(self, num_trial):
        result = list()
        for i in range(num_trial):
            result.append(self.simulate()[0])
        return np.mean(result)

class Env_AveReward(Env_standard):
    '''
    Return the average reward at the state.
    '''
    def step(self, action):
        # calculate the average reward at the current state
        num_ave = 100
        if self.t - (self.ct + self.cl + self.ch) <= 0.:
            reward = 0.
        else:
            sL_samples = self.get_lowres_score(num_samples=num_ave)
            #above_th = sL_samples >= action
            #reward_samples = np.zeros(num_ave)
            #for i in range(num_ave):
            #    if above_th[i]:
            #        reward_samples[i] = self.obj.without_update(self.get_highres_score(sL_samples[i])) - self.current_best
            reward_samples = self.obj.without_update(self.get_highres_score(sL_samples * (sL_samples >= action))) - self.current_best
            reward = np.mean(reward_samples)


        info = None
        self.t -= self.ct # time for moving to the next point and taking a trace
        if self.t-self.cl < 0: # not sufficent time to get a low-res scan
            done = True
            self.current_sL = 0.

        else:
            done = False
            self.t -= self.cl # time for measuring a low-res scan
            self.current_sL = self.get_lowres_score() # low-res score

            if self.current_sL < action or self.t - self.ch < 0:
                # do not measure high-res scan if low-res score is less than the threshold, or there is no time to do it
                pass
            else:
                sH = self.get_highres_score(self.current_sL)
                self.history_sH.append(sH)

                new_best = self.obj.update(np.asscalar(sH))
                self.current_best = new_best
                self.t -= self.ch # time for measuring a high-res scan


        return [self.get_state(), reward, done, info]

# https://gist.github.com/plasmaman/5508278
def fitweibull(x):
    def optfun(theta):
        # theta = [k, lambda]
        return -np.sum(exponweib.logpdf(x, 1, theta[0], scale = theta[1], loc=0))
    logx = np.log(x)
    shape = 1.2 / np.std(logx)
    scale = np.exp(np.mean(logx) + (0.572 / shape))
    return fmin(optfun, [shape, scale], disp = 0)


save_dir = Path('./save_Dominic_okscore_randomHS_origin100_2')

scores = np.loadtxt(save_dir/'result.csv', delimiter=',')
print(scores.shape)

num_examples = scores.shape[1]

peaks = scores[0,:]
score_lowres = scores[1,:]
score_highres = scores[2,:]

print(np.sum(score_lowres > 0.))
score_lowres_nonzero = score_lowres[score_lowres > 0.]

'''
theta = fitweibull(score_lowres_nonzero)
print('Theta: ', theta)

theta[0] *= 3. 

plt.figure()
x = np.linspace(0.0000001,0.5,100)
plt.subplot(1,2,1)
y = exponweib.pdf(x, 1, theta[0], scale=theta[1], loc=0)
plt.plot(x,y)
plt.ylim([0., 1.])
plt.title('pdf')
plt.subplot(1,2,2)
y = exponweib.cdf(x, 1, theta[0], scale=theta[1], loc=0)
plt.plot(x,y)
plt.ylim([0., 1.])
print(y)
plt.title('cdf')
plt.savefig('distribution')
plt.close()

#raise ValueError()
'''
empirical = Empirical(score_lowres_nonzero)

ct = 30 # time for moving to the next location and taking a trace
cl = 20 # time for low-res scan and score
ch = 200 # time for high-res scan and score
T_max = 6 * 3600 # total 6hr

T_all = np.linspace(T_max/10,T_max, 30)
th_all = np.linspace(0.00, 0.1, 30)

T_grid, th_grid = np.meshgrid(T_all, th_all, indexing='ij')
pairs = np.stack((T_grid.ravel(), th_grid.ravel()), axis=1)
#raise ValueError()

prob_peaks = np.sum(score_lowres > 0.) / num_examples
params_all = [{'ct': ct, 'cl':cl, 'ch':ch, 'prob_peaks': prob_peaks, 'T':T, 'th':th} for (T, th) in pairs] 


def func(param):
    #env = Env_threshold(param)
    #env = Env_standard(param)
    env = Env_AveReward(param, rv_lowres=empirical)
    return env.average_reward(100)

tic = time.time()

#result = list()
#for param in params_all:
#    #env = Env_standard(param)
#    env = Env_AveReward(param)
#    result.append(env.average_reward(10))
with Pool(22) as p:
    result = p.map(func, params_all)
print('Elapsed time: ', time.time() - tic)

result = np.array(result).reshape(T_grid.shape)
np.savez(str(save_dir/'expectedreward.npy'), val=result, T_grid=T_grid, th_grid=th_grid)
plt.figure()
plt.imshow(result.transpose(), origin='lower', aspect='auto',  extent=[ T_all.min()/3600, T_all.max()/3600, th_all.min(), th_all.max()])
plt.xlabel('Total allowed time')
plt.ylabel('Threshold')
plt.colorbar()
plt.savefig(save_dir/'expectedreward')
plt.close()
