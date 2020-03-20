import threading
import multiprocessing
import time
import numpy as np
from collections import deque
from .BO_common import lhs_hypersphere, random_hypersphere, random_hypercube

class Gaussian_proposal_move(object):
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

def ur_from_v(v, origin):
    v_origin = v - origin
    r = np.sqrt(np.sum(np.square(v_origin),axis=1))
    u = v_origin / r[:,np.newaxis]
    return u, r

class TesterCross(object):
    def __init__(self, gp=None, origin=None):
        '''
        Args:
            gp: gp model that has predict_f
        '''
        self.gp = gp
        self.origin = origin
    def __call__(self, a, b):
        '''
        Args:
            a : array of shape (num_samples, ndim)
            b : array of shape (num_samples, ndim)
        Returns:
            1D boolean array (num_samples) indicating whethere each line crosses the boundary
        '''
        u_a, r_a = ur_from_v(a, self.origin)
        u_b, r_b = ur_form_v(b, self.origin)

        r_a_surf, _  = self.gp.predict_f(u_a)
        r_b_surf, _  = self.gp.predict_f(u_b)

        inside_a = r_a < r_a_surf[:,0]
        inside_b = r_b < r_b_surf[:,0]

        cross = np.logical_xor(inside_a, inside_b)

        return cross

class TesterInside(object):
    def __init__(self, gp=None, origin=None, directions=None):
        self.gp = gp
        self.origin = origin
        self.directions = directions
    def __call__(self, z):
        u, r = ur_from_v(z, self.origin)
        r_surf, _ = self.gp.predict(u)

        return r < np.maximum( r_surf[:,0], 0.0)

class TesterBoundary(object):
    def __init__(self, lb, ub, conditions=tuple()):
        self.lb, self.ub = lb, ub
        self.conditions = conditions

    def __call__(self, z):
        '''
        Args:
            z : array of shape (num_samples, ndim)
        '''
        ax = 0
        ge_lb = np.all(z >= self.lb, axis=1)
        le_ub = np.all(z <= self.ub, axis=1)
        inside = np.logical_and(ge_lb, le_ub)
        # more conditions
        for cond in self.conditions:
            inside = np.logical_and(inside, cond(z))
        return inside

class LikelihoodBoundary(object):
    def __init__(self, tester_boundary):
        self.tester_boundary = tester_boundary
    def __call__(self, z, normalize=False, log_mode=False):
        if normalize: raise NotImplementedError()

        L = self.tester_boundary(z).astype(float)
        if log_mode: L = np.clip(np.log(L), -100, None)
        return L

#class MH_MCMC_Hypersurface(threading.Thread):
class MH_MCMC_Hypersurface(multiprocessing.Process):
    # TODO: add logging functionality
    def __init__(self, move, L_func, tester_inside, log_mode=False, history=False):
        super().__init__()
        self.move = move
        self.L_func = L_func
        self.tester_inside = tester_inside
        self.log_mode = log_mode
        self.z = None
        self.L_current = None
        self.history = history

    def reset(self, z=None, L_current=None, max_steps=1000, stopper=None, result_sender=None):
        if self.z is None and z is None:
            raise ValueError('z should be given.')
        if z is not None:
            self.z = z
        elif self.z is None:
            #self.z = random_points_inside(self.ndim, num_samples, self.gp_r, self.origin, factor=0.5)
            raise ValueError('No initial points are set.')
        if self.history:
            self.history_all = list()
            self.history_all.append(self.z)
        else:
            self.history_all = None
            

        # Check whether the samples are valid
        if not isinstance(self.z, np.ndarray): raise ValueError('Incompatible sample shape')
        if self.z.ndim != 2: raise ValueError('samples should be 2-dimensional array')
        inside = self.tester_inside(self.z) # test whether all points are inside of the surface
        
        #print(self.z,inside)
        if not np.all(inside):
            raise ValueError('At least one of initial points is outside of the surface.')

        # Initial likelihood
        if self.L_current is None:
            self.L_current = self.L_func(self.z, normalize=False, log_mode=self.log_mode)

        self.max_steps = max_steps
        self.stopper = stopper
        self.result_sender = result_sender
        #self.counter = multiprocessing.Value('i',0)
        self.counter = 0
        # for saving results
        self.queue = deque(maxlen=10000) # each item is a cross (trajectory index, point a, point b)

    def single_step(self):
        '''
        Update internal states: z, queue
        '''
        # Proposal move and acceptance probability
        z_proposal = self.move(self.z) # random move
        L_new = self.L_func(z_proposal, normalize=False, log_mode=self.log_mode) # evaluate the likelihood at the proposed positions
        if self.log_mode is True:
            acceptance_prob = np.clip(np.exp(L_new-self.L_current),0.0,1.0)
        else:
            acceptance_prob = np.clip(L_new/(self.L_current+1.0E-10),0.0,1.0)

        # Test the proposed points cross the boundary.
        inside_proposal = self.tester_inside(z_proposal)
        cross = np.logical_not(inside_proposal)
        cross = np.logical_and(cross, L_new > 1.e-10)
        #cross = np.logical_and(cross, np.all(z_proposal>-2000, axis=1))
        idxs = np.nonzero(cross)[0]

        for idx in idxs:
            self.queue.append((idx, self.z[idx], z_proposal[idx]))

        # Acceptance probability is zero if a point is outside
        acceptance_prob[cross] = 0.0

        # Accept the moves with the probability
        accept = np.random.uniform(size=acceptance_prob.shape) < acceptance_prob

        z_new = np.copy(self.z)
        z_new[accept] = z_proposal[accept]

        self.z = z_new
        self.L_current[accept] = L_new[accept] #update L_current

        if self.history:
            self.history_all.append(self.z)

    def get_result(self):
        return self.counter, self.z, list(self.queue)

    #def __call__(self, z=None, L_current = None, max_steps = 1000):
    #    self.reset(z, L_current, max_steps=max_steps)
    def __call__(self):
        for i in range(self.max_steps):
            self.single_step()
            self.counter += 1
        return self.get_result()

    def run(self):
        for i in range(self.max_steps):
            #if self.stopper is not None and self.stopper.is_set(): break
            if self.stopper is not None and self.stopper.value == 1: break
            self.single_step()
            self.counter += 1
        if self.result_sender is not None:
            self.result_sender.send(self.get_result())

def create_sampler(gp_r, origin, lb, ub, sigma=50, history=False):
    move_big = Gaussian_proposal_move(cov=sigma**2)
    tester_inside = TesterInside(gp_r, origin)
    tester_boundary = TesterBoundary(lb, ub)
    likelihood_boundary = LikelihoodBoundary(tester_boundary)
    mcmc = MH_MCMC_Hypersurface(move_big, likelihood_boundary, tester_inside, history=history)
    return mcmc


def random_points_inside_(ndim, num_samples, gp_r, origin, factor=0.5):
    u_samples = random_hypersphere(ndim, num_samples)
    if gp_r.model is not None:
        r_samples, _ = gp_r.predict_f(u_samples)
    else:
        r_samples = gp_r.center
    points_inside = u_samples * factor * r_samples + origin
    return points_inside

def random_points_inside(ndim, num_samples, gp_r, origin, lb, ub):
    samples = random_hypercube(lb, ub, num_samples)
    if gp_r.gp.model is not None:
        samples = project_points_to_inside(samples, gp_r, origin, 0.99)

    return samples

def pick_from_boundary_points(boundary_points, pick_last=True):
    idxs = [item[0] for item in boundary_points]
    idxs_unique = np.unique(idxs)
    idxs_selected = idxs_unique[np.random.randint(len(idxs_unique))]
    candidates = [item for item in boundary_points if item[0] == idxs_selected]
    if pick_last:
        one_item = candidates[-1]
    else:
        one_item = candidates[np.random.randint(len(candidates))]
    v = one_item[1] # it is slightly inside of the hypersurface
    return v

def project_points_to_inside(v, gp, origin, factor=0.5):
    u, r = ur_from_v(v, origin)
    r_surf, _ = gp.predict(u)
    idxs_outside = np.nonzero(r_surf[:,0] < r)[0]
    if idxs_outside.size > 0:
        v[idxs_outside] = u[idxs_outside]*factor*np.maximum(r_surf[idxs_outside],0.0) + origin
    return v

def project_points_to_boundary(v, gp, origin):
    u, r = ur_from_v(v, origin)
    r_surf, _ = gp.predict(u)

    v_boundary = u * r_surf + origin
    return v_boundary

def project_crosses_to_boundary(result_sampling, gp, origin):
    '''
    Args:
        result_sampling: list of (idx, point_inside, point_outside)
    '''
    # Simply map outside points to the boundary
    points_inside = np.array([item[2] for item in result_sampling])
    return project_points_to_boundary(points_inside, gp, origin)

def get_full_data(u_all, tester, step_back, origin=0.0, penalty_non_poff=0.0, do_extra_meas=None, save_dir=None):
    # Latin hypercube sampling (for initial measurement)

    r_all = list()
    d_all = list()
    poff_all = list()
    detected_all = list()
    time_all = list()
    extra_meas_all = list()
    for i, u in enumerate(u_all):
        print('Initial random iteration: ', i)
        print(u)

        t = time.time()
        r, vols_pinchoff, found, t_firstjump = tester.get_r(u, origin=origin) # Measure the distance
        d_vec, poff_vec, meas_each_axis, vols_each_axis = tester.measure_dist_all_axis(vols_pinchoff+step_back, penalty_non_poff=penalty_non_poff)

        r_all.append(r)
        detected_all.append(found)
        d_all.append(d_vec)
        poff_all.append(poff_vec)
        print('vols: ', vols_pinchoff, 'd_all: ', d_all[-1])
        if do_extra_meas is not None:
            extra_meas_all.append(meas_each_axis+do_extra_meas(vols_pinchoff))
        if save_dir is not None:
            save(save_dir, np.array(u_all), [], np.array(r_all), np.array(d_all), np.array(poff_all), detected_all, tester.logger, extra_meas_all, np.array([]), time_all, [])
        
        elapsed = time.time() - t
        print('Elapsed time: ', elapsed)
        time_all.append(elapsed)

    r_all = np.array(r_all)
    d_all = np.array(d_all)
    poff_all = np.array(poff_all)
    return u_all, r_all, d_all, poff_all, detected_all, time_all, extra_meas_all
