import multiprocessing
from pathlib import Path
import pickle
from multiprocessing import Pool

import numpy as np

import warnings


import Sampling.gp.util as util
from Sampling.test_common import Tester
from Sampling.BO_common import random_hypersphere
from Sampling.gp.GPy_wrapper import GPyWrapper_Classifier as GPC
import Sampling.gp.GP_util as GP_util

import Sampling.random_walk as rw




class Tuning_dict(dict):
    def __init__(self,*args,**kwargs):
        super(Tuning_dict,self).__init__(*args,**kwargs)
        
    def get(self,*var):
        return tuple(self[k] for k in var)
    
    def getd(self,*var):
        return dict(tuple((k,self[k]) for k in var))
    
    def getl(self,*var):
        cand = [self[k] for k in var]
        return tuple(c[-1] if isinstance(c,list) else c for c in cand)
 
    
    def app(self,**kwargs):
        dict_return = {}
        for key,item in kwargs.items():
            try:
                self[key].append(item)
                dict_return[key] = self[key]
            except AttributeError:
                raise warnings.warn("%s is not a list")
        return dict_return
    
    def add(self,**kwargs):
        for key,item in kwargs.items():
            self[key] = item
            
    def save(self,file_pth=None):
        if file_pth is None:
            file_pth = self['save_dir']+self['file_name']
        with open(file_pth,'wb') as h:
            pickle.dump(self,h)
            
            
            

                
                
        





class Base_Sampler(object):
    def __init__(self,configs):
        
        
        self.t = Tuning_dict(configs)
        
        self.all_data = []
        self.t.add(all_data=[],file_name="tuning.pkl",iterc=0)
        
        
        self.config(configs)
        
    def config(self,configs):
        
        self.t['iter'], self.iter=0, 0 #nn
        
        #Create the save directory
        self.save_dir = Path(configs['save_dir']) #nn
        self.save_dir.mkdir(exist_ok=True) #nn
    
        #Unpack control classes/functions

        
        self.t.add(**configs['general'])
        self.t.add(**get_real_bounds(*self.t.get('origin','bound')))

        
        self.t.add(n=len(self.t['origin']))
    
        self.set_detector_configs(configs['detector'])
    
        
        
        self.investigation_stage = configs['investigation_stage_class']
        self.do_extra_meas = lambda vols, th: self.investigation_stage.do_extra_measure(vols,self.minc,self.maxc, score_thresh=th)
        
        self.t.add(do_extra_meas=self.do_extra_meas)
        
        
        
        
         
    def set_detector_configs(self,configs):
        
        if configs.get('minc',None) is None:
            self.maxc, self.minc = get_current_domain(*self.t.get('jump', 'measure', 'origin', 'bound'))
        else:
            self.maxc,self.minc = configs.get('maxc',None),configs.get('minc',None)
        
        #print(minc,maxc)
    
        p_low_thresh,p_high_thresh = configs['th_low'],configs['th_high']
        
        threshold_low =  self.minc + ((self.minc+self.maxc)*p_low_thresh)
        threshold_high =  self.minc + ((self.minc+self.maxc)*p_high_thresh)

        detector_pinchoff = util.PinchoffDetectorThreshold(threshold_low)
        detector_conducting = util.ConductingDetectorThreshold(threshold_high)
        self.tester = Tester(*self.t.get('jump', 'measure', 'real_lb', 'real_ub'), detector_pinchoff, d_r=configs['d_r'], len_after_pinchoff=configs['len_after_poff'], logging=True, detector_conducting=detector_conducting)
        
        
        
        
class GP_base():
    def __init__(self,n,bound,origin,configs,GP_type=False):
        
        self.GP_type = GP_type
        self.n,self.bound,self.origin=n,np.array(bound),np.array(origin)
        
        self.c = Tuning_dict(configs)
        self.create_gp()
        
        
    def create_gp(self):
        l_p_mean, l_p_var = self.c.get('length_prior_mean','length_prior_var')
        n = self.n
        l_prior_mean = l_p_mean * np.ones(n)
        l_prior_var = (l_p_var**2) * np.ones(n)
        
        if self.GP_type == False:
            r_min, var_p_m_div = self.c.get('r_min','var_prior_mean_divisor')
            r_min, r_max =  r_min, np.linalg.norm(self.bound-self.origin)
            v_prior_mean = ((r_max-r_min)/var_p_m_div)**2
            self.gp = GP_util.create_GP(self.n, *self.c.get('kernal'), v_prior_mean, l_prior_mean, (r_max-r_min)/2.0)
            GP_util.set_GP_prior(self.gp, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
            GP_util.fix_hyperparams(self.gp, False, True)
        else:
            v_prior_mean, v_prior_var =  self.c.get('var_prior_mean','var_prior_var')
            v_prior_var = v_prior_var**2
            self.gp = GP_util.create_GP(self.n, *self.c.get('kernal'), lengthscale=l_prior_mean, const_kernel=True, GP=GPC)
            GP_util.set_GP_prior(self.gp, l_prior_mean, l_prior_var, v_prior_mean, v_prior_var)
            
            
    def train(self,x,y,*args,**kwargs):
        self.gp.create_model(x, y, *args, **kwargs)
        
    def optimsie(self):
        if self.GP_type == False:
            self.gp.optimize(self.c.get('restarts'),parallel=True)
        else:
            self.gp.optimize()
        #inside = get_between(self.origin,self.bounds,points_poff)
        #u_all_gp, r_all_gp = util.ur_from_vols_origin(points_poff[inside], self.origin, returntype='array')
        #self.gp.create_model(u_all_gp, r_all_gp[:,np.newaxis], (self.tester.d_r/2)**2, noise_prior='fixed')
        
    def predict(self,x):
        if self.GP_type == False:
            return self.gp.predict_f(x)
        else:
            return self.gp.predict_prob(x)
        
        
        
class GPC_heiracical():
    def __init__(self,n,bound,origin,configs):
        
        self.gp = []
        for config in configs:
            self.gp += [GP_base(n,bound,origin,config,GP_type=True)]
    def train(self,x,y_cond):
        for i,gp in enumerate(self.gp):
            gp.train(x,y_cond[:,i])
            
    def optimise(self,parallel=False):
        
        if parallel:
            def f(i):
                self.gp[i].optimise()
            #pool = multiprocessing.Pool(multiprocessing.cpu_count())
            with Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(f, range(len(self.gp)))
                
            #results
        else:
            [gp.optimsie() for gp in self.gp]
            
    def predict(self,x):
        results = [gp.predict(x) for gp in self.gp]
        return results
    
    
    def predict_comb_prob(self,x):
        probs = self.predict(x)
        total_prob = np.prod(probs, axis=0)
        return total_prob
        
        
        




class Paper_sampler(Base_Sampler):
    
    def __init__(self,configs):
        super(Paper_sampler, self).__init__(configs)
        
        
        self.t.add(samples=None,point_selected=None,boundary_points=[],
                   vols_poff=[],detected=[],vols_poff_axes=[],poff=[],
                   cand_v=[],all_v=[],vols_pinchoff=[],d_vec=[],poff_vec=[],meas_each_axis=[],vols_each_axis=[],extra_measure=[],
                   vols_pinchoff_axes=[],vols_detected_axes=[],changed_origin=[])
        
        
        
        
        self.t.add(d_r=self.t['detector']['d_r'])#bodge
        
        self.gpr = GP_base(*self.t.get('n','bound','origin'),self.t['gpr'])
        
        
        gpc_list = self.t['gpc']['gpc_list']
        gpc_configs = self.t['gpc']['configs']
        
        if not isinstance(gpc_configs,list):
            num_gpc = np.sum(gpc_list)
            gpc_configs = [gpc_configs]*num_gpc
            
        self.gpc = GPC_heiracical(*self.t.get('n','bound','origin'),gpc_configs)
        
        self.t.add(**configs['gpr'],**configs['gpc'],**configs['sampling'],**configs['pruning'])
    

    
   
    def project_samples_inside(self):
        self.samples = rw.project_points_to_inside(self.samples, self.gp_r, self.origin, factor=0.99)
        samples_outside = np.logical_or(np.any(self.samples>self.real_ub, axis=1), 
                                            np.any(self.samples<self.real_lb, axis=1)) # Invalid samples
            
        self.samples[samples_outside] = self.origin + self.direction
    
        
        
        
    def do_iter2(self):
        
        i = self.t['iter']
        
        th_score=0.01
        
        do_optim = (i%11==11) and (i > 0)
        do_gpr, do_gpc, do_pruning = (i>self.t['gpc_start']) and self.t['gpc_on'], (i>self.t['gpr_start']) and self.t['gpr_on'], (self.t['pruning_stop']<=i) and self.t['pruning_on']
        do_gpr_p1, do_gpc_p1 = (i-1>self.t['gpr_start']) and self.t['gpr_on'], (i-1>self.t['gpc_start']) and self.t['gpc_on']
        
        print(do_optim,do_gpr,do_gpc,do_pruning)
        #pick a uvec and start sampling
        u, r_est = select_point(self.gpr, self.gpc, *self.t.get('origin', 'cand_v', 'all_v', 'directions'), do_gpr_p1, do_gpc_p1)
        sampler_hook = start_sampling(self.gpr, *self.t.get('samples', 'origin', 'real_ub', 'real_lb',
                                             'directions', 'n_part', 'sigma', 'max_steps')) if do_gpr_p1 else None
    
            
        #observe true r
        r, vols_pinchoff, found, t_firstjump = self.tester.get_r(u, origin=self.t['origin'], r_est=r_est)
        self.t.app(vols_pinchoff=vols_pinchoff, detected=found)
        
        prune_results = self.tester.measure_dvec(vols_pinchoff+(self.t['step_back']*self.t['directions'])) if do_pruning else [None]*4
        self.t.app(**dict(zip(('d_vec', 'poff_vec', 'meas_each_axis', 'vols_each_axis'),prune_results)))
            
        em_results = self.t['do_extra_meas'](vols_pinchoff, th_score) if found else {'conditional_idx':0}
        self.t.app(extra_measure=em_results)
        
        if sampler_hook is not None: self.t.add(**stop_sampling(*sampler_hook)) 
            
        if do_pruning: self.t.add(**util.compute_hardbound(self.t.getl('poff_vec', 'found', 'vols_pinchoff', 'step_back', 'origin', 'directions', 'bound')))
            

        X_train_all, X_train, _ = util.merge_data(*self.t.get('vols_pinchoff', 'detected', 'vols_pinchoff_axes', 'vols_detected_axes'))           
        
        print(self.t['extra_measure'])
        
        if do_gpr: train_gpr(self.gpr,*self.t.get('origin', 'bound', 'd_r'), X_train, optimise = do_optim or self.t['changed_origin'])
        if do_gpc: train_hgpc(self.gpc, X_train, unpack('conditional_idx',self.t['extra_measure']), self.t['gpc_list'], optimise = do_optim)
        
        #if self.iter>self.start_gpr and self.gpr_on and self.samples is not None:
        #    self.project_samples_inside()
            
        self.t['iter'] += 1
        
        
def select_point(hypersurface, selection_model, origin, cand_v, all_v, directions, use_selection=True, estimate_r=True):
    """selects a point to investigate using thompson sampling, uniform sampling or random angles
    depending on use_selection flag or is no cand_v are present
    Args:
        hypersurface:  model of the hypersurface
        selection_model: model of probability of observing desirable features
        origin: (list) current origin of search
        can_v: (list) containing all candidate points found on modeled hypersurface
        all_v: (list) containing all observed points real hypersurface
        directions: (list) multiplyers specifying directions of search
    Returns:
        unit vector
    """
    
    if len(cand_v) > 0 and use_selection:
        points_candidate = rw.project_crosses_to_boundary(cand_v, hypersurface, origin)
        v = choose_next(points_candidate, all_v, selection_model, d_tooclose = 20.)
    elif len(cand_v) != 0:
        v = rw.pick_from_boundary_points(cand_v)
    else:
        print('WARNING: no boundary point is sampled')
        return random_angle_directions(len(origin), 1, np.array(directions))[0], None
    v_origin = v - origin
    u = v_origin / np.sqrt(np.sum(np.square(v_origin)))
    return u, hypersurface.predict(u) if estimate_r else None





def start_sampling(hypersurface,samples,origin,real_ub,real_lb,directions,n_part,sigma,max_steps):
    """selects a point to investigate using thompson sampling, uniform sampling or random angles
    depending on use_selection flag or is no cand_v are present
    Args:
        hypersurface:  model of the hypersurface
        selection_model: model of probability of observing desirable features
        origin: (list) current origin of search
        can_v: (list) containing all candidate points found on modeled hypersurface
        all_v: (list) containing all observed points real hypersurface
        directions: (list) multiplyers specifying directions of search
    Returns:
        unit vector
    """
    directions, origin = np.array(directions), np.array(origin)
    if samples is None:
        samples = rw.random_points_inside(len(origin), n_part, hypersurface, origin, real_lb, real_ub)
        samples = (samples)*(-directions[np.newaxis,:])+origin-500#update sample directions with config directions
    sampler = rw.create_sampler(hypersurface, origin, real_lb, real_ub, sigma=sigma)
    stopper = multiprocessing.Value('i', 0)
    listener, sender = multiprocessing.Pipe(duplex=False)
    sampler.reset(samples, max_steps=max_steps,stopper=stopper, result_sender=sender)
    sampler.start()
        
    return sampler, stopper, listener

def stop_sampling(sampler,stopper,listener):
    """selects a point to investigate using thompson sampling, uniform sampling or random angles
    depending on use_selection flag or is no cand_v are present
    Args:
        hypersurface:  model of the hypersurface
        selection_model: model of probability of observing desirable features
        origin: (list) current origin of search
        can_v: (list) containing all candidate points found on modeled hypersurface
        all_v: (list) containing all observed points real hypersurface
        directions: (list) multiplyers specifying directions of search
    Returns:
        unit vector
    """
    stopper.value = 1
    counter, samples, boundary_points = listener.recv()
    sampler.join()
    return {'samples':samples,'boundary_points':boundary_points}


def train_gpr(model,origin,bounds,d_r,X,Y=None,optimise=False):
    
    origin, bounds = np.array(origin), np.array(bounds)
    if Y is None:
        inside = get_between(origin,bounds,X)
        U, r = util.ur_from_vols_origin(X[inside], origin, returntype='array')
        model.train(U, r[:,np.newaxis], (d_r/2)**2, noise_prior='fixed')
        
        
    if optimise:
        model.optimize(num_restarts=5, opt_messages=False, print_result=True)
        
        
def train_hgpc(model,X,Y_count,mapping,optimise=False):
    
    model_mapping = [i for i, x in enumerate(mapping) if x]
    
    conditional_labels = []
    for idx in Y_count:
        conditional_labels += [[idx>c for c in model_mapping]]
    conditional_labels = np.array(conditional_labels)
                

    model.train(X, conditional_labels)
    
    model.optimise()
        
        
    
    
        
    def draw_point(self,point_selected):
        
        
        if point_selected is not None:
                v = point_selected
                v_origin = v - self.origin
                u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        elif len(self.boundary_points) != 0:
                v = rw.pick_from_boundary_points(self.boundary_points) 
                v_origin = v - self.origin
                u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        else:
                print('WARNING: no boundary point is sampled')
                u = random_angle_directions(self.num_active_gates, 1, self.direction)[0]
                
        return u
    
    
def unpack(key,list_of_dict):
    return [d[key] for d in list_of_dict]

            

def predict_probs(points, gpc_list):
    
    probs = []
    for gpc in gpc_list:
        probs += [gpc.predict_prob(points)[:,0]]


    total_prob = np.prod(probs, axis=0)
    log_total_prob = np.sum(np.log(probs), axis=0)
    return total_prob, log_total_prob, probs

def choose_next(points_candidate, points_observed, gpc_dict, d_tooclose = 100.):
    points_observed = np.array(points_observed)
    if len(points_candidate) == 0: # No cadidate points
        return None, None, None

    # Exclude samples that are too close to observed points
    tooclose = np.any(
            np.all(np.fabs(points_candidate[:,np.newaxis,:] - points_observed[np.newaxis,...]) <= d_tooclose, axis=2),
            axis=1)
    nottooclose = np.logical_not(tooclose)

    if np.sum(nottooclose) == 0: # All points are too close to observed points
        return None, None, None

    points_reduced = points_candidate[nottooclose]
    prob = predict_probs(points_reduced, gpc_dict)[0]
    p = prob / np.sum(prob)
    idx = np.random.choice(len(points_reduced), p=p) # Thompson sampling
    point_best =  points_reduced[idx]
    return point_best

def random_angle_directions(ndim, nsample, directions):
    u = random_hypersphere(ndim, nsample)
    mult = -directions # u is already negative directions for all
    u = u * mult[np.newaxis,:]
    return u


def get_current_domain(jump,measure,origin,bound):
    
    jump(bound)
    minc = measure()
    
    jump(origin)
    maxc = measure()
    
    return maxc, minc


def get_between(origin,bound,points):
    
    points = np.array(points)
    
    ub = np.maximum(origin,bound)
    lb = np.minimum(origin,bound)
    
    inside = np.all(np.logical_and(points < ub,points > lb), axis=1)
    
    return inside


def get_real_bounds(origin,bound):
    real_b = {}
    real_b['real_ub'] = np.maximum(origin,bound)
    real_b['real_lb'] = np.minimum(origin,bound)
    return real_b

