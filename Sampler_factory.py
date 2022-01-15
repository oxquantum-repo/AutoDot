from .main_utils.cmaes_config_angle import CMAESConfig
from .Sampling import evaluation_order
import time
import multiprocessing
from pathlib import Path

from .main_utils.dict_util import Tuning_dict
from .main_utils.utils import Timer, plot_conditional_idx_improvment
from .main_utils.model_surf_utils import show_gpr_gpc
import numpy as np
from sympy import Symbol, Or
from sympy.solvers.inequalities import reduce_rational_inequalities

from .Sampling.gp import util
from .Sampling.test_common import Tester
from .Sampling.BO_common import random_hypersphere
from .Sampling.gp.GP_models import GPC_heiracical, GP_base

from .Sampling import random_walk as rw

import cma
import math
import matplotlib.pyplot as plt


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
        self.timer = Timer()
        
        
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
        

class CMAES_sampler(Base_Sampler):
    def __init__(self, configs):
        super().__init__(configs)

        configs['cmaes'] = self.setup_cmaes(configs, len(self.t['origin']))

        self.t.add(d_r=self.t['detector']['d_r'])#bodge

        # TODO remove unnecessary
        self.t.add(samples=None,point_selected=None,boundary_points=[],
                   vols_poff=[],detected=[],vols_poff_axes=[],poff=[],poff_traces=[],
                   all_v=[],vols_pinchoff=[],d_vec=[],poff_vec=[],meas_each_axis=[],vols_each_axis=[],extra_measure=[],
                   vols_pinchoff_axes=[],vols_detected_axes=[],changed_origin=[],conditional_idx=[],r_vals=[],score=[])

        self.t.add(d_r=self.t['detector']['d_r'])#bodge
        self.gpr = GP_base(*self.t.get('n','bound','origin'),self.t['gpr'])

        self.t.add(**configs['cmaes'], **configs['pruning'], **configs['gpr'])


    def setup_cmaes(self, configs, dim):
        #TODO insert default parameter 
        #TODO undefined dimensions 
        configs_cma = configs['cmaes']
        evaluation_order_class = configs_cma.pop('evaluation_order', evaluation_order.EvaluationOrderAngle)
        if isinstance(evaluation_order_class, str):
            evaluation_order_class = getattr(evaluation_order,evaluation_order_class)
        evaluation_order_func = lambda population, curr_pos: evaluation_order_class(population, curr_pos).get_order()
        self.t.add(get_evaluation_order=evaluation_order_func)
        
        # SAMPLE UNITVECTORS WITH ANGLES
        dflt_config = CMAESConfig(configs['general']['lb_box'], configs['general']['ub_box'])
    
        cma_option_default = {
            "bounds": [dflt_config.get_lower_bound(), dflt_config.get_upper_bound()], 
            "popsize": 6}
        x0, sigma0 = configs_cma.pop('x0', dflt_config.get_x0()), configs_cma.pop('sigma0', dflt_config.get_sigma0())
        for option, value in cma_option_default.items():
            configs_cma.setdefault(option, value)

        self.cmaes = cma.CMAEvolutionStrategy(x0, sigma0, configs_cma)
        self.cmaes_config = dflt_config
        return {**configs_cma, 'x0': x0, 'sigma0': sigma0}


    def do_iter(self):
        
        th_score=0.01

        # PICK VECTORS
        vecs_angle = self.cmaes.ask()

        last_point = convert_angle_to_euclide_vector(self.cmaes_config.get_x0())
        if self.t['iter'] != 0:
            last_point = self.t['vols_pinchoff'][-1] 

        vecs = [convert_angle_to_euclide_vector(v_angle) for v_angle in vecs_angle]
        eval_order = self.t['get_evaluation_order'](np.array(vecs), last_point)
        for eval_idx in eval_order:
            self.timer.start()
            i = self.t['iter']
            print("------------###Child %s###------------"%(i % self.t['popsize']))

            do_optim = (i%11==0) and (i > 0)
            do_gpr, do_pruning = (i>self.t['gpr_start']) and self.t['gpr_on'], (self.t['pruning_stop']>i) and self.t['pruning_on']
            do_gpr_p1 = (i-1>self.t['gpr_start']) and self.t['gpr_on']
            print("GPR:",do_gpr,"GPR1:",do_gpr_p1,"Optim:",do_optim)

            # CONVERT VECTOR FROM ANGLE TO EUCLID
            v = vecs[eval_idx]
            u = v / np.sqrt(np.sum(np.square(v)))

            # CHECK IF LINE OF UNIT VECTOR IS IN BOUND
            in_bound = check_if_line_in_bound(u, self.t.get("real_lb")[0], self.t.get("real_ub")[0]) if self.t['pruning_on'] else True

            if in_bound:
                # ESTIMATE R AFTER FIRST OPTIMIZATION OF GPR (do_optim)
                r_est = estimate_r(u, self.gpr, do_gpr_p1) if i > 11 and do_gpr else None
                self.timer.logtime()

                # MAP VECTOR TO POINT IN PINCHOFF AREA
                r, vols_pinchoff, found, t_firstjump, poff_trace = self.tester.get_r(v, origin=self.t['origin'], r_est=r_est)
                self.timer.logtime()
                self.t.app(r_vals=r,vols_pinchoff=vols_pinchoff, detected=found, poff_traces=poff_trace)

                # CHECK FOR PRUNING
                prune_results = self.tester.measure_dvec(vols_pinchoff+(self.t['step_back']*np.array(self.t['directions']))) if do_pruning else [None]*4
                self.timer.logtime()
                self.t.app(**dict(zip(('d_vec', 'poff_vec', 'meas_each_axis', 'vols_each_axis'),prune_results)))

                # DO MEASUREMENTS
                em_results = self.t['do_extra_meas'](vols_pinchoff, th_score) if found else {'conditional_idx': 0, 'score': np.inf}
                self.timer.logtime()
                print("score: ", em_results['score'])
                self.t.app(extra_measure=em_results), self.t.app(conditional_idx=em_results['conditional_idx']), self.t.app(score=em_results['score'])

                # APPLY RESULTS FROM PRUNING-CECK
                if do_pruning: self.t.add(**util.compute_hardbound(*self.t.getl('poff_vec', 'detected', 'vols_pinchoff'), *self.t.get('step_back', 'origin', 'bound')))

                
            else:
                empty_parameters = ["r_vals", "vols_pinchoff", "d_vev", "poff_vec", "meas_each_axis", "vols_each_axis"]
                self.t.app(**dict(zip(empty_parameters, len(empty_parameters)*[None])))
                em_results = {'conditional_idx': 0, 'score': np.inf}
                self.t.app(extra_measure=em_results, conditional_idx=em_results['conditional_idx'], score=em_results['score'], detected=False)

            # TRAIN GPR
            X_train_all, X_train, _ = util.merge_data(*self.t.get('vols_pinchoff', 'detected', 'vols_pinchoff_axes', 'vols_detected_axes'))
            if do_gpr: train_gpr(self.gpr,*self.t.get('origin', 'bound', 'd_r'), X_train, optimise = do_optim or self.t['changed_origin'])

            self.t['times']=self.timer.times_list
            self.timer.stop()
            self.t.save(track=self.t['track'])
            
            self.t['iter'] += 1

        # TELL CMAES RESUTLS
        scores = np.zeros(len(vecs))
        for score_idx, eval_idx in enumerate(eval_order):
            scores[eval_idx] = self.t['score'][score_idx-len(vecs)]
        self.cmaes.tell(vecs_angle, scores)

        self.cmaes.disp()
        return self.t.getd(*self.t['verbose'])


    def plot(self, configs):
        max_score = max([-np.inf if np.isinf(sc) else sc for sc in self.t['score']])
        min_score = min([np.inf if np.isinf(sc) else sc for sc in self.t['score']])

        def score_to_color(sc):
            if np.isposinf(sc):
                return max_score
            elif np.isneginf(sc):
                return min_score
            dif = max_score - sc
            return dif/(max_score-min_score)
        x = [p[0] for p in self.t['vols_pinchoff']]
        y = [p[1] for p in self.t['vols_pinchoff']]
        z = [p[2] for p in self.t['vols_pinchoff']]
        colors = [score_to_color(sc) for sc in self.t['score']] 

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, c=colors, cmap='autumn')
        plt.show(block=True)
        plot_conditional_idx_improvment(self.t['conditional_idx'],configs)
        



class Paper_sampler(Base_Sampler):
    
    def __init__(self,configs):
        super(Paper_sampler, self).__init__(configs)
        
        self.sampler_hook = None #bodge
        
        self.t.add(samples=None,point_selected=None,boundary_points=[],
                   vols_poff=[],detected=[],vols_poff_axes=[],poff=[],poff_traces=[],
                   all_v=[],vols_pinchoff=[],d_vec=[],poff_vec=[],meas_each_axis=[],vols_each_axis=[],extra_measure=[],
                   vols_pinchoff_axes=[],vols_detected_axes=[],changed_origin=[],conditional_idx=[],r_vals=[])

        
        self.t.add(d_r=self.t['detector']['d_r'])#bodge
        
        self.gpr = GP_base(*self.t.get('n','bound','origin'),self.t['gpr'])
        
        
        gpc_list = self.t['gpc']['gpc_list']
        gpc_configs = self.t['gpc']['configs']
        
        if not isinstance(gpc_configs,list):
            num_gpc = np.sum(gpc_list)
            gpc_configs = [gpc_configs]*num_gpc
            
        self.gpc = GPC_heiracical(*self.t.get('n','bound','origin'),gpc_configs)
        
        self.t.add(**configs['gpr'],**configs['gpc'],**configs['sampling'],**configs['pruning'])
    
        
    def do_iter(self):
        
        i = self.t['iter']
        self.timer.start()
        
        th_score=0.01
        
        
        do_optim = (i%11==0) and (i > 0)
        do_gpr, do_gpc, do_pruning = (i>self.t['gpr_start']) and self.t['gpr_on'], (i>self.t['gpc_start']) and self.t['gpc_on'], (self.t['pruning_stop']>i) and self.t['pruning_on']
        do_gpr_p1, do_gpc_p1 = (i-1>self.t['gpr_start']) and self.t['gpr_on'], (i-1>self.t['gpc_start']) and self.t['gpc_on']
        print("GPR:",do_gpr,"GPC:",do_gpc,"prune:",do_pruning,"GPR1:",do_gpr_p1,"GPC1:",do_gpc_p1,"Optim:",do_optim)
        #pick a uvec and start sampling
        u, r_est = select_point(self.gpr, self.gpc, *self.t.get('origin', 'boundary_points', 'vols_pinchoff', 'directions'), do_gpr_p1, do_gpc_p1, d_tooclose = self.t['d_tooclose'])
        self.timer.logtime()
        self.sampler_hook = start_sampling(self.gpr, *self.t.get('samples', 'origin', 'real_ub', 'real_lb',
                                             'directions', 'n_part', 'sigma', 'max_steps'),sampler_hook=self.sampler_hook) if do_gpr_p1 else None

         
        r, vols_pinchoff, found, t_firstjump, poff_trace = self.tester.get_r(u, origin=self.t['origin'], r_est=r_est)
        self.timer.logtime()
        self.t.app(r_vals=r,vols_pinchoff=vols_pinchoff, detected=found, poff_traces=poff_trace)
        
        prune_results = self.tester.measure_dvec(vols_pinchoff+(self.t['step_back']*np.array(self.t['directions']))) if do_pruning else [None]*4
        self.timer.logtime()
        self.t.app(**dict(zip(('d_vec', 'poff_vec', 'meas_each_axis', 'vols_each_axis'),prune_results)))
        
        em_results = self.t['do_extra_meas'](vols_pinchoff, th_score) if found else {'conditional_idx':0}
        self.timer.logtime()
        self.t.app(extra_measure=em_results), self.t.app(conditional_idx=em_results['conditional_idx'])
        
        
        if self.sampler_hook is not None: self.t.add(**stop_sampling(*self.sampler_hook)) 
        
        if do_pruning: self.t.add(**util.compute_hardbound(*self.t.getl('poff_vec', 'detected', 'vols_pinchoff'), *self.t.get('step_back', 'origin', 'bound')))
        
        X_train_all, X_train, _ = util.merge_data(*self.t.get('vols_pinchoff', 'detected', 'vols_pinchoff_axes', 'vols_detected_axes'))           
        
        if do_gpr: train_gpr(self.gpr,*self.t.get('origin', 'bound', 'd_r'), X_train, optimise = do_optim or self.t['changed_origin'])
        self.timer.logtime()
        if do_gpc: train_hgpc(self.gpc, self.t['vols_pinchoff'], unpack('conditional_idx',self.t['extra_measure']), self.t['gpc_list'], optimise = do_optim)
        self.timer.logtime()
        
        if self.sampler_hook is not None and do_gpr:
            self.t.add(**project_samples_inside(self.gpr, *self.t.get('samples', 'origin', 'real_ub', 'real_lb')))
            
        self.timer.stop()
        self.t['times']=self.timer.times_list
        self.t['iter'] += 1
        self.t.save(track=self.t['track'])
     
        return self.t.getd(*self.t['verbose'])
    

    def plot(self, configs):
        fields = ['vols_pinchoff','conditional_idx','origin']
        show_gpr_gpc(self.gpr, configs, *self.t.get(*fields), gpc=self.gpc.predict_comb_prob)
        plot_conditional_idx_improvment(self.t['conditional_idx'],configs)
    
    
class Redo_sampler(Base_Sampler):
    
    def __init__(self,configs, point_cloud):
        super(Redo_sampler, self).__init__(configs)
        
        self.sampler_hook = None #bodge
        
        self.t.add(samples=None,point_selected=None,boundary_points=[],
                   vols_poff=[],detected=[],vols_poff_axes=[],poff=[],poff_traces=[],
                   all_v=[],vols_pinchoff=[],d_vec=[],poff_vec=[],meas_each_axis=[],vols_each_axis=[],extra_measure=[],
                   vols_pinchoff_axes=[],vols_detected_axes=[],changed_origin=[],conditional_idx=[],r_vals=[])
        
        self.t.add(d_r=self.t['detector']['d_r'])#bodge
        self.ulist = points_to_u(self.t['origin'], point_cloud)
        
    def do_iter(self):
        i = self.t['iter']
        self.timer.start()
        
        th_score=0.01
        
        #pick a uvec
        u = self.ulist[i]
        self.timer.logtime()
         
        r, vols_pinchoff, found, t_firstjump, poff_trace = self.tester.get_r(u, origin=self.t['origin'])
        self.timer.logtime()
        self.t.app(r_vals=r,vols_pinchoff=vols_pinchoff, detected=found, poff_traces=poff_trace)
        
        self.timer.logtime()
        
        em_results = self.t['do_extra_meas'](vols_pinchoff, th_score) if found else {'conditional_idx':0}
        self.timer.logtime()
        self.t.app(extra_measure=em_results), self.t.app(conditional_idx=em_results['conditional_idx'])
        
        self.timer.logtime()
        self.timer.logtime()
            
        self.timer.stop()
        self.t['times']=self.timer.times_list
        self.t['iter'] += 1
        self.t.save(track=self.t['track'])
        return self.t.getd(*self.t['verbose'])
    

    def plot(self, configs):
        fields = ['vols_pinchoff','conditional_idx','origin']
        show_gpr_gpc(self.gpr, configs, *self.t.get(*fields), gpc=self.gpc.predict_comb_prob)
        plot_conditional_idx_improvment(self.t['conditional_idx'],configs)
        
def select_point(hypersurface, selection_model, origin, boundary_points, vols_pinchoff, directions, gpr_in_use=True, gpc_in_use=True, d_tooclose = 20.):
    """selects a point to investigate using thompson sampling, uniform sampling or random angles
    depending on use_selection flag or is no samples are present
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
    
    boundary_points = [] if boundary_points is None else boundary_points
    
    if len(boundary_points) > 0 and gpc_in_use:
        points_candidate = rw.project_crosses_to_boundary(boundary_points, hypersurface, origin)
        v = choose_next(points_candidate, vols_pinchoff, selection_model, d_tooclose = d_tooclose)
    elif len(boundary_points) != 0:
        v = rw.pick_from_boundary_points(boundary_points)
    else:
        print('WARNING: no boundary point is sampled')
        return random_angle_directions(len(origin), 1, np.array(directions))[0], None
    v_origin = v - origin
    u = v_origin / np.sqrt(np.sum(np.square(v_origin)))

    return u.squeeze(), estimate_r(u, hypersurface, gpr_in_use)


def estimate_r(u, hypersurface, gpr_in_use):
    """
    estimates the associated point in the pinch-off area given an unit vector
    Args:
        u: unitvector
        hypersurface: model of hypersurface
        gpr_in_use: whether the approximation should be applied or not
    """
    r_est,r_std = hypersurface.predict(u[np.newaxis,:])
    r_est = np.maximum(r_est - 1.0*np.sqrt(r_std), 0.0)
    return r_est.squeeze() if gpr_in_use else None



def start_sampling(hypersurface,samples,origin,real_ub,real_lb,directions,n_part,sigma,max_steps,sampler_hook=None):
    """starts the sampling using multiprocessing while measurements are made on the device
    Args:
        hypersurface:  model of the hypersurface
        samples: (list) samples to use for the brownian motion
        origin: (list) current origin of search
        real_ub: (list) upper bound of search space
        real_lb: (list) lower bound of search space
        directions: (list) multiplyers specifying directions of search
    Returns:
        unit vector
    """
    print("START")
    if sampler_hook is not None: sampler, stopper, listener = sampler_hook
    directions, origin = np.array(directions), np.array(origin)
    if samples is None:
        samples = rw.random_points_inside(len(origin), n_part, hypersurface, origin, real_lb, real_ub)
        #print("S: ", samples,"O: ",origin)
        #samples = (samples)*(-directions[np.newaxis,:])+origin#update sample directions with config directions
    sampler = rw.create_sampler(hypersurface, origin, real_lb, real_ub, sigma=sigma)
    stopper = multiprocessing.Value('i', 0)
    listener, sender = multiprocessing.Pipe(duplex=False)
    sampler.reset(samples, max_steps=max_steps,stopper=stopper, result_sender=sender)
    sampler.start()
    #time.sleep(1)
        
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
    print("STOP")
    return {'samples':samples,'boundary_points':boundary_points}

def project_samples_inside(hypersurface, samples, origin, ub, lb):
    samples = rw.project_points_to_inside(samples, hypersurface, origin, factor=0.99)
    out_of_ub = samples>np.atleast_2d(ub)
    samples[out_of_ub] = samples[out_of_ub] - 1.0
    out_of_lb = samples<np.atleast_2d(lb)
    samples[out_of_lb] = samples[out_of_lb] + 1.0
    return {'samples':samples}
    
def train_gpr(model,origin,bounds,d_r,X,Y=None,optimise=False):
    
    origin, bounds = np.array(origin), np.array(bounds)
    if Y is None:
        inside = get_between(origin,bounds,X)
        U, r = util.ur_from_vols_origin(X[inside], origin, returntype='array')
        model.train(U, r[:,np.newaxis], (d_r/2)**2, noise_prior='fixed')
        
        
    if optimise:
        model.optimise(opt_messages=False, print_result=True)
        
        
def train_hgpc(model,X,Y_count,mapping,optimise=False):
    
    model_mapping = [i for i, x in enumerate(mapping) if x]
    
    conditional_labels = []
    for idx in Y_count:
        conditional_labels += [[idx>c for c in model_mapping]]
    conditional_labels = np.array(conditional_labels)
                

    model.train(X, conditional_labels)
    
    if optimise:
        model.optimise()
        
    
    
def unpack(key,list_of_dict):
    return [d[key] for d in list_of_dict]

            

def predict_probs(points, gpc_list):
    
    total_probs = gpc_list.predict_comb_prob(points)

    return total_probs.squeeze(), None, None

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
    
    idx = np.random.choice(len(points_reduced), p=p) #Thompson sampling
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

def points_to_u(origin, points):
    u_dirs = np.array(points) - np.array(origin)[np.newaxis,:]
    return u_dirs/np.linalg.norm(u_dirs,axis=-1)[:,np.newaxis]

def check_if_line_in_bound(u, lb, ub):
    # TODO replace OR check by something more meaningfull
    ineq = []
    alpha = Symbol('alpha', real=True)
    for i in range(len(u)):
        ineq.append([alpha*u[i] >= lb[i]])
        ineq.append([alpha*u[i] <= ub[i]])
    res = reduce_rational_inequalities(ineq, alpha)
    return not isinstance(res, Or)


def convert_angle_to_euclide_vector(v_angle):
    """"
    Converts an vector saved as angles between the components
    to an euclid vector.
    """
    v = [math.cos(v_angle[0]), math.sin(v_angle[0])]
    for i in range(1, len(v_angle)):
        scale_fac = v[i]/math.cos(v_angle[i])
        v.append(scale_fac*math.sin(v_angle[i]))

    return v / np.sqrt(np.sum(np.square(v)))
