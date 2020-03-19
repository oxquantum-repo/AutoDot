# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:19:24 2020

@author: Dominic
"""
from .GPy_wrapper import GPyWrapper_Classifier as GPC
from . import GP_util
import numpy as np
import multiprocessing 
from ...main_utils.dict_util import Tuning_dict

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
        
    def optimise(self,**kwargs):
        if self.GP_type == False:
            self.gp.optimize(*self.c.get('restarts'),parallel=True,**kwargs)
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
        
        self.built = [False]*len(configs)
        
        self.gp = []
        for config in configs:
            self.gp += [GP_base(n,bound,origin,config,GP_type=True)]
    def train(self,x,y_cond):
        x = np.array(x)
        for i,gp in enumerate(self.gp):
            
            use_to_train = np.array([True]*x.shape[0])if i == 0 else y_cond[:,i-1]
            count_pos = use_to_train[use_to_train].size
            count_pos_of_pos = np.sum(y_cond[use_to_train,i])
            print("There are %i training examples for model %i and %i are positive"%(count_pos,i,count_pos_of_pos))
            if count_pos>0:
                gp.train(x[use_to_train],y_cond[use_to_train,i])
                self.built[i] = True
            
    def optimise(self,parallel=False):
        
        if parallel:
            def f(i):
                self.gp[i].optimise()
            #pool = multiprocessing.Pool(multiprocessing.cpu_count())
            with Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(f, range(len(self.gp)))
                
            #results
        else:
            for i,gp in enumerate(self.gp):
                if self.built[i]:
                    gp.optimise()
            
    def predict(self,x):
        results = []
        
        for i,gp in enumerate(self.gp):
                if self.built[i]:
                    results += [gp.predict(x)]
        
        return results
    
    
    def predict_comb_prob(self,x):
        probs = self.predict(x)
        total_prob = np.prod(probs, axis=0)
        return total_prob