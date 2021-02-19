# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:25:59 2021

@author: thele
"""

from .GPy_wrapper import GPyWrapper_Classifier as GPC
from . import GP_util
import numpy as np
from ...main_utils.dict_util import Tuning_dict
from . import util




class GP_base():
    def __init__(self,n,bound,origin,configs):
        
        print(configs)
        
        GP_type = configs.get('type', 'gpr')
        
        self.GP_type = GP_type
        self.n,self.bound,self.origin=n,np.array(bound),np.array(origin)
        
        self.c = Tuning_dict(configs)
        self.create_gp()
        
        
    def create_gp(self):
        l_p_mean, l_p_var = self.c.get('length_prior_mean','length_prior_var')
        n = self.n
        l_prior_mean = l_p_mean * np.ones(n)
        l_prior_var = (l_p_var**2) * np.ones(n)
        
        if self.GP_type == 'gpr':
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
        if self.GP_type == 'gpr':
            return self.gp.predict_f(x)
        else:
            return self.gp.predict_prob(x)
        
        
        
        
        
        
        
        
class GPR():
    def __init__(self,n,bound,origin,configs):
        
        cprior = configs.get('cprior', 'gpr')
        sprior = configs.get('sprior', 'gpr')
        self.restarts = configs.get('restarts', 5)
        self.noise_var = configs.get('noise_var', 0.05)
        
        self.n,self.bound,self.origin=n,np.array(bound),np.array(origin)
        
        self.gp = GP_util.create_GP(self.n, **cprior)
        GP_util.set_GP_prior(self.gp, **sprior) # do not set prior for kernel var
        GP_util.fix_hyperparams(self.gp, False, True)
            
            
    def train(self, x, y,*args,**kwargs):
        self.gp.create_model(x, y, self.noise_var, *args, **kwargs)
        
    def optimise(self,**kwargs):
        self.gp.optimize(self.restarts ,parallel=True,**kwargs)

        
    def predict(self,x):
        return self.gp.predict_f(x)
    
    
class U_vec_GPR_batch_norm(GPR):
    
    def train(self, x, y,*args,**kwargs):
        U, r = util.ur_from_vols_origin(x, self.origin, returntype='array')
        
        self.gp.create_model(U, y, self.noise_var, *args, **kwargs)
    
    
    def predict(self,x):
        U, r = util.ur_from_vols_origin(x, self.origin, returntype='array')
        pred = self.gp.predict_f(U)[0]
        pred[pred<0] = 0.0
        return pred/np.sum(pred)
        