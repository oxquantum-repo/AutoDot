# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:28:23 2019

@author: thele
"""
from . import measurement_functions
from . import condition_functions
import numpy as np



class Investigation_stage():
    def __init__(self,jump,measure,check,configs,timer,pygor=None):
        self.jump = jump
        self.measure = measure
        self.check = check
        
        self.configure_investigation_sequence(configs)
        
        self.inv_max = len(self.aquisition_functions)
        self.isdynamic = configs.get('cond_meas',[False]*self.inv_max)
        
        self.stage_results = []
        self.timer = timer
        
        self.pygor = pygor
        
        self.cond=list(range(1,1+self.inv_max))
        
        self.conditional_idx_list = []
        
        
    def configure_investigation_sequence(self,configs):
        seq_keys = configs["measurement_seq"]
        
        self.aquisition_functions = []
        self.function_configs = []
        self.cond_functions = []
        
        
        for seq_key in seq_keys:
            
            afunc_name = configs[seq_key].get('func','do_nothing')
            cfunc_name = configs[seq_key].get('condition','check_nothing')
            self.aquisition_functions += [getattr(measurement_functions,afunc_name)]
            self.cond_functions += [getattr(condition_functions,cfunc_name)]
            self.function_configs += [configs[seq_key]]
            
    def do_extra_measure(self,params,minc,maxc,**kwags):
        self.jump(params)
        self.timer.start()
        
        plunger_jump = lambda params:self.jump(params,True)
        kwags['pygor'] = self.pygor
        anchor_vals = self.check()
        
        results_full = {}
        results = []
        
        all_resutls = [None]*self.inv_max
        for i in range(self.inv_max):
            data = self.aquisition_functions[i](plunger_jump,self.measure,anchor_vals,self.function_configs[i],**kwags)
            self.timer.logtime()
            
            check_result,continue_on,meta_info = self.cond_functions[i](data,minc,maxc,self.function_configs[i],**kwags)
            
            if len(self.stage_results)>0:
                np.array(self.stage_results,dtype=np.float)
                past_results = np.array(self.stage_results,dtype=np.float)[:,i]
                continue_on = bool_cond(check_result,past_results[~np.isnan(past_results)],**self.isdynamic[i])  if isinstance(self.isdynamic[i],dict) else continue_on
            
            all_resutls[i] = check_result
            
            if isinstance(meta_info,dict):
                new_kwags = meta_info.get('kwags',None)
                if new_kwags is not None:
                    kwags['last_check'] = new_kwags
            
            results += [[check_result,continue_on,meta_info,data]]
            
            if not continue_on:
                break
            
        self.stage_results += [all_resutls]    
        self.timer.stop()
        
        results_full['extra_measure'] = results
        results_full['conditional_idx'] = self.cond[i]
        results_full['times'] = self.timer.times_list[-1]
        
        return results_full
      
        
        
def bool_cond(score,past,min_thresh=0.0001,min_data=10,quantile=0.85):
    th_score = np.maximum(min_thresh, np.quantile(past, quantile)) if len(past)>min_data else min_thresh
    print("Score thresh: ",th_score)
    return score>=th_score
        
        
            
            
            
