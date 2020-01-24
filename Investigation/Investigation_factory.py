# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:28:23 2019

@author: thele
"""
import Investigation.measurement_functions as measurement_functions
import Investigation.condition_functions as condition_functions




class Investigation_stage():
    def __init__(self,jump,measure,check,configs,pygor=None):
        self.jump = jump
        self.measure = measure
        self.check = check
        
        self.configure_investigation_sequence(configs)
        
        self.inv_max = len(self.aquisition_functions)
        
        self.pygor = pygor
        
        self.cond=list(range(1,1+self.inv_max))
        
        
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
        plunger_jump = lambda params:self.jump(params,True)
        kwags['pygor'] = self.pygor
        anchor_vals = self.check()
        
        results_full = {}
        results = []
        
        for i in range(self.inv_max):
            data = self.aquisition_functions[i](plunger_jump,self.measure,anchor_vals,self.function_configs[i],**kwags)
            
            check_result,continue_on,meta_info = self.cond_functions[i](data,minc,maxc,self.function_configs[i],**kwags)
            
            
            if isinstance(meta_info,dict):
                new_kwags = meta_info.get('kwags',None)
                if new_kwags is not None:
                    kwags['last_check'] = new_kwags
            
            results += [[check_result,continue_on,meta_info,data]]
            
            if not continue_on:
                break
            
        results_full['extra_measure'] = results
        results_full['conditional_idx'] = self.cond[i]
            
        return results_full
        
        
            
            
            
