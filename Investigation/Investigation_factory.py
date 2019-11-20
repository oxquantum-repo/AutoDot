# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:28:23 2019

@author: thele
"""
import measurement_functions
import condition_functions




class Investigation_stage():
    def __init__(self,jump,measure,check,configs):
        self.jump = jump
        self.measure = measure
        self.check = check
        
        self.configure_investigation_sequence(configs)
        
        
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
            
    def do_extra_measure(self,params,**kwags):
        self.jump(params)
        plunger_jump = lambda params:self.jump(params,True)
        
        anchor_vals = self.check()
        
        results = []
        
        for i in range(len(self.aquisition_functions)):
            data = self.aquisition_functions[i](plunger_jump,self.measure,anchor_vals,self.function_configs[i],**kwags)
            
            check_result,continue_on,meta_info = self.cond_functions(data,self.function_configs[i],**kwags)
            
            if not continue_on:
                break
        
        
            
            
            
