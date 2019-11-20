# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:17:14 2019

@author: thele
"""

import numpy as np

def do1dcombo(jump,measure,anchor_vals,configs,**kwags):
    size = configs.get('size',128)
    direction = np.array(configs.get('direction',[-1]*len(anchor_vals)))
    res = configs.get('res',128)
    
    delta_volt = np.linspace(0,size,res)
    
    anchor_vals = np.array(anchor_vals)
    
    trace = combo1d(jump,measure,anchor_vals,delta_volt,direction)
    
    return trace


def do2d(jump,measure,anchor_vals,configs,**kwags):
    bound = kwags.get('bound',configs['size'])
    res = configs.get('res',20)
    direction = np.array(configs.get('direction',[-1]*len(anchor_vals)))
    
    iter_vals = [None]*2
    for i in range(2):
        iter_vals[i] = np.linspace(0,bound[i],res)
        
    iter_deltas = np.array(np.meshgrid(*iter_vals))
        
    data = np.zeros([res,res])
    
    for i in range(res):
        for j in range(res):
            params_c = anchor_vals + direction*iter_deltas[:,i,j]
            jump(params_c)
            data[i,j] = measure()
            
    return data
            
            
    
        
        
def combo1d(jump,measure,anc,deltas,dirs):
    
    trace = np.zeros(len(deltas))
    
    for i in range(len(deltas)):
        params_c = anc + dirs*deltas[i]
        
        jump(params_c)
        trace[i] = measure()
        
    return trace