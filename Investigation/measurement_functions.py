# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:17:14 2019

@author: thele
"""

import numpy as np

import matplotlib.pyplot as plt
import time


def do_nothing(jump,measure,anchor_vals,configs,**kwags):
    return None


def mock_measurement(jump,measure,anchor_vals,configs,**kwags):
    pause = configs.get('pause',0)
    time.sleep(pause)
    return anchor_vals


def do1dcombo(jump,measure,anchor_vals,configs,**kwags):
    size = configs.get('size',128)
    direction = np.array(configs.get('direction',[1]*len(anchor_vals)))
    res = configs.get('res',128)
    
    delta_volt = np.linspace(0,size,res)
    
    anchor_vals = np.array(anchor_vals)
    
    trace = combo1d(jump,measure,anchor_vals,delta_volt,direction)
    
    if configs.get('plot',False):
        plt.plot(delta_volt,trace)
        plt.show()
    return trace


def do2d(jump,measure,anchor_vals,configs,**kwags):
    bound = kwags.get('bound',configs['size'])
    res = configs.get('res',20)
    direction = np.array(configs.get('direction',[1]*len(anchor_vals)))
    
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
                
            
            
    if configs.get('plot',False):      
        plt.imshow(data,cmap='bwr')
        plt.show()
    
            
    return data


def measure_random_blob(jump,measure,anchor_vals,configs,**kwags):
    
    
    blobs = kwags['last_check'].get('blobs')
    
    old_size = kwags['last_check'].get('size_last')
    
    old_res = kwags['last_check'].get('res_last')
    
    size = configs['size']
    res = configs['res']
    
    x_old = np.linspace(0,old_size[0],old_res)
    y_old = np.linspace(0,old_size[1],old_res)
    
    
    try:
        random_blob_idx = np.random.choice(blobs.shape[-1])
        random_blob = blobs[:,random_blob_idx]
    except ValueError:
        return None
    
    

    random_blob = blobs[:,-1]
    
    random_blob = np.array([x_old[int(random_blob[0])],y_old[int(random_blob[1])]])
    
    anc_new = (anchor_vals - random_blob)+(size/2)
    
    print(anc_new)
    
    configs_new = {'size':[size,size],'res':res,'direction':[-1,-1]}
    
    
    data = do2d(jump,measure,anc_new,configs_new,**kwags)
    
    if configs.get('plot',False):      
        plt.imshow(data,cmap='bwr')
        plt.show()
    return data
            



def do2_do2d(jump,measure,anchor_vals,configs,**kwags):
    pygor = kwags.get('pygor')
    if pygor is None:
        raise ValueError("Pygor instance was not passed to investigation stage")
        
    var_par = configs['var params']
    
    assert len(var_par)==3
    
    data = []
    print(pygor.setvals(var_par[0]['keys'],var_par[0]['params']))
    
    data += [do2d(jump,measure,anchor_vals,configs,**kwags)]
    
    print(pygor.setvals(var_par[1]['keys'],var_par[1]['params']))
    time.sleep(60)
    
    data += [do2d(jump,measure,anchor_vals,configs,**kwags)]
    
    print(pygor.setvals(var_par[2]['keys'],var_par[2]['params']))
    time.sleep(60)
    
    return np.array(data)
        
        
def combo1d(jump,measure,anc,deltas,dirs):
    
    trace = np.zeros(len(deltas))
    
    for i in range(len(deltas)):
        params_c = anc + dirs*deltas[i]
        
        jump(params_c)
        trace[i] = measure()
        
    return trace