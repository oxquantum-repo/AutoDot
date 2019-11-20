# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 07:11:05 2019

@author: thele
"""

import scipy.signal as signal
import pickle
import numpy as np
from Last_score import final_score_cls

def peak_check(trace,minc,maxc,configs,**kwags):
    prominence = configs['prominance']
    
    #norm settings
    offset = minc
    maxval = maxc
    
    #peak detector settings
    height = 0.0178
   
    trace_norm=trace.copy()-offset
    trace_norm[trace_norm<0]=0
    trace_norm = (trace_norm)/((maxval-offset)) #normalize the current amplitude
    peaks, data = signal.find_peaks(trace_norm,prominence=prominence,height=height)
    return len(peaks)>=configs['minimum'], len(peaks)>=configs['minimum'], peaks


def reduce_then_clf_2dmap(data,minc,maxc,configs,**kwags):
    
    dim_reduction_fname = configs['dim_reduction']
    clf_fname = configs['clf']
    
    with open(dim_reduction_fname,'rb') as drf:
        dim_red = pickle.load(drf)
        
    with open(clf_fname,'rb') as cf:
        clf = pickle.load(cf)
        
    X = normilise(data,configs['norm'],minc,maxc)
    
    X_red = dim_red.transform(np.expand_dims(X,axis=0))
    
    Y = np.squeeze(clf.predict(X_red))
    return Y, Y, None


def last_score(data,minc,maxc,configs,**kwags):
    fsc = final_score_cls(minc,maxc,configs['noise'],configs['thresh'])
    
    score = getattr(fsc,configs.get('mode','score'))(data,diff=configs.get('diff',1))
    
    score_thresh = kwags.get('score_thresh')
    
    return score, score>score_thresh, None
    
def normilise(data,norm_type,minc,maxc):
    
    if norm_type is None:
        return data
    
    if isinstance(norm_type,list):
        min_val = norm_type[0]
        max_val = norm_type[1]
    elif norm_type == 'device_domain':
        min_val = minc
        max_val = maxc
    else:
        min_val = data.min()
        max_val = data.max()
        
    data_norm = np.copy(data)
    data_norm[data_norm>max_val] = max_val
    data_norm[data_norm<min_val] = min_val
    
    data_norm = (data_norm - min_val)/(max_val-min_val)
    return data_norm
        
    