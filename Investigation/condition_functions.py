# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 07:11:05 2019

@author: thele
"""

import scipy.signal as signal
import pickle
import numpy as np
from .scoring.Last_score import final_score_cls
from skimage.feature import blob_log
import time

def mock_peak_check(anchor,minc,maxc,configs,**kwags):
    a = configs.get('a',None)
    b = configs.get('b',None)
    verb = configs.get('verbose',False)
    
    if a is None and b is None:
        prob = configs.get('prob',0.5)
        
        c_peak = np.random.uniform(0,1)<prob
        if verb: print(c_peak)
        return c_peak, c_peak, None
    
    lb, ub = np.minimum(a,b), np.maximum(a,b)
    
    c_peak = np.all(anchor<ub) and np.all(anchor>lb)
    if verb: print(c_peak)
    return c_peak, c_peak, None


def mock_score_func(anchor,minc,maxc,configs,**kwags):
    a = np.array(configs.get('target',[-500,-500]))
    
    score = 100/ np.linalg.norm(a-anchor)
    print(score)
    return score, False, None
        
    

def check_nothing(trace,minc,maxc,configs,**kwags):
    output = configs.get('output',False)
    return output, output, None

def peak_check(trace,minc,maxc,configs,**kwags):
    prominence = configs['prominance']
    
    #norm settings
    offset = minc
    maxval = maxc
    
    #peak detector settings
    height = configs.get('height',0.0178)
   
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
    fsc = final_score_cls(minc,maxc,configs['noise'],configs['segmentation_thresh'])
    
    score = getattr(fsc,configs.get('mode','score'))(data,diff=configs.get('diff',1))
    

    
    s_cond = False
    
    
    
    print("Score: %f"%score)
    
    return score, s_cond, None



def last_score_then_blob(data,minc,maxc,configs,**kwags):
    fsc = final_score_cls(minc,maxc,configs['noise'],configs['segmentation_thresh'])
    
    score = getattr(fsc,configs.get('mode','score'))(data,diff=configs.get('diff',1))
    
    score_thresh = configs.get('score_thresh',None)
    if score_thresh is None:
        score_thresh = kwags.get('score_thresh')
    
    
    print("Score: %f"%score)
    
    blobs  = blob_detect_rough(data,minc,maxc)
    
    return score, score>score_thresh, {"kwags":{"blobs":blobs,"size_last":configs['size'],"res_last":configs['res']}}




def clf_then_blob(data,minc,maxc,configs,**kwags):
    
    data = normilise(data,configs['norm'],minc,maxc)
    clf_fname = configs['clf']

    with open(clf_fname,'rb') as cf:
        clf = pickle.load(cf)
        
        
    Y = np.squeeze(clf.predict(np.expand_dims(data,axis=0)))
    
    if Y:
        pass
    return



def count_above_thresh(data,minc,maxc,configs,**kwags):
    split_thresh = configs.get('split_thresh',0.0001)
    
    count_required = configs.get('count_required',0.0001)
    
    data_above = data[data>split_thresh]
    
    count_ratio = data_above.size/data.size
    
    blobs  = blob_detect_rough(data,minc,maxc)
    
    return count_ratio<count_required,count_ratio<count_required,{"kwags":{"cr":count_ratio,"blobs":blobs,"size_last":configs['size'],"res_last":configs['res']}}





def blob_detect_rough(data,minc,maxc):
    blobs = blob_log(normilise(data,'device_domain',minc,maxc),min_sigma=2,threshold=0.0001)[:,:2]
    return np.array([blobs[:,1],blobs[:,0]])
        
    
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
        

def plot_image_and_blobs(data,blobs):
    blob = blobs[:,:2]
    blob = np.array([blob[:,1],blob[:,0]])
    
    plt.imshow(data)
    for i in range(blob.shape[-1]):
        plt.scatter(*blob[:,i])
    plt.show()
    
    