import numpy as np
from scipy import signal
from Pygor_new.measurement_functions import measurement_funcs as meas
from Last_score import final_score_cls
import time

do_scale = False
def scaler(data):
    return (data*3.2) - 1.7E-10


def find_peaks(trace,prominence):
    
    #norm settings
    offset = -1.866e-10
    maxval = 4.606e-10
    
    #peak detector settings
    height = 0.0178
   
    trace_norm=trace.copy()-offset
    trace_norm[trace_norm<0]=0
    trace_norm = (trace_norm)/((maxval-offset)) #normalize the current amplitude
    peaks, data = signal.find_peaks(trace_norm,prominence=prominence,height=height)
    return peaks

  


def dodiag2d(pygor,varz,starts,stopsa1,stopsa2,res1,res2,chan=0):
    
    pygor.setvals(varz,starts)
    
    assert np.isclose(np.dot(np.array(starts)-np.array(stopsa1),np.array(starts)-np.array(stopsa2)),0)
    
    a1_iterate = []
    for i in range(len(starts)):
        a1_iterate += [np.linspace(starts[i],stopsa1[i],res1)]
    a1_iterate = np.array(a1_iterate)
    
    

    
    
    a2_vec = np.array(stopsa2)-np.array(starts)
    
    
    
    data2d = np.zeros([res1,res2])
    for i in range(res1):
        trace = meas.do1d_combo(pygor,varz,a1_iterate[:,i],a1_iterate[:,i]+a2_vec,res2).data[chan]
        if do_scale: trace = scaler(trace)
        data2d[i,:] = trace
        
    return data2d
    


def check_for_peaks(pygor,point,gates,varz,size,res,**kwargs):
    pygor.setvals(gates,point)
    starts = pygor.getvals(varz)
    ends = starts + (np.array([size,size])/np.sqrt(2))
    traces = np.array(meas.do1d_combo(pygor,varz,starts,ends,res).data)[0]
    pks = find_peaks(traces,0.02,**kwargs)
    return pks,starts,traces
    
    
    
def do_score(pygor,point,score_object,pks=None,res=20,resmax=128,size=100,window_mult=3.5,gates=["c3","c4","c5","c6","c7","c8","c9","c10"],varz=['c5','c9'],save_array=None,diff=1):
    if save_array is None:
        save_array = []
    a = time.time()
    if pks is None:
        pks,starts,traces = check_for_peaks(pygor,point,gates,varz,size,resmax)
    else:
        pygor.setvals(gates,point)
        starts = pygor.getvals(varz)
        traces = None
    b = time.time()
    save_array += [starts,traces,pks]
    
    if len(pks)>1:
        diffs = np.diff(pks)
        av_diff = np.average(diffs) 
        rng = np.minimum(av_diff*window_mult,size)
        
        sidelen = rng/np.sqrt(2)
    elif len(pks)==1:
        sidelen = size/np.sqrt(2)
    else:
        return 0, save_array, pks
        
    st = starts + np.array([sidelen/2,-sidelen/2])
    so1 = starts + np.array([-sidelen/2,sidelen/2])
    so2 = st + np.array([sidelen,sidelen])

    cu_map = dodiag2d(pygor,varz,st,so1,so2,res,res)
    c = time.time()
    full_score = score_object.score(cu_map,diff=diff)
    d = time.time()
    save_array += [cu_map,st,so1,so2,full_score,[a,b,c,d]]
    return full_score, save_array, pks
    
def decision_aqu(pygor,point, gates, varz, score_object=None,low_res=20,high_res=60,check_res=128,base_size=100,decision_function=None,**kwargs):

    if score_object is None:
        score_object = final_score_cls(-1.8e-10,4.4e-10,5e-11,-1.4781e-10,150)

    thresh = 0 if decision_function is None else decision_function
    def basic_decision(score):
        return score>=thresh 
        
    if decision_function is None or isinstance(decision_function,float):
        decision_function = basic_decision

    #diff = low_res/high_res
    #diff = low_res/high_res
    diff_low = 20 / low_res
    diff_high = 20 / high_res
    
    score, save_array, pks = do_score(pygor,point,score_object,res=low_res,resmax=check_res,size=base_size, diff=diff_low, gates=gates, varz=varz)

    if len(pks) == 0:
        return 0, save_array

    if decision_function(score):
        score, save_array, pks = do_score(pygor,point,score_object,pks=pks,res=high_res,save_array=save_array,size=base_size,diff=diff_high, gates=gates, varz=varz)
    
    return score, save_array

