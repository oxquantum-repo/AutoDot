import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import least_squares
from Pygor_new.measurement_functions import measurement_funcs as meas

def loss_dist_1line(u,x):
    u1 = u
    line1_loss_p1 = np.abs(x - u1)
    line1_loss_p2 = np.abs(x - (u1+(np.pi)))
    line1_loss= np.minimum(line1_loss_p1 , line1_loss_p2)       
    return line1_loss

def loss_dist_2line(u,x):
    u1,u2 = u
    line1_loss_p1 = np.abs(x - u1)
    line1_loss_p2 = np.abs(x - (u1+(np.pi)))
    line1_loss= np.minimum(line1_loss_p1 , line1_loss_p2)       
    line2_loss_p1 = np.abs(x - u2)
    line2_loss_p2 = np.abs(x - (u2+(np.pi)))
    line2_loss = np.minimum(line2_loss_p1 , line2_loss_p2)
    return np.minimum(line1_loss , line2_loss)



def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    
    
    
    

def find_peaks(trace,prominence):
    
    #norm settings
    offset = -2e-10
    maxval = 5e-10
    noise_level = 1E-11
    
    #peak detector settings
    height = 0.0178
   
    trace_norm=trace.copy()-offset
    trace_norm[trace_norm<0]=0
    trace_norm = (trace_norm)/((maxval-offset)) #normalize the current amplitude
    #print(trace_norm)
    peaks, data = scipy.signal.find_peaks(trace_norm,prominence=prominence,height=height)
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
        data2d[i,:] = trace
        
    return data2d
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


def less_crap_score(scan,filt=1e-11,ctarget=-1.5e-10,dev=2e-10):
    
    der = np.array(np.gradient(scan))
    
    der_mag = np.linalg.norm(der,axis=0)
            
            
    der_uvecs = der/der_mag
            
    x, y, z = der_uvecs[0].ravel(), der_uvecs[1].ravel(), der_mag.ravel()
    x_filt, y_filt, z_filt = x[z>filt], y[z>filt], z[z>filt]
            
    
    angles_filt = np.sign(y_filt)*np.arccos(x_filt/1)
    
    if len(angles_filt) < 2:
        return 0,0,0,0
    
    
    sol1 = least_squares(loss_dist_1line,[-3],args=(angles_filt,),method='lm',jac='2-point',max_nfev=2000)
    sol2 = least_squares(loss_dist_2line,[-3,-1.4],args=(angles_filt,),method='lm',jac='2-point',max_nfev=2000)
    
    
    c1 = sol1.cost/angles_filt.size
    c2 = sol2.cost/angles_filt.size
    
    asym = np.abs(angles_filt[angles_filt<0].size - angles_filt[angles_filt>=0].size)/np.maximum(angles_filt[angles_filt<0].size,angles_filt[angles_filt>=0].size)
    
    
    
    final_score = gaussian(np.average(scan),ctarget,dev)*(c1-c2)*(1-asym)
    
    return final_score,gaussian(np.average(scan),ctarget,dev),(c1-c2),(1-asym)
    
    
    
    
    
    
    
    
    
    
    
    
    
def do_less_crap_score(pygor,point,resmin=20,resmax=100,size=100,gates=["c3","c4","c5","c6","c7","c8","c9","c10"],varz=['c5','c9'],filt=1e-11):
    pygor.setvals(gates,point)
    starts = np.array(pygor.getvals(varz))
    ends = starts + (np.array([size,size])/np.sqrt(2))
    
    
    
    #traces = meas.do1d_combo(pygor,varz,starts,ends,resmax).data
    traces = np.array(meas.do1d_combo(pygor,varz,starts,ends,resmax).data)[0]
    
    pks = find_peaks(traces,0.02)
    
    save_array = [starts,traces,pks]
    
    if len(pks)==0:
        return 0, save_array
    elif len(pks)==1:
        
        
        sidelen = size/np.sqrt(2)
        
        st = starts + np.array([sidelen/2,-sidelen/2])
        so1 = starts + np.array([-sidelen/2,sidelen/2])
        so2 = st + np.array([sidelen,sidelen])
        
        cu_map = dodiag2d(pygor,varz,st,so1,so2,resmin,resmin)
        
        save_array += [cu_map,st,so1,so2]
        
    else:
    
        diffs = np.diff(pks)
            
        av_diff = np.average(diffs)
            
        rng = np.minimum(av_diff*3.5,100)
        
        sidelen = rng/np.sqrt(2)
        
        st = starts + np.array([sidelen/2,-sidelen/2])
        so1 = starts + np.array([-sidelen/2,sidelen/2])
        so2 = st + np.array([sidelen,sidelen])
        
        
        cu_map = dodiag2d(pygor,varz,st,so1,so2,resmin,resmin)
        save_array += [cu_map,st,so1,so2]
        sidelen = size/np.sqrt(2)
        
        st = starts + np.array([sidelen/2,-sidelen/2])
        so1 = starts + np.array([-sidelen/2,sidelen/2])
        so2 = st + np.array([sidelen,sidelen])
        
        largecu_map = dodiag2d(pygor,varz,st,so1,so2,resmin,resmin)
        
        save_array += [largecu_map]
        
    crap_score,copen,costdiff,asym = less_crap_score(cu_map,filt)
    
    save_array += [copen,costdiff,asym]
    
    
    if crap_score > 0.05:
        sidelen = size/np.sqrt(2)
        
        st = starts + np.array([sidelen/2,-sidelen/2])
        so1 = starts + np.array([-sidelen/2,sidelen/2])
        so2 = st + np.array([sidelen,sidelen])
        
        fullcu_map = dodiag2d(pygor,varz,st,so1,so2,resmax,resmax)
        
        save_array += [fullcu_map]
        
        
    #=================SAVE
    
    return crap_score, save_array


def bodge_score(pygor,point,resmin=20,resmax=100,size=100,gates=["c3","c4","c5","c6","c7","c8","c9","c10"],varz=['c5','c9'],filt=1e-11):
    pygor.setvals(gates,point)
    starts = np.array(pygor.getvals(varz))
    ends = starts + (np.array([size,size])/np.sqrt(2))
    
    
    
    traces = np.array(meas.do1d_combo(pygor,varz,starts,ends,resmax).data)[0]
    
    pks = find_peaks(traces,0.02)
    
    save_array = [starts,traces,pks]
    
    if len(pks)==0:
        return 0, [None]
    elif len(pks)==1:
        
        
        sidelen = size/np.sqrt(2)
        
        st = starts + np.array([sidelen/2,-sidelen/2])
        so1 = starts + np.array([-sidelen/2,sidelen/2])
        so2 = st + np.array([sidelen,sidelen])
        
        cu_map = dodiag2d(pygor,varz,st,so1,so2,resmin,resmin)
        
        save_array += [cu_map,st,so1,so2]
        
    else:
    
        diffs = np.diff(pks)
            
        av_diff = np.average(diffs)
            
        rng = np.minimum(av_diff*3.5,100)
        
        sidelen = rng/np.sqrt(2)
        
        st = starts + np.array([sidelen/2,-sidelen/2])
        so1 = starts + np.array([-sidelen/2,sidelen/2])
        so2 = st + np.array([sidelen,sidelen])
        
        
        cu_map = dodiag2d(pygor,varz,st,so1,so2,resmin,resmin)
        save_array += [cu_map,st,so1,so2]
        sidelen = size/np.sqrt(2)
        
        st = starts + np.array([sidelen/2,-sidelen/2])
        so1 = starts + np.array([-sidelen/2,sidelen/2])
        so2 = st + np.array([sidelen,sidelen])
        
        largecu_map = dodiag2d(pygor,varz,st,so1,so2,resmin,resmin)
        
        save_array += [largecu_map]
        
        
        
        
    der = np.array(np.gradient(cu_map))
            
    der_mag = np.linalg.norm(der,axis=0)
            
    der_uvecs = der/der_mag
            
    x, y, z = der_uvecs[0].ravel(), der_uvecs[1].ravel(), der_mag.ravel()
    x_filt, y_filt, z_filt = x[z>filt], y[z>filt], z[z>filt]
    angles_filt = np.sign(y_filt)*np.arccos(x_filt/1)
    
    if len(angles_filt) < 2:
        return 0, [None]
    
    sol1 = least_squares(loss_dist_1line,[0],args=(angles_filt,),method='lm',jac='2-point',max_nfev=2000)
    sol2 = least_squares(loss_dist_2line,[0,-1.4],args=(angles_filt,),method='lm',jac='2-point',max_nfev=2000)
    
    c1 = sol1.cost/angles_filt.size
    c2 = sol2.cost/angles_filt.size

    bodge_score = c1 - c2 # np.maximum(0.05-np.abs((c1-c2)/(c1/c2)-0.05),0)
    
    if bodge_score > 0.015:
        sidelen = size/np.sqrt(2)
        
        st = starts + np.array([sidelen/2,-sidelen/2])
        so1 = starts + np.array([-sidelen/2,sidelen/2])
        so2 = st + np.array([sidelen,sidelen])
        
        fullcu_map = dodiag2d(pygor,varz,st,so1,so2,resmax,resmax)
        
        save_array += [fullcu_map]
    
    return bodge_score, save_array
