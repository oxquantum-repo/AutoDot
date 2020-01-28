import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.optimize import least_squares
from Pygor_new.measurement_functions import measurement_funcs as meas


def ress_1line(u,x):
    u1 = u
    line1_loss_p1 = x - u1
    line1_loss_p2 = x - (u1+(np.pi))
    
    line1_loss = np.where(np.abs(line1_loss_p1) < np.abs(line1_loss_p2),line1_loss_p1,line1_loss_p2)
    return line1_loss
    
def ress_2line(u,x):
    u1, u2 = u
    line1_loss_p1 = x - u1
    line1_loss_p2 = x - (u1+(np.pi))
    
    line1_loss = np.where(np.abs(line1_loss_p1) < np.abs(line1_loss_p2),line1_loss_p1,line1_loss_p2)

    line2_loss_p1 = x - u2
    line2_loss_p2 = x - (u2+(np.pi))
    
    line2_loss = np.where(np.abs(line2_loss_p1) < np.abs(line2_loss_p2),line2_loss_p1,line2_loss_p2)
    return np.where(np.abs(line1_loss) < np.abs(line2_loss),line1_loss,line2_loss)
    
    
    
    
    
    
def squash(x,scale=10):
    return np.tanh(x*scale)


def cotunnel_score(image,mask,diff=1):
    image = (image-image.min())/(image.max()+(image-image.min()))
    kern = np.array([[0,1,0],[1,-4,1],[0,1,0]])/diff
    conv = signal.convolve2d(image,kern,boundary='symm', mode='same')
    val1 = np.average(conv[mask])
    return np.abs(squash(val1))
    
    
    
    
    
def grid_search(func,x,bounds,res=100):
    
    search_list = []
    u = []
    for bound in bounds:
        search_list += [np.linspace(bound[0],bound[1],res)]
        u += [bound[0]]
        
    search_length = res**len(bounds)
    counts = np.array([0]*len(bounds))
    results = np.zeros([search_length])
    best_result = np.inf
    best_params = None
    for i in range(search_length):
        
        while any(counts[1:] == res):
            iter_index = np.where(counts == res)[0]
            counts[iter_index] = 0 
            counts[iter_index-1] += 1
            
        params = []
        for j,count in enumerate(counts):
            params += [search_list[j][count]]
            
            
        result = np.average(np.power(func(params,x),2))
        results[i] = result
        if result<best_result:
            best_result = result
            best_params = params
        
        
        counts[-1] += 1
    return results,best_params,best_result
    
    

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
        data2d[i,:] = trace
        
    return data2d
    
    
    
    
    
    
    
    
    

def og_1var_score(scan,filt=2.2e-11,thresh=-1.5e-10,diff=1):
    
    der = np.array(np.gradient(scan,diff))
    
    der_mag = np.linalg.norm(der,axis=0)
            
    der_uvecs = der/der_mag
            
    x, y, z = der_uvecs[0].ravel(), der_uvecs[1].ravel(), der_mag.ravel()
    x_filt, y_filt, z_filt = x[z>filt], y[z>filt], z[z>filt]
            
    angles_filt = np.sign(y_filt)*np.arccos(x_filt/1)
    
    if len(angles_filt) < 2:
        return 0
    
    sol1 = least_squares(ress_1line,[-3],args=(angles_filt,),bounds=[-np.pi,0],method='dogbox',jac='2-point',max_nfev=2000)
    
    c1 = sol1.cost/angles_filt.size

    resid1 = ress_1line(sol1.x,angles_filt)
    
    singledot = np.std(resid1)/(2*np.pi)
    
    sdot = np.average(resid1)/(2*np.pi)
    
    asym = np.abs(angles_filt[angles_filt<0].size - angles_filt[angles_filt>=0].size)/np.maximum(angles_filt[angles_filt<0].size,angles_filt[angles_filt>=0].size)
    
    final_score1a = cotunnel_score(scan,scan>thresh,diff)*singledot*(1-asym)
    
    return final_score1a*4
    
    
def og_2av_score(scan,filt=2.2e-11,thresh=-1.5e-10,diff=1):
        der = np.array(np.gradient(scan,diff))
    
    der_mag = np.linalg.norm(der,axis=0)
            
    der_uvecs = der/der_mag
            
    x, y, z = der_uvecs[0].ravel(), der_uvecs[1].ravel(), der_mag.ravel()
    x_filt, y_filt, z_filt = x[z>filt], y[z>filt], z[z>filt]
            
    angles_filt = np.sign(y_filt)*np.arccos(x_filt/1)
    
    if len(angles_filt) < 2:
        return 0
    
    
    sol1 = least_squares(ress_1line,[-3],args=(angles_filt,),bounds=[-np.pi,0],method='dogbox',jac='2-point',max_nfev=2000)
    
    sol_grid = grid_search(ress_2line,angles_filt,[[-np.pi,0],[-np.pi,0]])
    
    c1 = sol1.cost/angles_filt.size
    
    resid1 = ress_1line(sol1.x,angles_filt)

    grid_c11 = np.average(np.power(resid1,2))
    grid_c21 = sol_grid[-1]

    
    asym = np.abs(angles_filt[angles_filt<0].size - angles_filt[angles_filt>=0].size)/np.maximum(angles_filt[angles_filt<0].size,angles_filt[angles_filt>=0].size)
    
    final_grid2 = cotunnel_score(scan,scan>thresh)*(grid_c11-grid_c21)*(1-asym)
    return final_grid2*1.5

    
    
    
    
    



    
    
    
    
    
    
    
    
    
def do_score(pygor,point,resmin=20,res2=60,resmax=128,size=100,gates=["c3","c4","c5","c6","c7","c8","c9","c10"],varz=['c5','c9'],filt=1e-11):
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
        
    ok_score1 = og_1var_score(cu_map)
    ok_score2 = og_2av_score(cu_map)
    
    save_array += [[ok_score1,ok_score2]]
    
    ok_score = np.maximum(ok_score1,ok_score2)
    
    if ok_score > 0.12:
        fullcu_map = dodiag2d(pygor,varz,st,so1,so2,res2,res2)
        ok_score_better1 = og_1var_score(fullcu_map,diff=0.33)
        ok_score_better2 = og_2av_score(fullcu_map,diff=0.33)
        
        save_array += [[ok_score_better1,ok_score_better2],fullcu_map]
        ok_score = np.maximum(ok_score_better1,ok_score_better2)
    
    if ok_score > 0.12:
        min1, max1 = starts[0]-100,starts[0]+100
        min2, max2 = starts[1]-100,starts[1]+100
        
        full_data = pygor.do2d(varz[0],min1,max1,resmax,varz[1],min2,max2,resmax).data[0]
        save_array += [full_data]
        
    #=================SAVE
 
    return ok_score, save_array
