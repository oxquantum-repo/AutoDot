import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.optimize import least_squares
from Pygor.measurement_functions import measurement_funcs as meas


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
    
    
    
    
    
    
def cotunnel_score2(image,mask,diff,scale):
    #rescale data between 0 and 1
    image = (image-image.min())/(image.max()+(image-image.min()))
    #calculate lap
    kern = np.array([[0,1,0],[1,-4,1],[0,1,0]])/diff
    conv = signal.convolve2d(image,kern,boundary='symm', mode='same')
    #take average 
    val1 = np.abs(np.average(conv[mask])/diff)
    #multiply by arbitrary scale factor and clip to one
    return np.minimum(val1*scale,1)
    
    
    
    
    
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
        data2d[i,:] = trace
        
    return data2d
    
    
    
    
    
    
    
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd),np.where(sd == 0, 0, sd)
    
    
    

    
def og_2av_score(scan,filt=None,thresh=-1.4781e-10,diff=1,verbose=False,scale=10):
    """
    score a current map based on gradient orientation and second order derivative
    
    takes
        scan: numpy array shape (n,n)
    
    kwarg's
        filt: value of gradient deemed unreliable if None then is estimated from average noise of scan
        thresh: current threshold for analysis 
        diff: paramater to alow for scale invariance of scans
        verbose: returns plots and score individual componants
        scale: sets trade off between gradient orentation (double dotness) and scan bluriness
    returns
        score
    """
    #get gradients of data
    der = np.array(np.gradient(scan,diff))
    
    #calculate gardient magnitudes and directions
    der_mag = np.linalg.norm(der,axis=0)        
    der_uvecs = der/der_mag
   
    z_cur = np.copy(scan).ravel()

    #estimate noise level and set derivative filter threshold
    if filt is None:
        filt = np.mean(signaltonoise(der_mag)[-1])

    #filter directions and magnitudes
    x, y, z = der_uvecs[0].ravel(), der_uvecs[1].ravel(), der_mag.ravel()
    
    #filter using threshold and filt
    x_filt, y_filt, z_filt = x[z_cur>thresh], y[z_cur>thresh], z[z_cur>thresh]
    x_filt, y_filt, z_filt = x_filt[z_filt>filt], y_filt[z_filt>filt], z_filt[z_filt>filt]
            
    #calculate angles
    angles_filt = np.sign(y_filt)*np.arccos(x_filt/1)
    
    if len(angles_filt) < 2:
        return 0
    
    #fit single line
    sol1 = least_squares(ress_1line,[-3],args=(angles_filt,),bounds=[-np.pi,0],method='dogbox',jac='2-point',max_nfev=2000)

    #fit two lines by grid search
    sol_grid = grid_search(ress_2line,angles_filt,[[-np.pi,0],[-np.pi,0]])
    
    #compute average of squared residuals for both cases
    resid1 = ress_1line(sol1.x,angles_filt)

    grid_c11 = np.average(np.power(resid1,2))
    grid_c21 = sol_grid[-1]
    
    final_grid2 = cotunnel_score2(scan,scan>thresh,diff,scale)*(grid_c11-grid_c21)
    
    if verbose:
        plt.plot(angles_filt,z_filt,'xk')
        plt.axvline(sol1.x,color='b')
        plt.axvline(sol1.x+(np.pi),color='b')
        plt.axvline(sol_grid[1][0],0,color='r', linestyle='--')
        plt.axvline(sol_grid[1][1],0,color='r', linestyle='--')
        
        plt.axvline(sol_grid[1][0]+(np.pi),0,color='r', linestyle='--')
        plt.axvline(sol_grid[1][1]+(np.pi),0,color='r', linestyle='--')
        
        plt.xlabel("$\\theta_g$ / rad")
        
        plt.xlim([-np.pi,np.pi])
        
        
        plt.ylabel("$|g|$")
        
        plt.xticks([-np.pi,0,np.pi])
        
        plt.locator_params(axis='y', nbins=2)
        
        plt.savefig("og_fig.svg")
        
        plt.show()
        return cotunnel_score2(scan,scan>thresh,diff,scale),(grid_c11-grid_c21)
    else:
        return final_grid2*1.5

    
    
    
def check_for_peaks(pygor,point,gates,varz,size,res,**kwargs):
    pygor.setvals(gates,point)
    starts = pygor.getvals(varz)
    ends = starts + (np.array([size,size])/np.sqrt(2))
    traces = np.array(meas.do1d_combo(pygor,varz,starts,ends,res).data)[0]
    pks = find_peaks(traces,0.02,**kwargs)
    return pks,starts,traces

    
    
    
    
import time
def do_score(pygor,point,pks=None,res=20,resmax=128,size=100,window_mult=3.5,gates=["c3","c4","c5","c6","c7","c8","c9","c10"],varz=['c5','c9'],save_array=None,diff=1):
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
    full_score = og_2av_score(cu_map,diff=diff)
    d = time.time()
    save_array += [cu_map,st,so1,so2,full_score]
    return full_score, save_array, pks
    
    
    
    
    
    
def decision_aqu(pygor,point,low_res=20,high_res=60,check_res=128,base_size=100,decision_function=None,**kwargs):

    thresh = 0 if decision_function is None else decision_function
    def basic_decision(score):
        score>thresh 
        return score>thresh 
        
    if decision_function is None or isinstance(decision_function,float):
        decision_function = basic_decision
    
    score, save_array, pks = do_score(pygor,point,res=low_res,resmax=check_res,size=base_size)
    diff = low_res/high_res
    if decision_function(score):
        score, save_array, pks = do_score(pygor,point,pks=pks,res=high_res,save_array=save_array,size=base_size,diff=diff)
    
    return score, save_array
    
"""
    
def do_score(pygor,point,resmin=20,res2=60,resmax=128,size=100,gates=["c3","c4","c5","c6","c7","c8","c9","c10"],varz=['c5','c9'],noise_grad=4.31e-12,score_thresh=0):
    pks = check_for_peaks(pygor,point,gates,varz,size,resmax)
    save_array = [starts,traces,pks]
    
    if len(pks)>1:
        diffs = np.diff(pks)
        av_diff = np.average(diffs) 
        rng = np.minimum(av_diff*3.5,100)
        
        sidelen = rng/np.sqrt(2)
    elif len(pks)==1:
        sidelen = size/np.sqrt(2)
    else:
        return 0, save_array
        
    st = starts + np.array([sidelen/2,-sidelen/2])
    so1 = starts + np.array([-sidelen/2,sidelen/2])
    so2 = st + np.array([sidelen,sidelen])

    cu_map = dodiag2d(pygor,varz,st,so1,so2,resmin,resmin)  
    save_array += [cu_map,st,so1,so2]
        
    full_score = og_2av_score(cu_map)
    save_array += [full_score]
    
    if full_score >= score_thresh:
        stepmax = sidelen/res2
    
        fullcu_map = dodiag2d(pygor,varz,st,so1,so2,res2,res2)
        full_score2 = og_2av_score(fullcu_map,filt=filt,diff=0.33)
        
        save_array += [full_score2,fullcu_map]


    return full_score2, save_array

"""