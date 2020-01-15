import numpy as np
import scipy.signal
from Pygor_new.measurement_functions import measurement_funcs as meas

#========================================================================================
#Base Functions -------------------------------------------------------------------------
#========================================================================================

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
    peaks, data = scipy.signal.find_peaks(trace_norm,prominence=prominence,height=height)
    return peaks

def dot_periodicity_score(trace1,trace2,x_vals,prominence,weight):

    peaks1_index = find_peaks(trace1,prominence)
    peaks2_index = find_peaks(trace2,prominence)    
    peaks1_co = x_vals[peaks1_index] #relative coordinates of the peaks in trace 1 
    peaks2_co = x_vals[peaks2_index] #relative coordinates of the peaks in trace 2    
 
    if len(peaks1_co)<3 or len(peaks2_co)<3: 
        extra_score=1
        return extra_score
    else:
        peak_avg1=np.convolve(peaks1_co,[0.5, 0.5],'valid')
        peak_avg2=np.convolve(peaks2_co,[0.5, 0.5],'valid') 
    
        dist_avg1=(np.mean(np.diff(peak_avg1)))
        dist_avg2=(np.mean(np.diff(peak_avg2)))
    
        extra_score=2-(abs(dist_avg1-dist_avg2)/max(dist_avg1,dist_avg2))
        return extra_score*weight


def dot_space_score_avg(trace1,trace2,x_vals,prominence):
    
    peaks1_index = find_peaks(trace1,prominence)
    peaks2_index = find_peaks(trace2,prominence)    
    peaks1_x = x_vals[peaks1_index] #relative coordinates of the peaks in trace 1 
    peaks2_x = x_vals[peaks2_index] #relative coordinates of the peaks in trace 2 
    
    
    diff1 = np.diff(peaks1_x)
    diff2 = np.diff(peaks2_x)
    
    if len(diff1)==0 or len(diff2)==0:
        dist=np.nan
        return dist

    peak_avg1=np.convolve(peaks1_x,[0.5, 0.5],'valid')
    peak_avg2=np.convolve(peaks2_x,[0.5, 0.5],'valid')
    
    
    
    
    #replaced peak rejection as it was buggy with following which perfoms the same job
    av_dif = np.abs(peak_avg1[...,np.newaxis] - peak_avg2[np.newaxis,...])
    while av_dif.shape[0]!=av_dif.shape[1]:
        
        
        
        
        ind,vl = min(enumerate(av_dif.shape), key=lambda x: x[1])
        mx,vl = max(enumerate(av_dif.shape), key=lambda x: x[1])
        closest_peaks = np.amin(av_dif,axis=ind)
        furthest_peak = closest_peaks.argmax()
        av_dif=np.delete(av_dif,furthest_peak,axis=mx)
        if(mx==0):
            diff1=np.delete(diff1,[furthest_peak])
            peak_avg1=np.delete(peak_avg1,[furthest_peak])
        else:
            diff2=np.delete(diff2,[furthest_peak])
            peak_avg2=np.delete(peak_avg2,[furthest_peak])

            
    dist = np.linalg.norm(diff1-diff2)/np.sqrt(diff1.size)
                
    return dist


#========================================================================================
#Pairwise Scoring -----------------------------------------------------------------------
#========================================================================================
def dot_score_pair(trace1, trace2,x_vals, prominence=0.02,weight=1.5,smooth=True,**kwargs): 
    if smooth:
        (trace1, trace2) = smooth_trace([trace1, trace2],kwargs.get('pad',(1,1)),kwargs.get('conv',3))
    
    
    use_periodicity = kwargs.get('periodicity',False)
      
    
    #score traces based on peak spacing space embedding with average based peak rejection
    spacing_score=dot_space_score_avg(trace1,trace2,x_vals,prominence)

    #scores traces based on their periodicity
    periodicity_score=dot_periodicity_score(trace1,trace2,x_vals,prominence,weight)
    
    print(spacing_score,periodicity_score)
    
    if use_periodicity:
        total_score = spacing_score*periodicity_score
    else:
        total_score = spacing_score
    
    if np.isnan(total_score):
        total_score = 0

    return total_score

def smooth_trace(traces,pad=(1,1),conv=3):
    s_traces = []
    for trace in traces:
        s_traces += [np.convolve(np.lib.pad(trace,pad,'edge'),np.ones((conv,))/conv,mode='valid')]
    return s_traces

#========================================================================================
#Multi Trace Scoring --------------------------------------------------------------------
#========================================================================================

def multitrace_dot_score(traces,x_vals,prominence=0.02,weight=1.5,**kwargs):
    i_iter = np.where(np.triu(np.ones([len(traces),len(traces)],dtype=np.bool),1))
    
    full_scores = np.zeros([len(traces),len(traces)])
    
    scores = []
    for i in range(i_iter[0].size):
        scores += [dot_score_pair(traces[i_iter[0][i]],traces[i_iter[1][i]],x_vals,prominence,weight,**kwargs)]
        full_scores[[i_iter[0][i]],[i_iter[1][i]]] = scores[-1]
        
    if kwargs.get('full',False) == True:
        return full_scores + full_scores.T
        
    if 'average' in kwargs:
        if kwargs['average'] == False:
            return scores
        else:
            return np.average(scores,weights=kwargs['average'])
    return np.average(scores)

#========================================================================================
#Sampling Functions ---------------------------------------------------------------------
#========================================================================================

def compute_multi_combo(point,l=100,s=8,n=8):
    l_step = (l)/np.sqrt(2)
    s_step = (s)/np.sqrt(2)
    
    point_c = point - (np.array([l_step,l_step])/2)#============================================CHANGE - TO + FOR TRACES MORE NEGATIVE THEN SAMPLE POINT
    
    anc_point = point_c+(np.array([s_step,-s_step])*(((n-2)/2)+0.5))+(np.array([l_step,l_step])/2)
    trace_start = [anc_point[...]]
    trace_end = [anc_point[...]-(np.array([l_step,l_step]))]
    
    for i in range(n-1):
        anc_point = anc_point+np.array([-s_step,s_step])
        trace_start += [anc_point[...]]
        trace_end += [anc_point[...]-(np.array([l_step,l_step]))]
    return trace_start,trace_end


def perform_multi_combo(pygor,varz,starts,ends,res=100):
    assert len(starts) == len(ends)
    traces = []
    for i in range(len(starts)):
        pygor.setval(varz,starts[i])
        traces += meas.do1d_combo(pygor,varz,starts[i],ends[i],res).data
    return traces



#========================================================================================
#MAIN DRIVER ////////////////////////////////////////////////////////////////////////////
#^^^^^^^^^^^-----------------------------------------------------------------------------
#========================================================================================


def dot_score(pygor,point,gates=["c3","c4","c5","c6","c7","c8","c9","c10"],v_gates=["c5","c9"],l=100,s=8,res=100,n=8):
    pygor.setvals(gates,point)
    anc = pygor.getvals(['c5','c9'])
    queue_s,queue_e = compute_multi_combo(anc,l=l,s=s,n=n)
    begining_s = [queue_s[0],queue_s[-1]]
    begining_e = [queue_e[0],queue_e[-1]]
    traces = perform_multi_combo(pygor,['c5','c9'],begining_s,begining_e,res=res)
    x_vals = np.linspace(0,100,100)
    score_inital = multitrace_dot_score(traces,x_vals)
    if score_inital == 0 :
        return 0
    else:
        begining_s = [queue_s[2],queue_s[-3]]
        begining_e = [queue_e[2],queue_e[-3]]
        traces_more = perform_multi_combo(pygor,['c5','c9'],begining_s,begining_e,res=res)
        traces += [traces_more]
        scores_full = multitrace_dot_score(traces,x_vals)
        
        print(scores_full)
        return scores_full
    
#========================================================================================
#MAIN DRIVER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^/////////////////////////////////////////////////////////////////////////////
#========================================================================================
