import numpy as np
import scipy.signal
from Pygor_new.measurement_functions import measurement_funcs as meas

def find_peaks(trace,prominence):
    
    #norm settings
    offset = -1.5e-10
    maxval = 5e-10
    noise_level = 1E-11
    
    #peak detector settings
    height = 0.0178
   
    trace_norm=trace.copy()-offset
    trace_norm[trace_norm<0]=0
    trace_norm = (trace_norm)/((maxval-offset)) #normalize the current amplitude
    peaks, data = scipy.signal.find_peaks(trace_norm,prominence=prominence,height=height)
    return peaks




def dot_score_avg(coor, trace1,trace2,prominence):
    
    peaks1_index = find_peaks(trace1,prominence)
    peaks2_index = find_peaks(trace2,prominence)    
    peaks1_co = coor[peaks1_index] #relative coordinates of the peaks in trace 1 
    peaks2_co = coor[peaks2_index] #relative coordinates of the peaks in trace 2 
    
    diff1 = np.diff(peaks1_co)
    diff2 = np.diff(peaks2_co)
    
    if len(diff1)==0 or len(diff2)==0:
        dist=np.nan
        return dist

    peak_avg1=np.convolve(peaks1_co,[0.5, 0.5],'valid')
    peak_avg2=np.convolve(peaks2_co,[0.5, 0.5],'valid')
    
    if len(diff1)>len(diff2):
        while len(diff1)>len(diff2):
                     
            avg_score=[]
            for num in peak_avg1:
                avg_score+=[min(abs(peak_avg2-num))] 
            max_avg_index=avg_score.index(max(avg_score))
            diff1=np.delete(diff1,[max_avg_index])
            peak_avg1=np.delete(peak_avg1,[max_avg_index])
       
    if len(diff2)>len(diff1):
        while len(diff2)>len(diff1):
            
            peak_avg1=np.convolve(peaks1_co,[0.5, 0.5],'valid')
            peak_avg2=np.convolve(peaks1_co,[0.5, 0.5],'valid')

            avg_score=[]
            for num in peak_avg2:
                avg_score+=[min(abs(peak_avg1-num))]
            max_avg_index=avg_score.index(max(avg_score))
            diff2=np.delete(diff2,[max_avg_index])
            peak_avg2=np.delete(peak_avg2,[max_avg_index])
            
    dist = np.linalg.norm(diff1-diff2)/np.sqrt(diff1.size)
                
    return dist   




def dot_score_weight(coor, trace1,trace2,prominence,weight):

    peaks1_index = find_peaks(trace1,prominence)
    peaks2_index = find_peaks(trace2,prominence)    
    peaks1_co = coor[peaks1_index] #relative coordinates of the peaks in trace 1 
    peaks2_co = coor[peaks2_index] #relative coordinates of the peaks in trace 2    
 
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




def execute_score(coor, traces_dir1, traces_dir2, prominence=0.02,weight=1.5): 
    scores1=[]
    scores2=[]
    #Scores based on distance difference
    for j in range(0,len(traces_dir1),2): #calculate the scores for the the traces in direction 1 and direction 2
        score1=dot_score_avg(coor,traces_dir1[j],traces_dir1[j+1],prominence)
        score2=dot_score_avg(coor,traces_dir2[j],traces_dir2[j+1],prominence)
        print(score1,score2)

        #Scores based on extra periodicity weight factor
        score_extra1=dot_score_weight(coor,traces_dir1[j],traces_dir1[j+1],prominence,weight)
        score_extra2=dot_score_weight(coor,traces_dir2[j],traces_dir2[j+1],prominence,weight)
        print(score_extra1,score_extra2)

        scores1+=[score1*score_extra1]
        scores2+=[score2*score_extra2]

    if np.isnan(scores1):
        scores1 = [0]
    if np.isnan(scores2):
        scores2 = [0]

    print(scores1,scores2)
    #nx is the number of x pixels
    #ny is the number of y pixels
    tot_score = scores1[0] + scores2[0]
    return tot_score

def smooth_trace(traces,pad=(1,1),conv=3):
    s_traces = []
    for trace in traces:
        s_traces += [np.convolve(np.lib.pad(trace,pad,'edge'),np.ones((conv,))/conv,mode='valid')]
    return s_traces



def dot_score_sample(pygor,params,gates=["c3","c4","c5","c6","c7","c8","c9","c10"],v_gates=["c5","c9"],l=100,s=20,res=100):
    
    def safe_shuffle(varz,vals):
        pygor.server.config_control("dac",{'set_settletime': 0.03,'set_shuttle': True})
        pygor.setvals(varz,vals)
        pygor.server.config_control("dac",{'set_settletime': 0.01,'set_shuttle': False})
    
    safe_shuffle(gates,params)
    v1_val,v2_val=pygor.getvals(["c5","c9"])
    l_step = (l/2)/np.sqrt(2)
    s_step = (s/2)/np.sqrt(2)
    tracepoints = np.zeros([4,2])
    endpoints = np.zeros([4,2])
    traces = np.zeros([tracepoints.shape[0],100])
    
    tracepoints[0,:] = [v1_val  +l_step  +s_step  ,v2_val  +l_step  -s_step]
    endpoints[0,:] = [v1_val  -l_step  +s_step,  v2_val  -l_step  -s_step]
    
    tracepoints[1,:] = [v1_val  -l_step  -s_step,  v2_val  -l_step  +s_step]
    endpoints[1,:] = [v1_val  +l_step  -s_step,  v2_val  +l_step  +s_step]
    
    tracepoints[2,:] = [v1_val  -l_step  +s_step,  v2_val  +l_step  +s_step]
    endpoints[2,:] = [v1_val  +l_step  +s_step,  v2_val  -l_step  +s_step]
    
    tracepoints[3,:] = [v1_val  +l_step  -s_step,  v2_val  -l_step  -s_step]
    endpoints[3,:] = [v1_val  -l_step  -s_step,  v2_val  +l_step  -s_step]

    for i in range(tracepoints.shape[0]):
        safe_shuffle(v_gates,tracepoints[i])
        traces[i,:] = meas.do1d_combo(pygor,v_gates,tracepoints[i],endpoints[i],res).data
        
    safe_shuffle(gates,params)
    x_vals = np.linspace(0,l,res)
    return traces,x_vals,tracepoints,endpoints
    
    
    
    
    
    
def dot_score(pygor,params,gates=["c3","c4","c5","c6","c7","c8","c9","c10"],v_gates=["c5","c9"],l=100,s=20,res=100):
    traces,coor,tracepoints,endpoints = dot_score_sample(pygor,params,gates,v_gates,l,s,res)
    s_traces = smooth_trace(traces)
    s1 = execute_score(coor,s_traces[0:2],s_traces[2:4])
    return s1
