# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:40:17 2019

@author: Dominic
"""
import sys
sys.path.insert(0,'C:\\Users\\Dominic\\Desktop\\MLdm\\dm-oxnm')
import numpy as np
from scipy import signal
from scipy.optimize import least_squares


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


def ress_2line_pm(u,x,umid=-np.pi/2):
    u1, u2 = umid-u,umid+u
    
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
    
    
    
    
    
def grid_search(func,x,bounds,res=100,**kwargs):
    
    search_list = []
    u = []
    for bound in bounds:
        search_list += [np.linspace(bound[0],bound[1],res)]
        u += [bound[0]]
        
    search_length = res**len(bounds)
    counts = np.array([0]*len(bounds))
    results = np.zeros([search_length])
    best_result = np.inf
    best_result2 = None
    best_params = 0
    for i in range(search_length):
        
        while any(counts[1:] == res):
            iter_index = np.where(counts == res)[0]
            counts[iter_index] = 0 
            counts[iter_index-1] += 1
            
        params = []
        for j,count in enumerate(counts):
            params += [search_list[j][count]]
            
        
        resids = func(params,x,**kwargs)
        result = np.average(np.power(resids,2))
        result2 = np.average(np.abs(resids))
        results[i] = result
        if result<best_result:
            best_result = result
            best_result2 = result2
            best_params = params
        
        
        counts[-1] += 1
    return [results,best_params,best_result2]
    
    
    
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd),np.where(sd == 0, 0, sd)
    

def og_features(scan,filt=None,base_noise=None,thresh=-1.4781e-10,diff=1,verbose=False,scale=10):
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
        
    
    if base_noise is not None:
        filt = np.maximum(filt,base_noise)
        


    #filter directions and magnitudes
    x, y, z = der_uvecs[0].ravel(), der_uvecs[1].ravel(), der_mag.ravel()
    
    #filter using threshold and filt
    x_filt, y_filt, z_filt = x[z_cur>thresh], y[z_cur>thresh], z[z_cur>thresh]
    #x_filt, y_filt, z_filt = x, y, z

    
    #print(len(z_filt))
    x_filt, y_filt, z_filt = x_filt[z_filt>filt], y_filt[z_filt>filt], z_filt[z_filt>filt]

            
    #calculate angles
    angles_filt = np.sign(y_filt)*np.arccos(x_filt/1)

    
    #print(len(angles_filt))
    
    if len(angles_filt) < 2:
        return 0,0,0
    
    #fit single line
    sol1 = least_squares(ress_1line,[-np.pi/2],args=(angles_filt,),bounds=[-np.pi,0],method='dogbox',jac='2-point',max_nfev=2000)

    #fit two lines by grid search
    #sol_grid = grid_search(ress_2line,angles_filt,[[-np.pi,0],[-np.pi,0]])
    
    
    singleline = sol1.x[0]
    
    mx = np.minimum(np.abs(singleline-(-np.pi)),np.abs(singleline))
    
    sol_grid = grid_search(ress_2line_pm,angles_filt,[[0,mx]],umid = singleline)
    spread_lines = sol_grid[1]
    sol_grid[1] = [singleline+spread_lines,singleline-spread_lines]
   
    
    #compute average of squared residuals for both cases
    resid1 = ress_1line(sol1.x,angles_filt)

    grid_c11 = np.average(np.power(resid1,2))
    
    grid_c11 = np.average(np.abs(resid1))
    
    grid_c21 = sol_grid[-1]
    
    
    multip = cotunnel_score2(scan,scan>thresh,diff,scale)
    
    final_grid2 = multip*(grid_c11-grid_c21)
    
    
    """
    plt.scatter(angles_filt,z_filt,marker='x',c='k',s=15,linewidth=0.4)
    plt.axvline(sol1.x,color='b')
    plt.axvline(sol1.x+(np.pi),color='b')
    plt.axvline(sol_grid[1][0],0,color='r', linestyle='--')
    plt.axvline(sol_grid[1][1],0,color='r', linestyle='--')
        
    plt.axvline(sol_grid[1][0]+(np.pi),0,color='r', linestyle='--')
    plt.axvline(sol_grid[1][1]+(np.pi),0,color='r', linestyle='--')
        
    plt.xlabel("$\\theta_g$ / rad")
        
    plt.xlim([-np.pi,np.pi])
    plt.ylim([0,z.max()])
        
        
    plt.ylabel("$|g|$")
        
    plt.xticks([-np.pi,0,np.pi])
        
    plt.locator_params(axis='y', nbins=2)
        
    plt.savefig("og_fig.svg")
        
    plt.show()
    """
    return final_grid2,multip,(grid_c11-grid_c21)
