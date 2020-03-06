# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:36:55 2019

@author: Dominic
"""



import numpy as np
from scipy import signal
from scipy.optimize import least_squares
from .score_helper import ress_1line,ress_2line_pm,signaltonoise,grid_search


class final_score_cls():
    def __init__(self,minc,maxc,base_noise,thresh,scale1=150,scale2=7):
        self.minc, self.maxc = minc, maxc
        self.base_noise = base_noise
        self.thresh = thresh
        self.scale1 = scale1
        self.scale2 = scale2
        
        self.fit1 = None
        self.fit2 = None
        self.fits = None
        
        self.og=None
        
    def sharpness_score(self,img,diff=1,old=False):
        
        self.fit1 = None
        self.fit2 = None
        
        mask = img>self.thresh
        
        img = (img-self.minc)/(self.maxc+(img-self.minc))
        
         
        
        kern = np.array([[0,1,0],[1,-4,1],[0,1,0]])/diff
        conv = signal.convolve2d(img,kern,boundary='symm', mode='same')
        val1 = np.abs(np.average(conv[mask]))
        val2 = np.std(conv[mask])
        
        if old:
            return np.minimum(val1*self.scale1/10,1)
        else:
            return np.minimum(val1* val2*self.scale1,1)
    
    def og_val(self,fit1,fit2,angles):
        
        
        kwargs = fit2['kwargs']
        
        params1 = fit1['params']
        params2 = fit2['params']
        
        
        resids1 = ress_1line(params1,angles)
        resids2 = ress_2line_pm(params2,angles,**kwargs)
        result1 = np.average(np.abs(resids1))
        result2 = np.average(np.abs(resids2))
        
        self.og = np.array([result1-result2,result1,result2,result2/result1])
        
        return result1-result2
    
    def og_filter(self,img,diff=1,filt=None):
        

        
        #get gradients of data
        der = np.array(np.gradient(img,diff))
    
        #calculate gardient magnitudes and directions
        der_mag = np.linalg.norm(der,axis=0)        
        der_uvecs = der/der_mag
   
        z_cur = np.copy(img).ravel()

        #estimate noise level and set derivative filter threshold
        if filt is None:
            filt = np.mean(signaltonoise(der_mag)[-1])
            filt = np.maximum(filt,self.base_noise)

        #filter directions and magnitudes
        x, y, z = der_uvecs[0].ravel(), der_uvecs[1].ravel(), der_mag.ravel()
    
        #filter using threshold and filt
        x_filt, y_filt, z_filt = x[z_cur>self.thresh], y[z_cur>self.thresh], z[z_cur>self.thresh]
        #if flag is None:
        #    print("a%i"%len(z_filt))
        x_filt, y_filt, z_filt = x_filt[z_filt>filt], y_filt[z_filt>filt], z_filt[z_filt>filt]
        ##if flag is None:
        #    print("b%i"%len(z_filt))

        #calculate angles
        angles_filt = np.sign(y_filt)*np.arccos(x_filt/1)
        return angles_filt,z_filt,filt
    
    def og_fit(self,angles):
        sol1 = least_squares(ress_1line,[-np.pi/2],args=(angles,),bounds=[-np.pi,0],method='dogbox',jac='2-point',max_nfev=2000)
        singleline = sol1.x[0]
        mx = np.minimum(np.abs(singleline-(-np.pi)),np.abs(singleline))
        sol_grid = grid_search(ress_2line_pm,angles,[[0,mx]],umid = singleline)
        
        fit1 = {'params':sol1.x,'raw':sol1}
        fit2 = {'params':sol_grid[1],'raw':sol_grid,'kwargs':{'umid':singleline}}
        
        return fit1,fit2
    
    
    
    def og_score(self,img,diff=1,min_angles=2,filt=None):
        
        angles,mag,filt = self.og_filter(img,diff=diff,filt=filt)
        
        
        if len(angles) < min_angles:
            self.fit1 = None
            self.fit2 = None

            
            return 0
        
        fit1,fit2 = self.og_fit(angles)
        
        og = self.og_val(fit1,fit2,angles)
        
        self.fit1 = fit1
        self.fit2 = fit2
        
        return og
    def dir_score(self,img,blocks=4,diff=1):
        
        a,b,filt = self.og_filter(img,diff=diff)
        ogs = self.tiled_score(img,filt=filt,diff=diff,blocks=blocks)
        fits = self.fits
    
        single_dirs = []
        flairs = []
    
        for fit in fits:
            sol1 = fit[0]
            sol2 = fit[1]
            
            if sol1['raw'] is not None:
                single_dirs += sol1['raw'].x.tolist()
                flairs += sol2['raw'][1]
            else:
                single_dirs += [np.nan]
                flairs += [np.nan]
        
        single_dirs = np.array(single_dirs,dtype = np.float).reshape([blocks,blocks]).T
        
        dir_score = np.minimum(np.nanmean(np.nanstd(single_dirs,axis=1))*7,1)
        if np.isnan(dir_score):
            dir_score = 0
            
        return dir_score

   
        
    def tiled_score(self,img,blocks=4,score_func='og',**kwargs):
        
        self.fits = None
        
        assert img.shape[0] == img.shape[1]
    
        img_t = np.array(np.split(img,blocks,axis=0))
        img_t = np.array(np.split(img_t,blocks,axis=-1))
        
        switch = {'og':self.og_score,
                  'sharp':self.sharpness_score}
        
        score_func = switch.get(score_func,score_func)
        
        scores = []
        
        fits = []
        
        for i in range(blocks):
            for j in range(blocks):
                scores += [score_func(img_t[i,j,:,:],**kwargs)]
                if self.fit1 is not None:
                    fits += [[self.fit1,self.fit2]]
                    self.fits = fits
                else:
                    fits += [[{'raw':None},{'raw':None}]]
                    self.fits = fits
                
        scores = np.array(scores)
        scores[np.isnan(scores)]=0
                
        return scores
    
    
    
    def score(self,img,blocks=4,diff=1,verb=False):
        og = self.og_score(img,diff=diff)
        
        if blocks is None:
            shrp = self.sharpness_score(img,diff=diff)
        else:
            shrps = self.tiled_score(img,score_func='sharp',diff=diff,blocks=blocks)
            shrp = np.average(shrps)
        
        dirs = self.dir_score(img,diff=diff)
        
        
        if verb:
            return np.maximum(og*shrp*dirs,0),og,shrp,dirs
        return np.maximum(og*shrp*dirs,0)
        