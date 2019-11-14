# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:21:29 2019

@author: Dominic
"""

import numpy as np

samples = np.load("uniform_surf_fixed\\vols_poff.npy")
e_meas = np.load("uniform_surf_fixed\\extra_measure.npy")
params=[]
labs=[]
for i,samp in enumerate(e_meas):
    params += [samples[i]]
    if (samples[i]<-1970).any():
        labs += [0]
    elif (len(samp)==25):   
        labs += [1]
    else:
        labs += [1]
        
        

params = np.array(params)
params_uvec = params/np.linalg.norm(params,axis=1)[:,np.newaxis]
labs = np.array(labs)

from sklearn.ensemble import RandomForestClassifier

clf_s = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

split = 20

clf_s.fit(params_uvec[:split],labs[:split])

from sklearn.metrics import confusion_matrix,log_loss
import cma


print(confusion_matrix(clf_s.predict(params_uvec[split:200]),labs[split:200]))


class orig_classifier():
    def __init__(self,max_b=-2000):
        self.val = [0]*8
        #self.val = [-680,-100,-100,-100,-100,-100,-100,-100]
        self.max_b = max_b
        self.Ys = []
        self.Xs = []
        
    def predict(self,X):
        '''
        proj_X = (X/np.amin(X,axis=1)[:,np.newaxis])*self.max_b
        
        print(proj_X[0])
        
        orig_vecs = proj_X-self.val
        print(orig_vecs[0])
        
        classes = ~np.any(orig_vecs>0,axis=1)
        '''
        classes = np.all( X < self.val, axis = 1 )
        
        return classes
    
    def loss(self,X,Y,verbose=True):
        
        pred_Y = self.predict(X)
        
        if verbose:
            print(confusion_matrix(pred_Y,Y))
        return log_loss(Y,pred_Y)
        
        
    def fit(self,X,Y):
        
        opts = cma.CMAOptions()
        opts.set("bounds", [[-2000]*8, [0]*8])
        
        
        self.Xs = X
        self.Ys = Y
        
        
        return cma.fmin(self.objective_func, self.val, 20, opts)
    
    
    def objective_func(self,val):
        self.val = val
        return self.loss(self.Xs,self.Ys)
 
oc = orig_classifier()

print(oc.loss(params_uvec[:400],labs[:400]))       

