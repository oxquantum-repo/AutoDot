# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:47:35 2019

@author: Dominic
"""

from functools import partial
import matplotlib.pyplot as plt
from registration_core import notranslation_affine_registration,simple_affine_registration,deformable_registration
import numpy as np

def visualize(iteration, error, X, Y):
    
    min_l = np.minimum(X.shape[0],Y.shape[0])
    
    abs_error = np.linalg.norm(X[:min_l]-Y[:min_l],axis=-1)
    
    print(iteration,error,np.sum(abs_error)/min_l)
    #diffs = X[:min_l]-Y[:min_l]
    #print(np.std(diffs,axis=0))
    
    
surface_points = np.load("data\\vols_poff_after.npy")
surface_points_target = np.load("data\\vols_poff_prev.npy")

min_l = np.minimum(len(surface_points<-1990),len(surface_points_target))
usefull = np.logical_and(~np.any(surface_points<-1990,axis=1)[:min_l],~np.any(surface_points_target<-1990,axis=1)[:min_l])
surface_points = surface_points[:min_l][usefull]
surface_points_target = surface_points_target[:min_l][usefull]



callback = partial(visualize)
reg = notranslation_affine_registration(**{ 'X': np.copy(surface_points), 'Y': np.copy(surface_points_target)})

reg.register(callback)
 
device_change = reg.B-np.diag(np.ones(7))
    
m_cmap = np.abs(device_change).max()
    
plt.imshow(device_change,vmin=-m_cmap,vmax=m_cmap,cmap='bwr')
plt.colorbar()
plt.show()