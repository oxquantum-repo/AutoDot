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
    
    
#surface_points = np.load("data//moving_B2t2_cd2.npy")
surface_points_target = np.load("data//save_Florian_redo//vols_poff_prev.npy")
surface_points = np.load("data//save_Florian_redo//vols_poff_after.npy")#[:,[0,1,2,3,4,6,7]]
#surface_points_target = np.load("data//target_B2t2_cd1.npy")

    
    
#surface_points = np.load("data//moving_B1t2_b1.npy")
#surface_points_target = np.load("data//target_B1t2_b2.npy")


#surface_points = np.load("data//moving_B1t2_b1.npy")
#surface_points_target = np.load("data//target_B2t2_cd1.npy")

#"""
min_l = np.minimum(len(surface_points<-1990),len(surface_points_target))
usefull = np.logical_and(~np.any(surface_points<-1990,axis=1)[:min_l],~np.any(surface_points_target<-1990,axis=1)[:min_l])
surface_points = surface_points[:min_l][usefull]
surface_points_target = surface_points_target[:min_l][usefull]
#"""
print(surface_points.shape,surface_points_target.shape)

callback = partial(visualize)
reg = notranslation_affine_registration(**{ 'X': np.copy(surface_points_target), 'Y': np.copy(surface_points)})

reg.register(callback)
 
device_change = reg.B-np.diag(np.ones(7))
    
m_cmap = np.abs(device_change).max()
m_cmap = 0.3


plt.imshow(device_change.T,vmin=-m_cmap,vmax=m_cmap,cmap='PuOr')

xlabels = ['$V_1$', '$V_2$', '$V_3$', '$V_4$', '$V_5$', '$V_7$', '$V_8$']
ylabels = ['$VT_1$', '$VT_2$', '$VT_3$', '$VT_4$', '$VT_5$', '$VT_7$', '$VT_8$']
xlabs = np.linspace(0,6,7)
plt.xticks(xlabs,xlabels)
plt.yticks(xlabs,ylabels)
plt.colorbar()
plt.savefig("B1t2_transformation.svg")
plt.show()


np.save("vols_poff_transformed_prevb1b2.npy",reg.TY)
np.save("vols_poff_prevb1b2.npy",reg.Y)
np.save("vols_poff_afterb1b2.npy",reg.X)

print(np.linalg.det(reg.B),m_cmap)

#np.save("data/registrated_pointset_B1toB2.npy",reg.TY)