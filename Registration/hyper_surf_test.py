# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:16:29 2019

@author: Dominic
"""

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from registration_core_old import affine_registration
from registration_core import notranslation_affine_registration as arnew
from registration_core import deformable_registration,simple_affine_registration
import numpy as np
import matplotlib.pyplot as plt


def visualize(iteration, error, X, Y, ax):
    
    min_l = np.minimum(X.shape[0],Y.shape[0])
    
    
    abs_error = np.linalg.norm(X[:min_l]-Y[:min_l],axis=-1)
    
    
    print(iteration,error,np.sum(abs_error)/min_l)
    
    diffs = X[:min_l]-Y[:min_l]
    
    
    print(np.std(diffs,axis=0))
    
    #plt.hist(())
    #plt.show()




def syst_noise(points,displacment):
    
    uvecs = points/np.linalg.norm(points,axis=1)[...,np.newaxis]
    
    return (-(uvecs+0.5)/0.5)*displacment



def syst_noise2(deltas,uvecdis,amount):
    return (deltas[:,np.newaxis])*uvecdis*amount



from random import gauss
def make_rand_vector(dims,n):
    vecs = []
    for i in range(n):
        vec = [gauss(0, 1) for i in range(dims)]
        mag = sum(x**2 for x in vec) ** .5
        u_vec = [x/mag for x in vec]
        vecs += [u_vec]
        
    return vecs


rand_uvec = make_rand_vector(8,1)


noise_search = np.linspace(0,1,5)

noise_grid = noise_search[...,np.newaxis]*rand_uvec



#Af = np.diag([0.7,1.3,0.8,0.95,1.1,0.5,0.6,1.05])

bound1 = 0.1
bound2 = 0.1
bound3 = 0.05
bound4 = 0.025
n = 8


main_diag = np.diag(np.random.uniform(1-bound1,1+bound1,n))

off_diag1 = np.diag(np.random.uniform(0,bound2,n-1),1)
off_diag1_T = np.diag(np.random.uniform(0,bound2,n-1),-1)

off_diag2 = np.diag(np.random.uniform(0,bound3,n-2),2)
off_diag2_T = np.diag(np.random.uniform(0,bound3,n-2),-2)

off_diag3 = np.diag(np.random.uniform(0,bound4,n-3),3)
off_diag3_T = np.diag(np.random.uniform(0,bound4,n-3),-3)

Af = main_diag + off_diag1 + off_diag1_T + off_diag2 + off_diag2_T + off_diag3 + off_diag3_T

print(Af)

erros = []


if True:

#for i in range(len(noise_grid)):
    surface_points = np.load("data\\save_Dominic_redo_Basel2_correct\\vols_poff_after.npy")#[:,[0,1,2,3,4,6,7]]
    print(surface_points.shape,surface_points[0])
    
    #surface_points2 = np.load("data\\vols_poffb1_cd1.npy")[:,[0,1,2,3,4,6,7]]
    
    #surface_points = surface_points[~np.any(surface_points<-1990,axis=1)]

    
    
    




    #sys_noi = syst_noise(surface_points,noise_grid[i],rand_uvec,noise_search)
    
    
    
    
    
    #surface_points_target = np.dot( surface_points ,Af)
    
    surface_points_target = np.load("data\\save_Dominic_redo_Basel2_correct\\vols_poff_prev.npy")#[:,[0,1,2,3,4,6]]
    
    
    
    min_l = np.minimum(len(surface_points<-1990),len(surface_points_target))
    usefull = np.logical_and(~np.any(surface_points<-1990,axis=1)[:min_l],~np.any(surface_points_target<-1990,axis=1)[:min_l])
    surface_points = surface_points[:min_l][usefull]
    surface_points_target = surface_points_target[:min_l][usefull]
    #surface_points_target = surface_points_target[~np.any(surface_points_target<-1990,axis=1)]

    #sys_noi2 = syst_noise2(np.linalg.norm(surface_points_target-surface_points,axis=1),rand_uvec,noise_search[i])
    
    surface_pointsn = surface_points 
    
    print(surface_pointsn.shape,surface_points_target.shape)


    callback = partial(visualize, ax=None)

    reg = arnew(**{ 'X': np.copy(surface_pointsn), 'Y': np.copy(surface_points_target)})
    
    reg2 = arnew(**{ 'X': np.copy(surface_points_target), 'Y': np.copy(surface_pointsn)})
    
    
    #reg2 = arnew(**{ 'X': np.copy(surface_points2), 'Y': np.copy(surface_points_target)})
    
    
    reg.register(callback)
    reg2.register(callback)
    
    #reg2 = deformable_registration(**{ 'X': np.copy(surface_pointsn), 'Y': np.copy(surface_points_target)})
    
    #reg2.register(callback)

    Af_est = reg.B
    
    
    #reg2 = arnew(**{ 'X': surface_points, 'Y': surface_points_target })
    #reg2.register(callback)
    
    #Af_est2 = reg2.registration_parameters()[0]
    

    #true_Af = np.linalg.inv(Af)

    #error_Af = Af_est-true_Af
    #error_Af2 = Af_est2-true_Af

    #plt.imshow(true_Af,vmin=-0.1,vmax=1.4)
    #plt.show()
    
    device_change = reg.B-np.diag(np.ones(reg.B.shape[0]))
    device_change2 = reg2.B-np.diag(np.ones(reg.B.shape[0]))
    
    m_cmap = np.maximum(np.abs(device_change).max(),np.abs(device_change2).max())
    
    m_cmap = 0.3
    
    
    plt.imshow(device_change.T,vmin=-m_cmap,vmax=m_cmap,cmap='PuOr')
    plt.colorbar()
    plt.show()
    
    plt.imshow(device_change2.T,vmin=-m_cmap,vmax=m_cmap,cmap='PuOr')
    plt.colorbar()
    plt.savefig("basel_1_cd_reg.svg")
    plt.show()

    print(np.diag(Af_est))#,np.diag(true_Af))
    #print(Af_est[1,0],true_Af[1,0])
    #print(np.diag(error_Af),error_Af[1,0])
    #print(np.sum(np.abs(error_Af)))
    #erros += [np.sum(np.abs(error_Af))]
    
    print(np.linalg.det(reg.B))
    



#plt.plot(noise_search,erros)


#print(rand_uvec)
#plt.show()

    
    