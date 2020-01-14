# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:47:35 2019

@author: Dominic
"""

import numpy as np

def generate_points_on_hypercube(nsamples,origin,poffs,p=None,uvecs=None):
    
    
    
    
    if uvecs is None:
        
        
        epsilon = []
        bounds = []
        for i in range(len(origin)):
            origin_c = np.copy(origin)
            poffs_c = np.copy(poffs)
            origin_c[i] = poffs_c[i]
            bounds += [origin_c]
            print(origin_c,poffs_c)
                
            epsilon += [np.linalg.norm(origin_c-poffs_c)]
        epsilon = np.array(epsilon)
        if p is None:
            p = epsilon/epsilon.sum()
        
        
        print(p)
        points = []
        for i in range(nsamples):
            face = np.random.choice(len(origin),p=p)
            
            points+=[np.random.uniform(bounds[face],poffs)]
    return np.array(points)


def clean_pointset(pointset):
    
    pointset = np.copy(pointset)
    
    for point in pointset:
        toremove = np.where(np.all(np.less(pointset,point),axis=1))[0]
        
        pointset = np.delete(pointset,toremove,axis=0)
    #for point in pointset:
    #    print(np.less(pointset,point))
    #    print(np.where(np.logical_all(pointset<point)))
    return pointset


if __name__ == "__main__":
    
    p = generate_points_on_hypercube(200,[120,40],[-200,-300],None)
    print(p)
    
    import matplotlib.pyplot as plt
    plt.scatter(*p.T)
    plt.show()