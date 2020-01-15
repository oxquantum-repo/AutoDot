# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:42:32 2019

@author: Dominic
"""

import numpy as np
from functools import partial
from Registration.registration_core import notranslation_affine_registration,simple_affine_registration,deformable_registration



def error(iteration, error, X, Y):
    
    min_l = np.minimum(X.shape[0],Y.shape[0])
    
    abs_error = np.linalg.norm(X[:min_l]-Y[:min_l],axis=-1)
    
    print(iteration,error,np.sum(abs_error)/min_l)


def register_point_clouds_from_file(fname1,fname2,reg_func=notranslation_affine_registration):
    pointset1 = np.load(fname1)
    pointset2 = np.load(fname2)
    
    assert np.array_equal(pointset1.shape,pointset2.shape)
    
    return register_point_clouds(pointset1,pointset2,reg_func)


def register_point_clouds(pointset1,pointset2,reg_func=notranslation_affine_registration):
    """
    register point sets NOTE: pointset1 is the fixed set
    """
    
    cb = partial(error)
    
    reg = reg_func(X=np.copy(pointset1),Y=np.copy(pointset2))
    
    reg.register(cb)
    
    return [reg.registration_parameters()],reg.TY


def compare_surf_to_hypercube(pointset1,reg_func=notranslation_affine_registration,origin=None,dirs=None,nhypercube=None):
    
    n_samps = pointset1.shape[0]
    
    if origin is None:
        origin = np.array([0]*n_samps)
        
    if dirs is None:
        dirs = np.array([-1]*n_samps)
    
    