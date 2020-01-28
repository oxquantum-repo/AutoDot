# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:42:32 2019

@author: Dominic
"""

import numpy as np
from builtins import super

def initialize_sigma2(X, Y):
    (N, D), (M, _)  = X.shape, Y.shape
    diff = X[np.newaxis,...] - Y[:,np.newaxis,:]
    err  = diff * diff
    return np.sum(err) / (D * M * N)

class expectation_maximization_registration(object):
    def __init__(self, X, Y, sigma2=None, max_iterations=1000, tolerance=0.001, w=0, *args, **kwargs):
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Both point clouds need to have the same number of dimensions.")

        self.X, self.Y = X, Y
        self.sigma2 = sigma2
        (self.N, self.D),(self.M, _) = self.X.shape, self.Y.shape
        self.tolerance      = tolerance
        self.w              = w
        self.max_iterations = max_iterations
        self.iteration      = 0
        self.err            = self.tolerance + 1
        self.P, self.Pt1, self.P1 = np.zeros((self.M, self.N)), np.zeros((self.N, )), np.zeros((self.M, ))
        self.Np             = 0

    def register(self, callback=lambda **kwargs: None):
        self. TY = self.transform_point_cloud(self.Y)
        if self.sigma2 is None:
            self.sigma2 = initialize_sigma2(self.X, self.TY)
        self.q = -self.err - self.N * self.D/2 * np.log(self.sigma2)
        while self.iteration < self.max_iterations and self.err > self.tolerance:
            self.do_iteration()
            if callable(callback):
                callback(iteration=self.iteration, error=self.err, X=self.X, Y=self.TY)
        return self.TY, self.registration_parameters()

    def registration_parameters(self):
        raise NotImplementedError("Registration parameters should be defined in child classes.")

    def do_iteration(self):
        self.e_step()
        self.m_step()
        self.iteration += 1

    def e_step(self):
        """
        Perform E step of registration
        """
        #compute distance between points
        diff     = self.X[:,np.newaxis] - self.TY
        diff     = diff*diff
        
        #store dists
        P  = np.sum(diff, axis=-1).T
        
        #compute constant factor in denominator
        c = ((2 * np.pi * self.sigma2) ** (self.D / 2)) * (self.w / (1 - self.w)) * self.M / self.N
        
        #compute denominator
        P = np.exp(-P / (2 * self.sigma2)) 
        denom = np.sum(P, axis=0)
        denom[denom==0] = np.finfo(float).eps
        denom += c
        
        #compute P
        self.P = np.divide(P,denom)

    def m_step(self):
        """
        Perform M step of registration
        """
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1  = np.sum(self.P, axis=1)
        self.Np  = np.sum(self.P1)
        
        self.solve()
        self.TY = self.transform_point_cloud(self.Y)
        self.update_variance()




class simple_affine_registration(expectation_maximization_registration):
    def __init__(self, B=None, t=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.B = np.eye(self.D) if B is None else B
        self.t = np.zeros([1, self.D]) if t is True else t

    def solve(self):
        """
        Main bulk if the m step calculations specific to type of registration
        """
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.Xhat = self.X - muX
        Yhat = self.Y - muY
                
        self.A = np.transpose(self.Xhat) @ np.transpose(self.P) @ Yhat
        
        self.YPY = np.transpose(Yhat) @ np.diag(self.P1) @ Yhat
        
        self.B = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        if self.t is not None:
            self.t = np.transpose(muX) - np.transpose(self.B) @ np.transpose(muY)
    def transform_point_cloud(self, Y):
        """
        Transform a given point cloud
        """
        if self.t is None:
            return Y @ self.B
        else:
            return Y @ self.B + self.t
    def inverse_transform_point_cloud(self,Y):
        """
        Inverse transform a given point cloud
        """
        return (Y - self.t) @ np.linalg.inv(self.B) 

    def update_variance(self):
        """
        Compute new sigma
        """
        qprev = self.q
        
        trAB     = np.trace(self.A @ self.B)
        xPx      = np.transpose(self.Pt1) @ np.sum(self.Xhat*self.Xhat, axis=1)
        self.q   = (xPx - 2 * trAB + np.trace(self.B @ self.YPY @ self.B)) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.err = np.abs(self.q - qprev)

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)
        
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def registration_parameters(self):
        return self.B, self.t
    
  