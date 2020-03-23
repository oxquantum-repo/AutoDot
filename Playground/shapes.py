import numpy as np


def L2_norm(x):
    return np.sqrt(np.sum(np.square(x),axis=-1))
    


class Circle(object):
    def __init__(self, ndim, r = 1000, origin=0.0):
        if np.isscalar(origin):
            self.origin = origin * np.ones(ndim)
        else:
            self.origin = np.array(origin)
        self.r = r
        self.ndim = ndim

    def __call__(self, x):
        return L2_norm(np.array(x) - self.origin) <= self.r

class Box(object):
    def __init__(self, ndim, a=-1000, b=0):
        if np.isscalar(a):
            a = a * np.ones(ndim)
        if np.isscalar(b):
            b = b * np.ones(ndim)
            
        a = np.array(a)
        b = np.array(b)
           
        if len(a) != ndim or len(b) != ndim:
            raise ValueError('Wrong dimensions for defining a box')
        if all( a < b):
            self.lb, self.ub  = a, b
        elif all( a > b):
            self.lb, self.ub  = b, a
        else:
            raise ValueError('Wrong points for defining a box')
        self.ndim = ndim

    def __call__(self, x):
        x = np.array(x)
        inside = np.logical_and(np.all(x > self.lb[np.newaxis,:], axis=-1), np.all(x < self.ub[np.newaxis,:], axis=-1))
        return inside

class Leakage(object):
    def __init__(self, ndim, th_leak=-500, idx=0):
        self.th_leak = th_leak
        self.leak_gate = idx
    def __call__(self, x):
        x = np.array(x)
        leak = x[:,self.leak_gate] > self.th_leak
        return leak
#change
class Convexhull(object):
    def __init__(self, ndim, points=[-1000,0,[-1000,0],[0,-1000]]):
        # points: 2D array (num_points x ndim)
        for i,point in enumerate(points):
            if np.isscalar(point):
                points[i] = point * np.ones(ndim)
              
        points = np.array(points)
        
        from scipy.spatial import Delaunay
        self.hull = Delaunay(points)
        self.ndim = points.shape[1]
    def __call__(self, x):
        return self.hull.find_simplex(x) >= 0

class Crosstalk_box(Convexhull):
    def __init__(self, ndim, a=-1500, b=1500, a_prime=-1000):
    
    
        if np.isscalar(a):
            a = a * np.ones(ndim)
        if np.isscalar(b):
            b = b * np.ones(ndim)
        if np.isscalar(a_prime):
            a_prime = a_prime * np.ones(ndim)
    
    
        a = np.array(a)
        b = np.array(b)

        if np.any(a > b): raise ValueError('a should be less than b')

        vertices = np.array(np.meshgrid(*list(zip(a,b)))).T.reshape(-1,ndim)
        # Replace the first vertex with b
        vertices[0] = a_prime

        from scipy.spatial import Delaunay
        self.hull = Delaunay(vertices)
        self.ndim = vertices.shape[1]
