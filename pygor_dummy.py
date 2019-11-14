import numpy as np

def L2_norm(x):
    return np.sqrt(np.sum(np.square(x),axis=-1))

class Circle(object):
    def __init__(self, r, ndim, origin=0.0):
        if np.isscalar(origin):
            self.origin = origin * np.ones(ndim)
        else:
            self.origin = np.array(origin)
        self.r = r
        self.ndim = ndim

    def __call__(self, x):
        return L2_norm(np.array(x) - self.origin) <= self.r

class Box(object):
    def __init__(self, ndim, a, b):
        a = np.array(a)
        b = np.array(b)

        if np.isscalar(a):
            a = a * np.ones(ndim)
        if np.isscalar(b):
            b = b * np.ones(ndim)
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
        inside = np.logical_and(np.all(x > self.lb, axis=-1), np.all(x < self.ub, axis=-1))
        return inside

class Leakage(object):
    def __init__(self, shape, th_leak):
        self.shape = shape
        self.th_leak = th_leak
        self.ndim = shape.ndim
    def __call__(self, x):
        x = np.array(x)
        inside = self.shape(x)
        leak = np.any(x > self.th_leak, axis=-1)
        inside = np.logical_or(inside, leak)
        return inside

class Convexhull(object):
    def __init__(self, points):
        # points: 2D array (num_points x ndim)
        from scipy.spatial import Delaunay
        self.hull = Delaunay(points)
        self.ndim = points.shape[1]
    def __call__(self, x):
        return self.hull.find_simplex(x) >= 0

class Box_FreeLowerVertex(Convexhull):
    def __init__(self, ndim, a, b, a_prime):
        a = np.array(a)
        b = np.array(b)

        if np.any(a > b): raise ValueError('a should be less than b')

        vertices = np.array(np.meshgrid(*list(zip(a,b)))).T.reshape(-1,ndim)
        # Replace the first vertex with b
        vertices[0] = a_prime

        from scipy.spatial import Delaunay
        self.hull = Delaunay(vertices)
        self.ndim = vertices.shape[1]

class PygorDummyBinary(object):
    def __init__(self, lb, ub, test_in):
        if len(lb) != len(ub):
            raise ValueError('Length of lb should be same with ub')
        self.num_params = len(lb)
        self.params = [0.0] * self.num_params
        self.names = ["c{}".format(i+1) for i in range(self.num_params)]
        self.name_to_idx = dict(zip(self.names, range(self.num_params)))
        self.test_in = test_in

    def set_params(self, params):
        assert len(self.params) == len(params)
        assert isinstance(params, list)
        self.params = params
        return self.params

    def get_params(self):
        return self.params
    
    def setval(self, var, val):
        self.params[self.name_to_idx[var]] = val
        return self.params[self.name_to_idx[var]]

    def getval(self, var):
        return self.params[self.name_to_idx[var]]

    def do0d(self):
        if self.test_in(self.params):
            return [[1.0]]
        else:
            return [[0.0]]

    def do1d(self, var1, min1, max1, res1):
        result = np.zeros(res1)
        for i, val in enumerate(np.linspace(min1,max1,res1)):
            self.setval(var1, val)
            result[i] = self.do0d()[0][0]
        return result[np.newaxis,...].tolist()

    def do2d(self, var1, min1, max1, res1, var2, min2, max2, res2):
        result = np.zeros((res1,res2))
        for i , val1 in enumerate(np.linspace(min1,max1,res1)):
            self.setval(var1,val1)
            for j, val2 in enumerate(np.linspace(min2,max2,res2)):
                self.setval(var2,val2)
                result[i,j] = self.do0d()[0][0]
        return result[np.newaxis,...].tolist()

