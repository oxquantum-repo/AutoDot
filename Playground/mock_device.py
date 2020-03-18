

from . import standard_shapes
import numpy as np

def build_mock_device_with_json(config):

    shapes = []
    for key,item in config['Shapes'].items():
        shapes += [getattr(standard_shapes,key)(config['ndim'], **item)]
        
        
    device = Device(shapes,3)
        
        
    return device




class scale_for_device():
    def __init__(self,origin,dir):
        self.origin = origin
        self.dir = dir
    def __call__(self,params):
        return  (params - self.origin)*self.dir

class Device():
    def __init__(self,shapes,ndim,origin=None,dir=None):
    
        origin = np.array([0]*ndim) if origin is None else origin
        dir = np.array([-1]*ndim) if dir is None else dir
        
        #Negative dir prefered so flip signs
        dir = dir*-1
        self.scaledev = scale_for_device(origin,dir)
        
        self.shapes = shapes
        
    def jump(self, params):
        self.params = params
        return params
        
    def measure(self):
        return float(np.any([shape(self.params) for shape in self.shapes]))
        
    def check(self):
        return self.params