

from . import shapes
import numpy as np

def build_mock_device_with_json(config):

    shapes_list = []
    for key,item in config['shapes'].items():
        shapes_list += [getattr(shapes,key)(config['ndim'], **item)]
        
        
    device = Device(shapes_list,config['ndim'])
        
        
    return device




class scale_for_device():
    def __init__(self,origin,dir):
        self.origin = origin
        self.dir = dir
    def __call__(self,params):
        return  (params - self.origin)*self.dir

class Device():
    def __init__(self,shapes_list,ndim,origin=None,dir=None):
    
        origin = np.array([0]*ndim) if origin is None else origin
        dir = np.array([-1]*ndim) if dir is None else dir
        
        #Negative dir prefered so flip signs
        dir = dir*-1
        self.sd = scale_for_device(origin,dir)
        
        self.shapes_list = shapes_list
        
    def jump(self, params):
        self.params = np.array(params)
        return params
        
    def measure(self):
        return float(np.any([shape(self.sd(self.params)) for shape in self.shapes_list]))
        
    def check(self,idx=None):
        return self.params if idx is None else self.params[idx]