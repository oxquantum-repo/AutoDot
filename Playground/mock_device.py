

from . import shapes
import numpy as np

def build_mock_device_with_json(config):

    shapes_list = []
    for key,item in config['shapes'].items():
        shapes_list += [getattr(shapes,key)(config['ndim'], **item)]
        
    dirs = config.get('dir',None)
    origin = config.get('origin',None)
    device = Device(shapes_list,config['ndim'],origin,dirs)
        
        
    return device




class scale_for_device():
    def __init__(self,origin,dir):
        self.origin = origin
        self.dir = dir
    def __call__(self,params,bc=True):
        if bc:
            return  (params - self.origin[np.newaxis,:])*self.dir[np.newaxis,:]
        else:
            return  (params - self.origin)*self.dir

class Device():
    def __init__(self,shapes_list,ndim,origin=None,dir=None):
    
        origin = np.array([0]*ndim) if origin is None else np.array(origin)
        dir = np.array([-1]*ndim) if dir is None else np.array(dir)
        
        #Negative dir prefered so flip signs
        dir = dir*-1
        self.sd = scale_for_device(origin,dir)
        
        self.shapes_list = shapes_list
        
    def jump(self, params):
        self.params = np.array(params)[np.newaxis,:]
        return params
        
    def measure(self):
        return float(np.any([shape(self.sd(self.params)) for shape in self.shapes_list]))
        
    def check(self,idx=None):
        return self.params.squeeze() if idx is None else self.params.squeeze()[idx]
    
    def arr_measure(self,params):
        shape_logical = [shape(self.sd(params)) for shape in self.shapes_list]
        
        shape_logical = np.array(shape_logical)
        
        return np.any(shape_logical,axis=0)
        
        
