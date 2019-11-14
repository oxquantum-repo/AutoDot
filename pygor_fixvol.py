import numpy as np

class PygorRewire(object):
    def __init__(self, pg_raw, new_wires):
        '''
        Example of new_wires
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 1, 1, 0, 0]]
        '''
        self.pg_raw = pg_raw
        self.pygor = pg_raw.pygor
        #assert len(lb_short) == len(ub_short)
        #ndim_raw = len(lb_short)
        #if not np.all([len(x) == ndim_raw for x in new_wires]):
        #    raise ValueError('Invalid wiring(1).')
        if not np.all(np.sum(new_wires, axis=0) == 1):
            raise ValueError('Invalid wiring(2).')
        if not np.all(np.sum(new_wires, axis=1) > 0):
            raise ValueError('Invalid wiring(3).')
        self.new_wires = np.array(new_wires)

        '''
        ndim_new = len(new_wires)
        self.lb_new = np.zeros(ndim_new)
        self.ub_new = np.zeros(ndim_new)
        for i in range(ndim_new):
            lb_raw_ = lb_raw[new_wires[i]]
            if not np.all(lb_raw_[1:] == lb_raw_[0]):
                raise ValueError('Invalid wiring(4).')
            self.lb_new[i] = lb_raw_[0]

            ub_raw_ = ub_raw[new_wires[i]]
            if not np.all(ub_raw_[1:] == ub_raw_[0]):
                raise ValueError('Invalid wiring(5).')
            self.ub_new[i] = ub_raw_[0]
        '''

    def convert_to_raw(self, params):
        return np.matmul(self.new_wires.T, params)

    ### pygor functions ###
    def set_params(self, params):
        params_raw = self.convert_to_raw(params).tolist()
        return self.pg_raw.set_params(params_raw)

    def do0d(self):
        return self.pygor.do0d()

class PygorFixVol(object):
    def __init__(self, pygor):
        self.pygor = pygor
        self.num_params = len(pygor.get_params())
        self.fixed_mask = [False] * self.num_params
        self.fixed_vals = [0.0] * self.num_params
        self.names = ["c{}".format(i+1) for i in range(16)] # index -> name
        # name -> index:  self.names.index("name")

    def fix_vols(self, fix_new):
        # fix_new: (name, param) dictionary
        for name, param in fix_new.items():
            idx = self.names.index(name)
            self.fixed_vals[idx] = param
            self.fixed_mask[idx] = True
            self.pygor.setval(name, param)

    def release_vols(self, names):
        # names: string or list of strings of parameter names
        if type(names) is list:
            for name in names:
                idx = self.names.index(name)
                self.fixed_vals[idx] = 0.0
                self.fixed_mask[idx] = False
        elif type(names) is str:
            idx = self.names.index(names)
            self.fixed_vals[idx] = 0.0
            self.fixed_mask[idx] = False
        else:
            raise ValueError('names should be a list or string')

    def get_extended_vols(self, params_short):
        num_params_short = self.num_params - np.sum(self.fixed_mask)
        if len(params_short) != num_params_short:
            raise ValueError('Wrong length: params_short')

        params_long = self.fixed_vals.copy()
        counter = 0
        for i in range(self.num_params):
            if not self.fixed_mask[i]:
                params_long[i] = params_short[counter]
                counter += 1
        #params_long = np.zeros((self.num_params,))
        #params_long[np.logical_not(self.fixed_mask)] = params_short
        #params_long[self.fixed_mask] = self.fixed_vals[self.fixed_mask]
        return params_long

    def get_shortened_vols(self, params_long):
        if len(params_long) != self.num_params:
            raise ValueError('Wrong length: params_long')

        params_short = [params_long[i] for i in range(self.num_params) if not self.fixed_mask[i]]
        #params_short = params_long[np.logical_not(self.fixed_mask)]
        return params_short

    ### pygor functions ###
    def set_params(self, params_short):
        params_long = self.get_extended_vols(params_short)
        return self.pygor.set_params(params_long)

    def get_params(self):
        params_long = self.pygor.get_params()
        return self.get_shortened_vols(params_long)

    def setval(self, var, val):
        idx = self.names.index(var)
        if self.fixed_mask[idx]:
            self.fixed_vals[idx] = val
        return self.pygor.setval(var,val)

    def getval(self, var):
        return self.pygor.getval(var)

    def do0d(self):
        return self.pygor.do0d()

    def do1d(self, var1, min1, max1, res1):
        return self.pygor.do1d(var1, min1, max1, res1)

    def do2d(self, var1, min1, max1, res1, var2, min2, max2, res2):
        return self.pygor.do2d(var1, min1, max1, res1, var2, min2, max2, res2)
