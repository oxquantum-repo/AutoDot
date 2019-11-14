class Hypersurface(object):
    '''
    A probabilistic hypersurface model should have the following functions:
    - Calculate the probability that 'v' is inside of the hypersurface.
    - Generate pdf, cdf, or random samples of 'd_g' given 'v'
    
    Available data: 2D array of points (num_points x ndim)  on the hypersurface
    '''
    def set_data(self, vs, **kargs):
        self.data = vs

    def is_inside(self, v, **kargs):
        '''
        Return: probability that 'v' is inside of the hypersurface
        '''
        return 0.0
    
    def generate_samples_d(self, v, num_samples=1, **kargs):
        '''
        Return: returns 2D array (num_samples x ndim) of vector 'd' given 'v'
        '''
        return None

    def prob_dg(self, v, g_idx, dg, pdf_on=False, cdf_on=False, **kargs):
        '''
        Inputs
            v: 1D array (length == ndim)
            g_idx: integer index of an axis
            dg: scalar or 1D array (allow multiple g to get cdf and pdf)
        Returns
            pdf: None or 1D array (length == len(g))
            cdf: None or 1D array (length == len(g))
        '''
        return None, None

