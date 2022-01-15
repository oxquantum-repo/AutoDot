import math

class CMAESConfig(object):
    """
    Calculates the the starting position, variance and bounds for the CMA-ES.
    Saves the position and bounds es vectors defined by angles.
    """

    def __init__(self, lb, ub):
        """"
        Calculates the the starting position, variance and bounds given the bounds of the Sampler.
        """
        dim = len(lb)-1
        self.lower_bound = self.eval_lower_bound(lb, ub)
        self.upper_bound = self.eval_upper_bound(lb, ub)
        self.x0 = self.eval_x0()
        self.sigma0 = self.eval_sigma0()


    def eval_lower_bound(self, lb, ub): 
        dim = len(lb)-1
        return [lower_bound_of_angle_between(lb[i], ub[i], lb[i+1], ub[i+1]) for i in range(dim)]


    def eval_upper_bound(self, lb, ub):
        dim = len(lb)-1
        return [upper_bound_of_angle_between(lb[i], ub[i], lb[i+1], ub[i+1]) for i in range(dim)]

    
    def eval_x0(self):
        """
        Mean vector between lower and upper bound
        """
        return [(self.upper_bound[i]+self.lower_bound[i])/2 for i in range(len)]

    
    def eval_sigma0(self):
        """
        Recommended from paper
        """
        return 0.3* (max(self.upper_bound)-max(self.lower_bound))

    
    def get_lower_bound(self):
        return self.lower_bound


    def get_upper_bound(self):
        return self.upper_bound

    
    def get_x0(self):
        return self.x0


    def get_sigma0(self):
        return self.sigma0


def lower_bound_of_angle_between(lb_i, ub_i, lb_j, ub_j):
    """
    Given the lower and upper bound of two dimensions,
    it returns the lower bound of the included angle
    depending on the covered sector in the coordiante system. 
    """
    # Arrangements with three sectors aren't considered, 
    # because it's not possible to include them with bounds.
    
    if lb_i >= 0 and ub_i > 0 and lb_j < 0 and ub_j > 0:
        # sector I, IV
        return -math.pi/2
    if ub_i > 0 and lb_j >= 0 and ub_j > 0:
        # sector I and I, II
        return 0
    if lb_i < 0 and ub_i <= 0 and ub_j > 0:
        # sector II and II, III
        return math.pi/2
    if lb_i < 0 and lb_j < 0 and ub_j <= 0:
        # sector III and III, IV
        return math.pi
    if lb_i >= 0 and ub_i > 0 and lb_j < 0 and ub_j <= 0:
        # sector IV
        return (3/2)*math.pi

    # sector I, II, III, IV
    return 0
    

def upper_bound_of_angle_between(self, lb_i, ub_i, lb_j, ub_j):
    """
    Given the lower and upper bound of two dimensions,
    it returns the upper bound of the included angle
    depending on the covered sector in the coordiante system. 
    """
    # Arrangements with three sectors aren't considered, 
    # because it's not possible to include them with bounds.

    if lb_i >= 0 and ub_i > 0 and ub_j > 0:
        # sector I and I, IV
        return math.pi/2
    if lb_i < 0 and lb_j >= 0 and ub_j > 0:
        # sector II and I, II
        return math.pi
    if lb_i < 0 and ub_i <= 0 and lb_j < 0:
        # sector III and II, III
        return (3/2)*math.pi
    if ub_i > 0 and lb_j < 0 and ub_j <= 0:
        # sector IV and III, IV
        return 2*math.pi

    # sector I, II, III, IV
    return 2*math.pi
