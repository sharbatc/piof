import numpy as np
from math import factorial

def normalise_angle ( phi ):
    """
        Normalises the angle to the unit circle, in the interval ]-pi,pi]. 
        The small epsilon here is in order to deal with angles of the form pi+2*k*pi, k in Z,
        and avoiding an infinite recursion
    """
    epsilon = 5e-16
    if phi <= np.pi+epsilon and phi > -np.pi+epsilon:
        return phi
    elif phi > np.pi+epsilon:
        return normalise_angle(phi-2*np.pi)
    else:
        return normalise_angle(phi+2*np.pi)

def make_D_t ( alpha ):
    """
        Returns the time kernel D_t(\tau) formatted for a fixed alpha.
        
        N.B. nan_to_num prevents the function from returning nan when evaluated at a large negative number 
        (it might however make it slower)
    """
    def D_t ( t ):
        at = alpha*t
        return np.nan_to_num( (t > 0) *alpha*np.exp(-at)*(at**5/factorial(5)) )#-at**7/factorial(7))) 
    return D_t

def make_D_t_reverse ( alpha ):
    """
        Returns the time kernel D_t(\tau) formatted for a fixed alpha, including the term in power 7 so that the kernel reverses.
        
        N.B. nan_to_num prevents the function from returning nan when evaluated at a large negative number 
        (it might however make it slower)
    """
    def D_t_rev ( t ):
        at = alpha*t
        return np.nan_to_num((t > 0) *alpha*np.exp(-at)*(at**5/factorial(5)-at**7/factorial(7)))
    return D_t_rev
        