"""
Statistical tests
"""
import numpy as np
from scipy import stats
import warnings
from pycircstat import descriptive

def rayleigh(alpha, w=None, d=None, axis=0):
    """
    Computes Rayleigh test for non-uniformity of circular data.
    
    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle
 
    Assumption: the distribution has maximally one mode and the data is 
    sampled from a von Mises distribution!

    :param alpha: sample of angles in radian
    :param w: 	  number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is 1
                  if axis=None, array is raveled  
    :return pval: two-tailed p-value
    :return z:    value of the z-statistic

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    if axis is None:
        axis = 0
        alpha = alpha.ravel()

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"
    

    r = resultant_vector_length(alpha, w=w, d=d, axis=axis)
    n = np.sum(w)
    
    # compute Rayleigh's R (equ. 27.1)
    R = n*r

    # compute Rayleigh's z (equ. 27.2)
    z = R**2 / n;

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1+4*n+4*(n**2-R**2))-(1+2*n))

    return pval, z



