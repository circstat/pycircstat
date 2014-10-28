"""
Statistical tests
"""
import numpy as np
from scipy import misc
#import warnings
from pycircstat import descriptive
from pycircstat import utils

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
    :param axis:  compute along this dimension, default is 0
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
    

    r = descriptive.resultant_vector_length(alpha, w=w, d=d, axis=axis)
    n = np.sum(w)
    
    # compute Rayleigh's R (equ. 27.1)
    R = n*r

    # compute Rayleigh's z (equ. 27.2)
    z = R**2 / n;

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1+4*n+4*(n**2-R**2))-(1+2*n))

    return pval, z


def omnibus(alpha, w=None, sz=np.radians(1), axis=0):
    """
    Computes omnibus test for non-uniformity of circular data. The test is also 
    known as Hodges-Ajne test.
    
    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle
 
    Alternative to the Rayleigh and Rao's test. Works well for unimodal,
    bimodal or multimodal data. If requirements of the Rayleigh test are 
    met, the latter is more powerful.

    :param alpha: sample of angles in radian
    :param w: 	 number of incidences in case of binned angle data
    :param sz:    step size for evaluating distribution, default 1 deg
    :param axis:  compute along this dimension, default is 0
                  if axis=None, array is raveled  
    :return pval: two-tailed p-value
    :return m:    minimum number of samples falling in one half of the circle

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    if axis is None:
        axis = 0
        alpha = alpha.ravel()

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"
    
    alpha = alpha % (2*np.pi)
    n = np.sum(w);
    dg = np.arange(0,np.pi,utils.ang2rad(1))


    m1 = np.zeros_like(dg)
    m2 = np.zeros_like(dg)   
    
    for i,dg_val in enumerate(dg):
        m1[i] = np.sum(w[(alpha>dg_val) & (alpha < np.pi+dg_val)])
        m2[i] = n - m1[i]
        
    m = np.hstack((m1,m2)).min()

    if n > 50:
      # approximation by Ajne (1968)
      A = np.pi*np.sqrt(n) / 2 / (n-2*m)
      pval = np.sqrt(2*np.pi) / A * np.exp(-np.pi**2/8/A**2)
    else:
      # exact formula by Hodges (1955)
      pval = 2**(1-n) * (n-2*m) * misc.comb(n,m)  
    
    return pval, m


