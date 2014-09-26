"""
Descriptive statistical functions
"""

import numpy as np
from scipy import stats
import warnings

def mean(alpha, w=None, axis=0):
    """
    Compute mean direction of circular data.

    :param alpha: circular data
    :param w: 	 weightings in case of binned angle data
    :param axis: compute along this dimension, default is 0
    :return: circular mean
    """

    if w is None:
        w = np.ones_like(alpha)

    if alpha.shape != w.shape:
        raise ValueError('Input dimensions do not match:', alpha.shape)

    r = (w * np.exp(1j*alpha)).sum(axis=axis)

    mu = np.angle(r)

    return mu

def mean_ci(alpha, w=None, xi=0.05, d=None, axis=0):
    """
    Compute mean, upper, and lower confidence interval. For parameters see
    :func:`mean` and :func:`confmean`.

    :return: mean, upper confidence limit, lower confidence limit
    """

    mu = mean(alpha, w=w, axis=axis)
    t = confmean(alpha,xi=xi,w=w,d=d, axis=axis)
    return mu, mu + t, mu - t


def resultant_vector_length(alpha, w=None, d=None, axis=0):
    """
    Computes mean resultant vector length for circular data.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is 0
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of data and w do not match!"

    r = np.sum(w*np.exp(1j*alpha), axis=axis)

    # obtain length
    r = np.abs(r)/np.sum(w, axis=axis)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        c = d/2/np.sin(d/2);
        r = c*r

    return r


def confmean(alpha, xi=0.05, w=None, d=None, axis=0):
    """
    Computes the confidence limits on the mean for circular data.

    :param alpha: sample of angles in radians
    :param xi: (1-xi)-confidence limits are computed, default 0.05
    :param w: number of incidences in case of binned angle data
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is 0

    :return: confidence limit width d; mean +- d yields upper/lower (1-xi)% confidence limit

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha)

    assert alpha.shape == w.shape, "Dimensions of data and w do not match!"

    r = resultant_vector_length(alpha,w=w,d=d,axis=axis)
    n = np.sum(w,axis=axis)
    R = n*r
    c2 = stats.chi2.ppf(1-xi, df=1)

    t = np.NaN * np.empty_like(r)

    idx = (r < .9) & (r > np.sqrt(c2/2/n))
    t[idx] = np.sqrt((2*n[idx]*(2*R[idx]**2-n[idx]*c2))/(4*n[idx]-c2)) # eq. 26.24

    idx2 = (r >= .9)
    t[idx2] = np.sqrt(n[idx2]**2-(n[idx2]**2-R[idx2]**2)*np.exp(c2/n[idx2])) # equ. 26.25

    if not np.all(idx | idx2):
        warnings.warn('Requirements for confidence levels not met.')

    return np.arccos(t/R)


