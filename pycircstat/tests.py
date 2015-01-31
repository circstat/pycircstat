"""
Statistical tests
"""
from __future__ import absolute_import

import numpy as np
from scipy import misc
from scipy import stats
#import warnings
from . import descriptive, swap2zeroaxis
from . import utils


@swap2zeroaxis(['alpha','w'], [0, 1])
def rayleigh(alpha, w=None, d=None, axis=None):
    """
    Computes Rayleigh test for non-uniformity of circular data.

    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle

    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!

    :param alpha: sample of angles in radian
    :param w:       number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return z:    value of the z-statistic

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    # if axis is None:
    #     axis = 0
    #     alpha = alpha.ravel()

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    r = descriptive.resultant_vector_length(alpha, w=w, d=d, axis=axis)
    n = np.sum(w, axis=axis)

    # compute Rayleigh's R (equ. 27.1)
    R = n * r

    # compute Rayleigh's z (equ. 27.2)
    z = R ** 2 / n

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))

    return pval, z

@swap2zeroaxis(['alpha','w'], [0, 1])
def omnibus(alpha, w=None, sz=np.radians(1), axis=None):
    """
    Computes omnibus test for non-uniformity of circular data. The test is also
    known as Hodges-Ajne test.

    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle

    Alternative to the Rayleigh and Rao's test. Works well for unimodal,
    bimodal or multimodal data. If requirements of the Rayleigh test are
    met, the latter is more powerful.

    :param alpha: sample of angles in radian
    :param w:      number of incidences in case of binned angle data
    :param sz:    step size for evaluating distribution, default 1 deg
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return m:    minimum number of samples falling in one half of the circle

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    alpha = alpha % (2 * np.pi)
    n = np.sum(w, axis=axis)

    dg = np.arange(0, np.pi, np.radians(1))

    m1 = np.zeros((len(dg),) + alpha.shape[1:])
    m2 = np.zeros((len(dg),) + alpha.shape[1:])


    for i, dg_val in enumerate(dg):
        m1[i,...] = np.sum(w * ((alpha > dg_val) & (alpha < np.pi + dg_val)), axis=axis)
        m2[i,...] = n - m1[i,...]

    m = np.concatenate((m1, m2), axis=0).min(axis=axis)

    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
    A = np.empty_like(n)
    pval = np.empty_like(n)
    idx50 = (n > 50)

    if np.any(idx50):
        A[idx50] = np.pi * np.sqrt(n[idx50]) / 2 / (n[idx50] - 2 * m[idx50])
        pval[idx50] = np.sqrt(2 * np.pi) / A[idx50] * np.exp(-np.pi ** 2 / 8 / A[idx50] ** 2)

    if np.any(~idx50):
        pval[~idx50] = 2 ** (1 - n[~idx50]) * (n[~idx50] - 2 * m[~idx50]) * misc.comb(n[~idx50], m[~idx50])

    return pval.squeeze(), m

@swap2zeroaxis(['alpha'], [0, 1, 2])
def raospacing(alpha, axis=None):
    """
    Calculates Rao's spacing test by comparing distances between points on
    a circle to those expected from a uniform distribution.

    H0: Data is distributed uniformly around the circle.
    H1: Data is not uniformly distributed around the circle.

    Alternative to the Rayleigh test and the Omnibus test. Does not assume
    a unimodal distribution as alternative. Less powerful than the Rayleigh
    test when the distribution is unimodal on a global scale but uniform
    locally.

    Due to the complexity of the distribution of the test statistic, we
    resort to the tables published by Russel and Levitin (references below).

    Therefore the reported p-value is the smallest alpha level at which the
    test would still be significant. If the test is not significant at the
    alpha=0.1 level, we return the critical value for alpha = 0.05 and p =
    0.5.

    :param alpha: sample of angles in radian
    :param axis:  compute along this dimension, default is 0
                  if axis=None, array is raveled
    :return pval: smallest p-value at which test is significant
    :return U:    test statistic
    :return Uc:   critical value at the p-value returned

    References: [Batschelet1981]_, [RusselLevitin1995]_
    """


    alpha = np.degrees(alpha)
    alpha = np.sort(alpha, axis=axis)

    n = alpha.shape[axis]
    assert n >= 4, 'Rao spacing test requires at least 4 samples'

    # compute test statistic along 0 dimension (swap2zeroaxis)
    U = 0.
    kappa = 360 / n
    for j in range(0, n - 1):
        ti = alpha[j + 1,...] - alpha[j,...]
        U = U + np.abs(ti - kappa)

    tn = 360 - alpha[-1,...] + alpha[0,...]
    U = U + abs(tn - kappa)

    U = .5 * U

    # get critical value from table
    pval, Uc = _critical_value_raospacing(n, U)

    return pval, U, Uc


def _critical_value_raospacing(n, U):
    # Table II from Russel and Levitin, 1995

    alpha_level = np.array([0.001, .01, .05, .10])

    table = np.array([
        4,   247.32, 221.14, 186.45, 168.02,
        5,   245.19, 211.93, 183.44, 168.66,
        6,   236.81, 206.79, 180.65, 166.30,
        7,   229.46, 202.55, 177.83, 165.05,
        8,   224.41, 198.46, 175.68, 163.56,
        9,   219.52, 195.27, 173.68, 162.36,
        10,  215.44, 192.37, 171.98, 161.23,
        11,  211.87, 189.88, 170.45, 160.24,
        12,  208.69, 187.66, 169.09, 159.33,
        13,  205.87, 185.68, 167.87, 158.50,
        14,  203.33, 183.90, 166.76, 157.75,
        15,  201.04, 182.28, 165.75, 157.06,
        16,  198.96, 180.81, 164.83, 156.43,
        17,  197.05, 179.46, 163.98, 155.84,
        18,  195.29, 178.22, 163.20, 155.29,
        19,  193.67, 177.08, 162.47, 154.78,
        20,  192.17, 176.01, 161.79, 154.31,
        21,  190.78, 175.02, 161.16, 153.86,
        22,  189.47, 174.10, 160.56, 153.44,
        23,  188.25, 173.23, 160.01, 153.05,
        24,  187.11, 172.41, 159.48, 152.68,
        25,  186.03, 171.64, 158.99, 152.32,
        26,  185.01, 170.92, 158.52, 151.99,
        27,  184.05, 170.23, 158.07, 151.67,
        28,  183.14, 169.58, 157.65, 151.37,
        29,  182.28, 168.96, 157.25, 151.08,
        30,  181.45, 168.38, 156.87, 150.80,
        35,  177.88, 165.81, 155.19, 149.59,
        40,  174.99, 163.73, 153.82, 148.60,
        45,  172.58, 162.00, 152.68, 147.76,
        50,  170.54, 160.53, 151.70, 147.05,
        75,  163.60, 155.49, 148.34, 144.56,
        100, 159.45, 152.46, 146.29, 143.03,
        150, 154.51, 148.84, 143.83, 141.18,
        200, 151.56, 146.67, 142.35, 140.06,
        300, 148.06, 144.09, 140.57, 138.71,
        400, 145.96, 142.54, 139.50, 137.89,
        500, 144.54, 141.48, 138.77, 137.33,
        600, 143.48, 140.70, 138.23, 136.91,
        700, 142.66, 140.09, 137.80, 136.59,
        800, 142.00, 139.60, 137.46, 136.33,
        900, 141.45, 139.19, 137.18, 136.11,
        1000,  140.99, 138.84, 136.94, 135.92])
    table = table.reshape((-1, 5))

    if not hasattr(U,'shape'):
        U = np.array(U)

    old_shape = U.shape
    U = U.ravel()
    Uc, p = 0*U, 0*U

    for i, loop_u in enumerate(U):
        ridx = (table[:, 0] >= n).argmax()
        cidx = (table[ridx, 1:] < loop_u).argmax()

        if (cidx > 0) | ((cidx == 0) & (table[ridx, cidx + 1] < loop_u)):
            Uc[i] = table[ridx, cidx + 1]
            p[i] = alpha_level[cidx]
        else:
            Uc[i] = table[ridx, -1]
            p[i] = .5

    return p.reshape(old_shape), Uc.reshape(old_shape)
    
@swap2zeroaxis(['alpha','w'], [0, 1])
def vtest(alpha, mu, w=None, d=None, axis=None):
    """
    Computes V test for nonuniformity of circular data with a known mean 
    direction of dir.

    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle but
        has a mean of mu
        
    Note: Not rejecting H0 may mean that the population is uniformly
    distributed around the circle OR that it has a mode but that this mode
    is not centered at dir.

    The V test has more power than the Rayleigh test and is preferred if
    there is reason to believe (before seeing the data!) in a specific
    mean direction. 


    :param alpha: sample of angles in radian
    :param mu:   suspected mean direction
    :param w:     number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return v:    value of the v-statistic

    References: [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha)
    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    r = descriptive.resultant_vector_length(alpha, w=w, d=d, axis=axis)
    m = descriptive.mean(alpha,w=w,d=d, axis=axis)
    n = np.sum(w, axis=axis)

    # compute Rayleigh's R (equ. 27.1)
    R = n * r

    # compute V and u (equ. 27.5)
    V = R * np.cos(m-mu)
    u = V *np.sqrt(2/n)
    # compute p value using approxation in Zar, p. 617
    pval = 1 - stats.norm.cdf(u)

    return pval, V
    

@swap2zeroaxis(['alpha'], [0, 1])
def symtest(alpha, axis=None):
    """
    Non-parametric test for symmetry around the median. Works by performing a 
    Wilcoxon sign rank test on the differences to the median.

    H0: the population is symmetrical around the median
    HA: the population is not symmetrical around the median
        

    :param alpha: sample of angles in radian
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return T:    test statistics of underlying wilcoxon test
   

    References: [Zar2009]_
    """

    m = descriptive.median(alpha, axis=axis)

    d = np.angle(np.exp(1j * m[np.newaxis]) / np.exp(1j * alpha))

    if axis is not None:
        oshape = d.shape[1:]
        d2 = d.reshape((d.shape[0], np.prod(d.shape[1:])))
        T, pval = map(lambda x: np.asarray(x).reshape(oshape), zip(*[stats.wilcoxon(dd) for dd in d2.T]))
    else:
        T, pval = stats.wilcoxon(d)


    return pval, T
    












