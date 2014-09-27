"""
Descriptive statistical functions
"""

import numpy as np
from scipy import stats
import warnings
from PyCircStat.iterators import nd_bootstrap


def mean(alpha, w=None, ci=None, d=None, axis=0):
    """
    Compute mean direction of circular data.

    :param alpha: circular data
    :param w: 	 weightings in case of binned angle data
    :param ci: if not None, the upper and lower 100*ci% confidence interval is returned as well
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is 0
    :return: circular mean if ci=None, or circular mean as well as lower and upper confidence interval limits

    Example:
    >>> import numpy as np
    >>> data = 2*np.pi*np.random.rand(10)
    >>> mu, ci_l, ci_u = mean(data, ci=0.95)
    """

    if w is None:
        w = np.ones_like(alpha)

    if alpha.shape != w.shape:
        raise ValueError('Input dimensions do not match:', alpha.shape)

    r = (w * np.exp(1j * alpha)).sum(axis=axis)

    mu = np.angle(r)

    if ci is None:
        return mu
    else:
        t = mean_ci_limits(alpha, ci=ci, w=w, d=d, axis=axis)
        return mu, mu - t, mu + t


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

    r = np.sum(w * np.exp(1j * alpha), axis=axis)

    # obtain length
    r = np.abs(r) / np.sum(w, axis=axis)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        c = d / 2 / np.sin(d / 2);
        r = c * r

    return r


def mean_ci_limits(alpha, ci=0.95, w=None, d=None, axis=0):
    """
    Computes the confidence limits on the mean for circular data.

    :param alpha: sample of angles in radians
    :param ci: ci-confidence limits are computed, default 0.95
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

    r = np.atleast_1d(resultant_vector_length(alpha, w=w, d=d, axis=axis))
    n = np.atleast_1d(np.sum(w, axis=axis))

    R = n * r
    c2 = stats.chi2.ppf(ci, df=1)

    t = np.NaN * np.empty_like(r)

    if type(n) == np.ndarray:
        idx = (r < .9) & (r > np.sqrt(c2 / 2 / n))
        t[idx] = np.sqrt((2 * n[idx] * (2 * R[idx] ** 2 - n[idx] * c2)) / (4 * n[idx] - c2))  # eq. 26.24

        idx2 = (r >= .9)
        t[idx2] = np.sqrt(n[idx2] ** 2 - (n[idx2] ** 2 - R[idx2] ** 2) * np.exp(c2 / n[idx2]))  # equ. 26.25
        if not np.all(idx | idx2):
            warnings.warn('Requirements for confidence levels not met.')
    else:
        if r < .9 and r > np.sqrt(c2 / 2 / n):
            t = np.sqrt((2 * n * (2 * R ** 2 - n * c2)) / (4 * n - c2))  # eq. 26.24
        elif r >= .9:
            t = np.sqrt(n ** 2 - (n ** 2 - R ** 2) * np.exp(c2 / n))  # equ. 26.25
        else:
            warnings.warn('Requirements for confidence levels not met.')

    return np.arccos(t / R)


def corrcc(alpha1, alpha2, ci=None, axis=0, bootstrap_max_iter=1000):
    """
    Circular correlation coefficient for two circular random variables.

    If a confidence level is specified, confidence limits are bootstrapped. The number of bootstrapping
    iterations is min(number of data points along axis, bootstrap_max_iter).

    :param alpha1: sample of angles in radians
    :param alpha2: sample of angles in radians
    :param axis: correlation coefficient is computed along this dimension (default axis=0)
    :param ci: if not None, confidence level is bootstrapped
    :param bootstrap_max_iter: maximal number of bootstrap iterations
    :return: correlation coefficient if ci=None, otherwise correlation
             coefficient with lower and upper confidence limits

    References: [Jammalamadaka2001]_
    """
    assert alpha1.shape == alpha2.shape, 'Input dimensions do not match.'

    # center data on circular mean
    alpha1, alpha2 = center(alpha1, alpha2, axis=axis)

    # compute correlation coeffcient from p. 176
    num = np.sum(np.sin(alpha1) * np.sin(alpha2), axis=axis)
    den = np.sqrt(np.sum(np.sin(alpha1) ** 2, axis=axis) * np.sum(np.sin(alpha2) ** 2, axis=axis))

    if ci is None:
        return num / den
    else:
        r = [corrcc(a1,a2, ci=None, axis=axis) for a1, a2 in
             nd_bootstrap((alpha1, alpha2),min(alpha1.shape[axis],bootstrap_max_iter), axis=axis)]

        ci_low, ci_high = np.percentile(r, [(1-ci)/2*100, (1+ci)/2*100], axis=0)
        return np.mean(r, axis=0), ci_low, ci_high





def center(*args, **kwargs):
    """
    Centers the data on its circular mean.

    Each non-keyword argument is another data array that is centered.

    :param axis: the mean is computed along this dimension (default axis=0).
                **Must be used as a keyword argument!**
    :return: tuple of centered data arrays

    """
    axis = kwargs.pop('axis', 0)
    reshaper = tuple(slice(None,None) if i != axis else np.newaxis for i in range(len(args[0].shape)))
    if len(args) == 1:
        return args[0] - mean(args[0], axis=axis)
    else:
        return tuple([a - mean(a, axis=axis)[reshaper] for a in args if type(a) == np.ndarray])
