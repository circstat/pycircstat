"""
Descriptive statistical functions
"""
import itertools

import numpy as np
from scipy import stats
import warnings
from PyCircStat import CI
from PyCircStat.decorators import mod2pi
from PyCircStat.iterators import nd_bootstrap


def median(alpha, axis=0, ci=None, bootstrap_max_iter=1000):
    """
    Computes the median direction for circular data.

    :param alpha: sample of angles in radians
    :param axis:  compute along this dimension, default is 0
    :param ci:    if not None, the upper and lower 100*ci% confidence interval is returned as well
    :param bootstrap_max_iter: maximal number of bootstrap iterations
    :return: median direction
    """
    if ci is None:
        dims = [range(alpha.shape[i]) for i in range(len(alpha.shape))]
        dims[axis] = [slice(None)]

        med = np.empty(alpha.shape[:axis] + alpha.shape[axis+1:])
        n = alpha.shape[axis]
        is_odd = (n % 2 == 1)
        for idx in itertools.product(*dims):
            out_idx = idx[:axis] + idx[axis+1:]

            beta = alpha[idx] % (2*np.pi)

            dd = pairwise_cdiff(beta)
            m1 = np.sum(dd>=0,0)
            m2 = np.sum(dd<=0,0)
            dm = np.abs(m1-m2)

            if is_odd:
                min_idx = np.argmin(dm)
                m = dm[min_idx]
            else:
                m = np.min(dm)
                min_idx = np.argsort(dm)[:2]

            if m > 1:
                warnings.warn('Ties detected in median computation')

            md = mean(beta[min_idx])
            if np.abs(cdiff(mean(beta),md)) > np.abs(cdiff(mean(beta),md+np.pi)):
                md = (md+np.pi) % (2*np.pi)

            med[out_idx] = md

        return med
    else:
        warnings.warn('Median bootstrapping uses linear percentile function. ')
        r = [median(a, ci=None, axis=axis) for a in nd_bootstrap((alpha,) , min(alpha.shape[axis], bootstrap_max_iter), axis=axis)]

        ci_low, ci_high = np.percentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100], axis=0) # TODO: write circular percentile (opposite)
        return median(alpha, ci=None, axis=axis), CI(ci_low, ci_high)

def cdiff(alpha, beta):
    """
    Difference between pairs :math:`x_i-y_i` around the circle computed efficiently.

    :param alpha:  sample of circular random variable
    :param beta:   sample of circular random variable
    :return: distance between the pairs
    """
    assert alpha.shape == beta.shape, 'Input dimensions do not match!'

    return np.angle(np.exp(1j*alpha) / np.exp(1j*beta))

def pairwise_cdiff(alpha, beta=None):
    """
    All pairwise difference :math:`x_i-y_j` around the circle computed efficiently.

    :param alpha: sample of circular random variable
    :param beta: sample of circular random variable
    :return: array with pairwise differences

    References: [Zar2009]_, p. 651
    """
    if beta is None:
        beta = alpha

    # advanced slicing and broadcasting to make pairwise distance work between arbitrary nd arrays
    reshaper_alpha = len(alpha.shape)*(slice(None,None),) + len(beta.shape)*(np.newaxis,)
    reshaper_beta = len(alpha.shape)*(np.newaxis,) + len(beta.shape)*(slice(None,None),)

    return np.angle(np.exp(1j*alpha[reshaper_alpha]) / np.exp(1j*beta[reshaper_beta]))


def mean(alpha, w=None, ci=None, d=None, axis=0, axial_correction=1):
    """
    Compute mean direction of circular data.

    :param alpha: circular data
    :param w: 	 weightings in case of binned angle data
    :param ci: if not None, the upper and lower 100*ci% confidence interval is returned as well
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is 0
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :return: circular mean if ci=None, or circular mean as well as lower and upper confidence interval limits

    Example:

    >>> import numpy as np
    >>> data = 2*np.pi*np.random.rand(10)
    >>> mu, ci_l, ci_u = mean(data, ci=0.95)

    """

    cmean = _complex_mean(alpha, w=w, axis=axis, axial_correction=axial_correction)

    mu = np.angle(cmean) / axial_correction

    if ci is None:
        return mu
    else:
        if axial_correction > 1:  # TODO: implement CI for axial correction
            warnings.warn("Axial correction ignored for confidence intervals.")
        t = mean_ci_limits(alpha, ci=ci, w=w, d=d, axis=axis)
        return mu, CI(mu - t, mu + t)


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

    idx = (r < .9) & (r > np.sqrt(c2 / 2 / n))
    t[idx] = np.sqrt((2 * n[idx] * (2 * R[idx] ** 2 - n[idx] * c2)) / (4 * n[idx] - c2))  # eq. 26.24

    idx2 = (r >= .9)
    t[idx2] = np.sqrt(n[idx2] ** 2 - (n[idx2] ** 2 - R[idx2] ** 2) * np.exp(c2 / n[idx2]))  # equ. 26.25
    if not np.all(idx | idx2):
        warnings.warn('Requirements for confidence levels not met.')

    return np.arccos(t / R)


def resultant_vector_length(alpha, w=None, d=None, axis=0, axial_correction=1, ci=None, bootstrap_max_iter=1000):
    """
    Computes mean resultant vector length for circular data.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping, default 0.95
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is 0
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_max_iter: maximal number of bootstrap iterations
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    if ci is None:
        cmean = _complex_mean(alpha, w=w, axis=axis, axial_correction=axial_correction)

        # obtain length
        r = np.abs(cmean)

        # for data with known spacing, apply correction factor to correct for bias
        # in the estimation of r (see Zar, p. 601, equ. 26.16)
        if d is not None:
            if axial_correction > 1:
                warnings.warn("Axial correction ignored for bias correction.")
            r *= d / 2 / np.sin(d / 2)
        return r
    else:
        r = [resultant_vector_length(a, w=w, axial_correction=axial_correction, d=d, ci=None, axis=axis) for a in
             nd_bootstrap((alpha,), min(alpha.shape[axis], bootstrap_max_iter), axis=axis)]

        ci_low, ci_high = np.percentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100], axis=0)
        return resultant_vector_length(alpha, w=w, axial_correction=axial_correction, d=d, ci=None, axis=axis), CI(ci_low, ci_high)


def _complex_mean(alpha, w=None, axis=0, axial_correction=1):
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)
    assert w.shape == alpha.shape, "Dimensions of data and w do not match!"

    return (w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) / np.sum(w, axis=axis)


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

    if ci is None:
        # center data on circular mean
        alpha1, alpha2 = center(alpha1, alpha2, axis=axis)

        # compute correlation coeffcient from p. 176
        num = np.sum(np.sin(alpha1) * np.sin(alpha2), axis=axis)
        den = np.sqrt(np.sum(np.sin(alpha1) ** 2, axis=axis) * np.sum(np.sin(alpha2) ** 2, axis=axis))
        return num / den
    else:
        r = [corrcc(a1, a2, ci=None, axis=axis) for a1, a2 in
             nd_bootstrap((alpha1, alpha2), min(alpha1.shape[axis], bootstrap_max_iter), axis=axis)] # TODO: new bootstrap iteration number

        ci_low, ci_high = np.percentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100], axis=0)
        return corrcc(alpha1, alpha2, ci=None, axis=axis), CI(ci_low, ci_high)

@mod2pi
def center(*args, **kwargs):
    """
    Centers the data on its circular mean.

    Each non-keyword argument is another data array that is centered.

    :param axis: the mean is computed along this dimension (default axis=0).
                **Must be used as a keyword argument!**
    :return: tuple of centered data arrays

    """
    axis = kwargs.pop('axis', 0)

    reshaper = tuple(slice(None, None) if i != axis else np.newaxis for i in range(len(args[0].shape)))
    if len(args) == 1:
        return args[0] - mean(args[0], axis=axis)
    else:
        return tuple([a - mean(a, axis=axis)[reshaper] for a in args if type(a) == np.ndarray])
