"""
Descriptive statistical functions
"""
import itertools

import numpy as np
from scipy import stats
import warnings
from PyCircStat import CI
from PyCircStat.iterators import nd_bootstrap
from PyCircStat.decorators import mod2pi


class bootstrap:
    """
    Decorator to implement bootstrapping. It looks for the arguments ci, axis, and bootstrap_iter
    to determine the proper parameters for bootstrapping. The argument scale determines whether
    the percentile is taken on a circular scale or on a linear scale.

    :param no_bootstrap: the number of arguments that are bootstrapped (e.g. for correlation it would be two,
                        for median it would be one)
    :param scale: linear or ciruclar scale (default is 'linear')
    """

    def __init__(self, no_bootstrap, scale='linear'):
        self.no_boostrap = no_bootstrap
        self.scale = scale

    def _get_var(self, f, what, default, args, kwargs, remove=False):
        varnames = f.__code__\
            .co_varnames

        if what in varnames:
            what_idx = varnames.index(what)
        else:
            raise ValueError('Function %s does not have variable %s.' % (f.__name__, what))

        if len(args) >= what_idx + 1:
            val = args[what_idx]
            if remove:
                args[what_idx] = default
        elif what in kwargs:
            if remove:
                val = kwargs.pop(what, default)
            else:
                val = kwargs[what]
        else:
            val = default

        return val


    def __call__(self, f):

        def wrapped_f(*args, **kwargs):
            ci = self._get_var(f, 'ci', None, args, kwargs, remove=True)
            bootstrap_iter = self._get_var(f, 'bootstrap_iter', None, args, kwargs, remove=True)
            axis = self._get_var(f, 'axis', 0, args, kwargs)  # TODO: change that to None if decided

            alpha = args[:self.no_boostrap]
            args0 = args[self.no_boostrap:]

            if bootstrap_iter is None:
                bootstrap_iter = alpha[0].shape[axis]

            r0 = f(*(alpha + args0), **kwargs)
            if ci is not None:
                r = np.asarray([f(*(a + args0), **kwargs) for a in nd_bootstrap(alpha, bootstrap_iter, axis=axis,
                                                                                strip_tuple_if_one=False)])

                if self.scale == 'linear':
                    ci_low, ci_high = np.percentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100], axis=0)
                elif self.scale == 'circular':
                    ci_low, ci_high = percentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100],
                                                 q0=(r0 + np.pi) % (2 * np.pi), axis=0)
                else:
                    raise ValueError('Scale %s not known!' % (self.scale, ))
                return r0, CI(ci_low, ci_high)
            else:
                return r0


        return wrapped_f


@bootstrap(1, 'circular')
def median(alpha, axis=0, ci=None, bootstrap_iter=None):
    """
    Computes the median direction for circular data.

    :param alpha: sample of angles in radians
    :param axis:  compute along this dimension, default is 0
    :param ci:    if not None, the upper and lower 100*ci% confidence interval is returned as well
    :param bootstrap_iter: number of bootstrap iterations (number of samples if None)
    :return: median direction
    """
    dims = [range(alpha.shape[i]) for i in range(len(alpha.shape))]
    dims[axis] = [slice(None)]

    med = np.empty(alpha.shape[:axis] + alpha.shape[axis + 1:])
    n = alpha.shape[axis]
    is_odd = (n % 2 == 1)
    for idx in itertools.product(*dims):
        out_idx = idx[:axis] + idx[axis + 1:]

        beta = alpha[idx] % (2 * np.pi)

        dd = pairwise_cdiff(beta)
        m1 = np.sum(dd >= 0, 0)
        m2 = np.sum(dd <= 0, 0)
        dm = np.abs(m1 - m2)

        if is_odd:
            min_idx = np.argmin(dm)
            m = dm[min_idx]
        else:
            m = np.min(dm)
            min_idx = np.argsort(dm)[:2]

        if m > 1:
            warnings.warn('Ties detected in median computation')

        md = mean(beta[min_idx])
        if np.abs(cdiff(mean(beta), md)) > np.abs(cdiff(mean(beta), md + np.pi)):
            md = (md + np.pi) % (2 * np.pi)

        med[out_idx] = md

    return med


def cdiff(alpha, beta):
    """
    Difference between pairs :math:`x_i-y_i` around the circle computed efficiently.

    :param alpha:  sample of circular random variable
    :param beta:   sample of circular random variable
    :return: distance between the pairs
    """
    assert alpha.shape == beta.shape, 'Input dimensions do not match!'

    return np.angle(np.exp(1j * alpha) / np.exp(1j * beta))


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
    reshaper_alpha = len(alpha.shape) * (slice(None, None),) + len(beta.shape) * (np.newaxis,)
    reshaper_beta = len(alpha.shape) * (np.newaxis,) + len(beta.shape) * (slice(None, None),)

    return np.angle(np.exp(1j * alpha[reshaper_alpha]) / np.exp(1j * beta[reshaper_beta]))


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
    >>> mu, (ci_l, ci_u) = mean(data, ci=0.95)

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


@bootstrap(1, 'linear')
def resultant_vector_length(alpha, w=None, d=None, axis=0, axial_correction=1, ci=None, bootstrap_iter=None):
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
    :param bootstrap_iter: number of bootstrap iterations (number of samples if None)
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    cmean = _complex_mean(alpha, w=w, axis=axis, axial_correction=axial_correction)

    # obtain length
    r = np.abs(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    print(r)
    return r


def _complex_mean(alpha, w=None, axis=0, axial_correction=1):
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)
    assert w.shape == alpha.shape, "Dimensions of data and w do not match!"

    return (w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) / np.sum(w, axis=axis)


@bootstrap(2, 'linear')
def corrcc(alpha1, alpha2, ci=None, axis=0, bootstrap_iter=None):
    """
    Circular correlation coefficient for two circular random variables.

    If a confidence level is specified, confidence limits are bootstrapped. The number of bootstrapping
    iterations is min(number of data points along axis, bootstrap_max_iter).

    :param alpha1: sample of angles in radians
    :param alpha2: sample of angles in radians
    :param axis: correlation coefficient is computed along this dimension (default axis=0)
    :param ci: if not None, confidence level is bootstrapped
    :param bootstrap_iter: number of bootstrap iterations (number of samples if None)
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
    return num / den


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

@mod2pi
@bootstrap(1, 'circular')
def percentile(alpha, q, q0, axis=None, ci=None, bootstrap_iter=None):
    """
    Computes circular percentiles

    :param alpha: array with circular samples
    :param q: percentiles in [0,100] (single number or iterable)
    :param q0: value of the 0 percentile
    :param axis: percentiles will be computed along this axis. If None percentiles will be computed
                over the entire array
    :param ci: if not None, confidence level is bootstrapped
    :param bootstrap_iter: number of bootstrap iterations (number of samples if None)

    :return: percentiles

    """
    if axis is None:
        alpha = (alpha.ravel() - q0) % (2 * np.pi)
    else:
        if len(q0.shape) == len(alpha.shape) - 1:
            reshaper = tuple(slice(None, None) if i != axis else np.newaxis for i in range(len(alpha.shape)))
            q0 = q0[reshaper]
        elif not len(q0.shape) == len(alpha.shape):
            raise ValueError("Dimensions of start and alpha are inconsistent!")

        alpha = (alpha - q0) % (2 * np.pi)

    ret = []
    if axis is not None:
        selector = tuple(slice(None) if i != axis else 0 for i in range(len(alpha.shape)))
        q0 = q0[selector]

    for qq in np.atleast_1d(q):
        ret.append(np.percentile(alpha, qq, axis=axis) + q0)

    if not hasattr(q, "__iter__"):  # if q is not some sort of list, array, etc
        return np.asarray(ret).squeeze()
    else:
        return np.asarray(ret)


@bootstrap(1, 'linear')
def var(alpha, w=None, d=None, axis=0, ci=None, bootstrap_iter=None):
    """
    Computes circular variance for circular data (equ. 26.17/18, Zar).
    :param alpha: sample of angles in radian
    :param w: 	  number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is 0
    :param bootstrap_iter: if not None, confidence level is bootstrapped
    :param ci:   number of bootstrap iterations (number of samples if None)
    :return:      circular variance 1 - resultant vector length

    References: [Zar2009]_
    """
    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    r = resultant_vector_length(alpha, w=w, d=d, axis=axis)


    return 1 - r

@bootstrap(1, 'linear')
def std(alpha, w=None, d=None, axis=0, ci=None, bootstrap_iter=None):
    """
    Computes circular standard deviation for circular data.
    :param alpha: sample of angles in radian
    :param w: 	  number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is 0
    :param bootstrap_iter: if not None, confidence level is bootstrapped
    :param ci:   number of bootstrap iterations (number of samples if None)
    :return:      circular variance 1 - resultant vector length

    References: [Zar2009]_
    """
    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    r = resultant_vector_length(alpha, w=w, d=d, axis=axis)


    return np.sqrt(-2*np.log(r))


@bootstrap(1, 'linear')
def avar(alpha, w=None, d=None, axis=0, ci=None, bootstrap_iter=None):
    """
    Computes angular variance for circular data (equ. 26.17/18, Zar).

    :param alpha: sample of angles in radian
    :param w: 	  number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is 0
    :param bootstrap_iter: if not None, confidence level is bootstrapped
    :param ci:   number of bootstrap iterations (number of samples if None)
    :return:      2 * circular variance

    References: [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha)

    return 2 * var(alpha, w=w, d=d, axis=axis, ci=None)

@bootstrap(1, 'linear')
def astd(alpha, w=None, d=None, axis=0, ci=None, bootstrap_iter=None):
    """
    Computes angular standard deviation for circular data.

    :param alpha: sample of angles in radian
    :param w: 	  number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is 0
    :param bootstrap_iter: if not None, confidence level is bootstrapped
    :param ci:   number of bootstrap iterations (number of samples if None)
    :return:      2 * circular variance

    References: [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha)

    return np.sqrt(avar(alpha, w=w, d=d, axis=axis, ci=None))