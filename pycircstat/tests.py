"""
Statistical tests
"""
from __future__ import absolute_import, division
import warnings
from nose.tools import nottest

import numpy as np
from scipy import stats
# import warnings
from . import descriptive, swap2zeroaxis
from . import utils
from .distributions import kappa
import pandas as pd
from pycircstat.data import load_kuiper_table
from scipy import special

@swap2zeroaxis(['alpha', 'w'], [0, 1])
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
    # axis = 0
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


@swap2zeroaxis(['alpha', 'w'], [0, 1])
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
        m1[i, ...] = np.sum(
            w * ((alpha > dg_val) & (alpha < np.pi + dg_val)), axis=axis)
        m2[i, ...] = n - m1[i, ...]

    m = np.concatenate((m1, m2), axis=0).min(axis=axis)

    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
    A = np.empty_like(n)
    pval = np.empty_like(n)
    idx50 = (n > 50)

    if np.any(idx50):
        A[idx50] = np.pi * np.sqrt(n[idx50]) / 2 / (n[idx50] - 2 * m[idx50])
        pval[idx50] = np.sqrt(2 * np.pi) / A[idx50] * \
                      np.exp(-np.pi ** 2 / 8 / A[idx50] ** 2)

    if np.any(~idx50):
        pval[~idx50] = 2 ** (1 - n[~idx50]) * (n[~idx50] - \
                                               2 * m[~idx50]) * special.comb(n[~idx50], m[~idx50])

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
        ti = alpha[j + 1, ...] - alpha[j, ...]
        U = U + np.abs(ti - kappa)

    tn = 360 - alpha[-1, ...] + alpha[0, ...]
    U = U + abs(tn - kappa)

    U = .5 * U

    # get critical value from table
    pval, Uc = _critical_value_raospacing(n, U)

    return pval, U, Uc


def _critical_value_raospacing(n, U):
    # Table II from Russel and Levitin, 1995

    alpha_level = np.array([0.001, .01, .05, .10])

    table = np.array([
        4, 247.32, 221.14, 186.45, 168.02,
        5, 245.19, 211.93, 183.44, 168.66,
        6, 236.81, 206.79, 180.65, 166.30,
        7, 229.46, 202.55, 177.83, 165.05,
        8, 224.41, 198.46, 175.68, 163.56,
        9, 219.52, 195.27, 173.68, 162.36,
        10, 215.44, 192.37, 171.98, 161.23,
        11, 211.87, 189.88, 170.45, 160.24,
        12, 208.69, 187.66, 169.09, 159.33,
        13, 205.87, 185.68, 167.87, 158.50,
        14, 203.33, 183.90, 166.76, 157.75,
        15, 201.04, 182.28, 165.75, 157.06,
        16, 198.96, 180.81, 164.83, 156.43,
        17, 197.05, 179.46, 163.98, 155.84,
        18, 195.29, 178.22, 163.20, 155.29,
        19, 193.67, 177.08, 162.47, 154.78,
        20, 192.17, 176.01, 161.79, 154.31,
        21, 190.78, 175.02, 161.16, 153.86,
        22, 189.47, 174.10, 160.56, 153.44,
        23, 188.25, 173.23, 160.01, 153.05,
        24, 187.11, 172.41, 159.48, 152.68,
        25, 186.03, 171.64, 158.99, 152.32,
        26, 185.01, 170.92, 158.52, 151.99,
        27, 184.05, 170.23, 158.07, 151.67,
        28, 183.14, 169.58, 157.65, 151.37,
        29, 182.28, 168.96, 157.25, 151.08,
        30, 181.45, 168.38, 156.87, 150.80,
        35, 177.88, 165.81, 155.19, 149.59,
        40, 174.99, 163.73, 153.82, 148.60,
        45, 172.58, 162.00, 152.68, 147.76,
        50, 170.54, 160.53, 151.70, 147.05,
        75, 163.60, 155.49, 148.34, 144.56,
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
        1000, 140.99, 138.84, 136.94, 135.92])
    table = table.reshape((-1, 5))

    if not hasattr(U, 'shape'):
        U = np.array(U)

    old_shape = U.shape
    U = U.ravel()
    Uc, p = 0 * U, 0 * U

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


@swap2zeroaxis(['alpha', 'w'], [0, 1])
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
    m = descriptive.mean(alpha, w=w, d=d, axis=axis)
    n = np.sum(w, axis=axis)

    # compute Rayleigh's R (equ. 27.1)
    R = n * r

    # compute V and u (equ. 27.5)
    V = R * np.cos(m - mu)
    u = V * np.sqrt(2 / n)
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
        d2 = d.reshape((d.shape[0], int(np.prod(d.shape[1:]))))
        T, pval = map(lambda x: np.asarray(x).reshape(
            oshape), zip(*[stats.wilcoxon(dd) for dd in d2.T]))
    else:
        T, pval = stats.wilcoxon(d)

    return pval, T


@nottest
def watson_williams(*args, **kwargs):
    """
    Parametric Watson-Williams multi-sample test for equal means. Can be
    used as a one-way ANOVA test for circular data.

    H0: the s populations have equal means
    HA: the s populations have unequal means

    Note:
    Use with binned data is only advisable if binning is finer than 10 deg.
    In this case, alpha is assumed to correspond
    to bin centers.

    The Watson-Williams two-sample test assumes underlying von-Mises
    distributrions. All groups are assumed to have a common concentration
    parameter k.

    :param args: number of arrays containing the data; angles in radians
    :param w:    list the same size as the number of args containing the number of
                 incidences for each arg. Must be passed as keyword argument.
    :param axis: the test will be performed along this axis. Must be passed as keyword
                 argument.

    :return pval, table: p-value and pandas dataframe containing the ANOVA table

    """

    axis = kwargs.get('axis', None)
    w = kwargs.get('w', None)

    # argument checking
    if w is not None:
        assert len(w) == len(
            args), "w must have the same length as number of arrays"
        for i, (ww, alpha) in enumerate(zip(w, args)):
            assert ww.shape == alpha.shape, "w[%i] and argument %i must have same shape" % (
                i, i)
    else:
        w = [np.ones_like(a) for a in args]

    if axis is None:
        alpha = list(map(np.ravel, args))
        w = list(map(np.ravel, w))
    else:
        alpha = args

    k = len(args)

    # np.asarray(list())
    ni = list(map(lambda x: np.sum(x, axis=axis), w))
    ri = np.asarray([descriptive.resultant_vector_length(
        a, ww, axis=axis) for a, ww in zip(alpha, w)])

    r = descriptive.resultant_vector_length(
        np.concatenate(
            alpha, axis=axis), np.concatenate(
            w, axis=axis), axis=axis)
    # this must not be the numpy sum since the arrays are to be summed
    n = sum(ni)

    rw = sum([rii * nii / n for rii, nii in zip(ri, ni)])
    kk = kappa(rw[None, ...], axis=0)

    beta = 1 + 3. / (8 * kk)
    A = sum([rii * nii for rii, nii in zip(ri, ni)]) - r * n
    B = n - sum([rii * nii for rii, nii in zip(ri, ni)])

    F = (beta * (n - k) * A / (k - 1) / B).squeeze()
    pval = stats.f.sf(F, k - 1, n - k).squeeze()

    if np.any((n >= 11) & (rw < .45)):
        warnings.warn(
            'Test not applicable. Average resultant vector length < 0.45.')
    elif np.any((n < 11) & (n >= 7) & (rw < .5)):
        warnings.warn(
            'Test not applicable. Average number of samples per population 6 < x < 11 '
            'and average resultant vector length < 0.5.')
    elif np.any((n >= 5) & (n < 7) & (rw < .55)):
        warnings.warn(
            'Test not applicable. Average number of samples per population 4 < x < 7 and '
            'average resultant vector length < 0.55.')
    elif np.any(n < 5):
        warnings.warn(
            'Test not applicable. Average number of samples per population < 5.')

    if np.prod(pval.shape) > 1:
        T = np.zeros_like(pval, dtype=object)
        for idx, p in np.ndenumerate(pval):
            T[idx] = pd.DataFrame({'Source': ['Columns', 'Residual', 'Total'],
                                   'df': [k - 1, n[idx] - k, n[idx] - 1],
                                   'SS': [A[idx], B[idx], A[idx] + B[idx]],
                                   'MS': [A[idx] / (k - 1), B[idx] / (n[idx] - k), np.NaN],
                                   'F': [F[idx], np.NaN, np.NaN],
                                   'p-value': [p, np.NaN, np.NaN]}).set_index('Source')

    else:
        T = pd.DataFrame({'Source': ['Columns', 'Residual', 'Total'],
                          'df': [k - 1, n - k, n - 1],
                          'SS': [A, B, A + B],
                          'MS': [A / (k - 1), B / (n - k), np.NaN],
                          'F': [F, np.NaN, np.NaN],
                          'p-value': [pval, np.NaN, np.NaN]}).set_index('Source')

    return pval, T


@swap2zeroaxis(['alpha1', 'alpha2'], [0, 1])
def kuiper(alpha1, alpha2, res=100, axis=None):
    """
    The Kuiper two-sample test tests whether the two samples differ
    significantly.The difference can be in any property, such as mean
    location and dispersion. It is a circular analogue of the
    Kolmogorov-Smirnov test.

    H0: The two distributions are identical.
    HA: The two distributions are different.

    :param alpha1: fist sample (in radians)
    :param alpha2: second sample (in radians)
    :param res:    resolution at which the cdf is evaluated (default 100)
    :returns: p-value and test statistic
              p-value is the smallest of .10, .05, .02, .01, .005, .002,
              .001, for which the test statistic is still higher
              than the respective critical value. this is due to
              the use of tabulated values. if p>.1, pval is set to 1.

    References: [Batschelet1980]_ p. 112

    """

    if axis is not None:
        assert alpha1.shape[
               1:] == alpha2.shape[
                      1:], "Shapes of alphas not consistent with computation along axis."
    n, m = alpha1.shape[axis], alpha2.shape[axis]

    _, cdf1 = _sample_cdf(alpha1, res, axis=axis)
    _, cdf2 = _sample_cdf(alpha2, res, axis=axis)

    dplus = np.atleast_1d((cdf1 - cdf2).max(axis=axis))
    dplus[dplus < 0] = 0.
    dminus = np.atleast_1d((cdf2 - cdf1).max(axis=axis))
    dminus[dminus < 0] = 0.

    k = n * m * (dplus + dminus)
    mi = np.min([m, n])
    fac = np.sqrt(n * m * (n + m))
    pval = np.asarray([_kuiper_lookup(mi, kk / fac)
                       for kk in k.ravel()]).reshape(k.shape)
    return pval, k


def _kuiper_lookup(n, k):
    ktable = load_kuiper_table()

    alpha = np.asarray([.10, .05, .02, .01, .005, .002, .001])
    nn = ktable[:, 0]

    isin = (nn == n)
    if np.any(isin):
        row = np.where(isin)[0]
    else:
        row = len(nn) - np.sum(n < nn) - 1

        if row == 0:
            raise ValueError('N too small.')
        else:
            warnings.warn(
                'N=%d not found in table, using closest N=%d present.' %
                (n, nn[row]))

    idx = (ktable[row, 1:] < k).squeeze()
    if np.any(idx):
        return alpha[idx].min()
    else:
        return 1.


@swap2zeroaxis(['alpha'], [1])
def _sample_cdf(alpha, resolution=100., axis=None):
    """

    Helper function for circ_kuipertest.
    Evaluates CDF of sample in thetas.

    :param alpha: sample (in radians)
    :param resolution: resolution at which the cdf is evaluated (default 100)
    :param axis: axis along which the cdf is computed
    :returns: points at which cdf is evaluated, cdf values

    """

    if axis is None:
        alpha = alpha.ravel()
        axis = 0
    bins = np.linspace(0, 2 * np.pi, resolution + 1)
    old_shape = alpha.shape
    alpha = alpha % (2 * np.pi)

    alpha = alpha.reshape((alpha.shape[0], int(np.prod(alpha.shape[1:])))).T
    cdf = np.array([np.histogram(a, bins=bins)[0]
                    for a in alpha]).cumsum(axis=1) / float(alpha.shape[1])
    cdf = cdf.T.reshape((len(bins) - 1,) + old_shape[1:])

    return bins[:-1], cdf


@nottest
def cmtest(*args, **kwargs):
    """
    Non parametric multi-sample test for equal medians. Similar to a
    Kruskal-Wallis test for linear data.

    H0: the s populations have equal medians
    HA: the s populations have unequal medians

    :param alpha1: angles in radians
    :param alpha2: angles in radians
    :returns: p-value and test statistic of the common median test


    References: [Fisher1995]_

    """
    axis = kwargs.get('axis', None)
    if axis is None:
        alpha = list(map(np.ravel, args))
    else:
        alpha = args

    s = len(alpha)
    n = [(0 * a + 1).sum(axis=axis) for a in alpha]
    N = sum(n)

    med = descriptive.median(np.concatenate(alpha, axis=axis), axis=axis)
    if axis is not None:
        med = np.expand_dims(med, axis=axis)

    m = [np.sum(descriptive.cdiff(a, med) < 0, axis=axis) for a in alpha]
    if np.any([nn < 10 for nn in n]):
        warnings.warn('Test not applicable. Sample size in at least one group to small.')
    M = sum(m)
    P = (N ** 2. / (M * (N - M))) * sum([mm ** 2. / nn for mm, nn in zip(m, n)]) - N * M / (N - M)
    pval = stats.chi2.sf(P, df=s - 1)
    return pval, P


@nottest
def mtest(alpha, dir, xi=0.05, w=None, d=None, axis=None):
    """
    One-Sample test for the mean angle.

    H0: the population has mean dir.
    HA: the population has not mean dir.

    Note: This is the equvivalent to a one-sample t-test with specified
          mean direction.

    :param alpha: sample of angles in radians
    :param dir: assumed mean direction
    :param w: number of incidences in case of binned angle data
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: test is computed along this axis
    :returns: 0 if H0 can not be rejected, 1 otherwise, mean, confidence interval

    References: [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha, dtype=float)
    else:
        assert alpha.shape == w.shape, "Shape of w and alpha must match"

    dir = np.atleast_1d(dir)

    mu, ci = descriptive.mean(alpha, w=w, d=d, axis=axis, ci=1. - xi)
    t = np.abs(descriptive.cdiff(mu, ci.lower))
    h = np.abs(descriptive.cdiff(mu, dir)) > t

    return h, mu, ci


@nottest
def medtest(alpha, md, axis=None):
    """
    Tests for difference in the median against a fixed value.

    H0: the population has median angle md
    HA: the population has not median angle md

    :param alpha: sample of angles in radians
    :param md:    median to test for
    :param axis:  test is performed along this axis
    :returns:     p-value
    """

    md = np.atleast_1d(md)

    n = alpha.shape[axis] if axis is not None else len(alpha)

    d = descriptive.cdiff(alpha, md)

    n1 = np.atleast_1d(np.sum(d < 0, axis=axis))
    n2 = np.atleast_1d(np.sum(d > 0, axis=axis))

    # compute p-value with binomial test
    n_min = np.array(n1)
    n_min[n1 > n2] = n2[n1 > n2]

    n_max = np.array(n1)
    n_max[n1 < n2] = n2[n1 < n2]
    # TODO: this formula can actually give more than 1, e.g. if n_max == n_min; possibly change that
    return stats.binom.cdf(n_min, n, 0.5) + 1 - stats.binom.cdf(n_max - 1, n, 0.5)


@nottest
def hktest(alpha, idp, idq, inter=True, fn=None):
    if fn is None:
        fn = ['A', 'B']
    p = len(np.unique(idp))
    q = len(np.unique(idq))
    df = pd.DataFrame({fn[0]: idp, fn[1]: idq, 'dependent': alpha})
    n = len(df)
    tr = n * descriptive.resultant_vector_length(df['dependent'])
    kk = kappa(tr / n)

    # both factors
    gr = df.groupby(fn)
    cn = gr.count()
    cr = gr.agg(descriptive.resultant_vector_length) * cn
    cn = cn.unstack(fn[1])
    cr = cr.unstack(fn[1])

    # factor A
    gr = df.groupby(fn[0])
    pn = gr.count()['dependent']
    pr = gr.agg(descriptive.resultant_vector_length)['dependent'] * pn
    pm = gr.agg(descriptive.mean)['dependent']
    # factor B
    gr = df.groupby(fn[1])
    qn = gr.count()['dependent']
    qr = gr.agg(descriptive.resultant_vector_length)['dependent'] * qn
    qm = gr.agg(descriptive.mean)['dependent']

    if kk > 2:  # large kappa
        # effect of factor 1
        eff_1 = sum(pr ** 2 / cn.sum(axis=1)) - tr ** 2 / n
        df_1 = p - 1
        ms_1 = eff_1 / df_1

        # effect of factor 2
        eff_2 = sum(qr ** 2. / cn.sum(axis=0)) - tr ** 2 / n
        df_2 = q - 1
        ms_2 = eff_2 / df_2

        # total effect
        eff_t = n - tr ** 2 / n
        df_t = n - 1
        m = cn.values.mean()

        if inter:
            # correction factor for improved F statistic
            beta = 1 / (1 - 1 / (5 * kk) - 1 / (10 * (kk ** 2)))
            # residual effects
            eff_r = n - (cr**2./cn).values.sum()
            df_r = p*q*(m-1)
            ms_r = eff_r / df_r

            # interaction effects
            eff_i = (cr**2./cn).values.sum() - sum(qr**2./qn) - sum(pr**2./pn) + tr**2/n
            df_i = (p-1)*(q-1)
            ms_i = eff_i/df_i;

            # interaction test statistic
            FI = ms_i / ms_r
            pI = 1 - stats.f.cdf(FI,df_i,df_r)
        else:
            # residual effect
            eff_r = n - sum(qr**2./qn)- sum(pr**2./pn) + tr**2/n
            df_r = (p-1)*(q-1)
            ms_r = eff_r / df_r

            # interaction effects
            eff_i = None
            df_i = None
            ms_i = None

            # interaction test statistic
            FI = None
            pI = np.NaN
            beta = 1


        F1 = beta * ms_1 / ms_r
        p1 = 1 - stats.f.cdf(F1,df_1,df_r)

        F2 = beta * ms_2 / ms_r
        p2 = 1 - stats.f.cdf(F2,df_2,df_r)

    else: #small kappa
        # correction factor
        # special.iv is Modified Bessel function of the first kind of real order
        rr = special.iv(1,kk) / special.iv(0,kk)
        f = 2/(1-rr**2)

        chi1 = f * (sum(pr**2./pn)- tr**2/n)
        df_1 = 2*(p-1)
        p1 = 1 - stats.chi2.cdf(chi1, df=df_1)

        chi2 = f * (sum(qr**2./qn)- tr**2/n)
        df_2 = 2*(q-1)
        p2 = 1 - stats.chi2.cdf(chi2, df=df_2)

        chiI = f * ( (cr**2./cn).values.sum() - sum(pr**2./pn) - sum(qr**2./qn) + tr**2/n)
        df_i = (p-1) * (q-1)
        pI = stats.chi2.sf(chiI, df=df_i)



    pval = (p1.squeeze(), p2.squeeze(), pI.squeeze())

    if kk>2:
        table = pd.DataFrame({
            'Source': fn + ['Interaction', 'Residual', 'Total'],
            'DoF': [df_1, df_2, df_i, df_r, df_t],
            'SS': [eff_1, eff_2, eff_i, eff_r, eff_t],
            'MS': [ms_1, ms_2, ms_i, ms_r, np.NaN],
            'F': [F1.squeeze(), F2.squeeze(), FI, np.NaN, np.NaN],
            'p': list(pval) + [np.NaN, np.NaN]
        })
        table = table.set_index('Source')
    else:
        table = pd.DataFrame({
            'Source': fn + ['Interaction'],
            'DoF': [df_1, df_2, df_i],
            'chi2': [chi1.squeeze(), chi2.squeeze(), chiI.squeeze()],
            'p': pval
        })
        table = table.set_index('Source')

    return pval, table
