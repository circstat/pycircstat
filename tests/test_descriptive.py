from __future__ import absolute_import

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal

import PyCircStat

def test_mean_constant_data():
    data = np.ones(10)

    # We cannot use `assert_equal`, due to numerical rounding errors.
    assert_allclose(PyCircStat.mean(data), 1.0)

def test_resultant_vector_length():
    data = np.ones(10)
    assert_allclose(PyCircStat.resultant_vector_length(data), 1.0)

def test_resultant_vector_length_axis():
    data = np.ones((10,2))
    assert_allclose(PyCircStat.resultant_vector_length(data, axis=1), np.ones(10))


def test_mean_ci_limits():
    data = np.array([
            [0.58429,   0.88333],
            [1.14892 ,  2.22854],
            [2.87128 ,  3.06369],
            [1.07677,   1.49836],
            [2.96969,   1.51748],
        ])
    out1 = np.array([0.76976, 0.50149])
    out2 = np.array([0.17081, 0.72910, 0.10911, 0.24385, 0.95426])
    assert_allclose(PyCircStat.mean_ci_limits(data, ci=0.8, axis=0), out1, rtol=1e-4)
    assert_allclose(PyCircStat.mean_ci_limits(data, ci=0.8, axis=1), out2, rtol=1e-4)

def test_mean_ci_2d():
    data = np.array([
            [0.58429,   0.88333],
            [1.14892 ,  2.22854],
            [2.87128 ,  3.06369],
            [1.07677,   1.49836],
            [2.96969,   1.51748],
        ])
    muplus = np.array([np.NaN, 2.7003])
    muminus = np.array([np.NaN, 0.89931])
    mu = np.array([1.6537, 1.7998])

    mu_tmp, muplus_tmp, muminus_tmp = PyCircStat.mean(data, ci=0.95, axis=0)
    assert_allclose(muplus, muplus_tmp, rtol=1e-4)
    assert_allclose(muminus, muminus_tmp, rtol=1e-4)
    assert_allclose(mu, mu_tmp, rtol=1e-4)

def test_mean_ci_1d():
    data = np.array([0.88333, 2.22854, 3.06369, 1.49836, 1.51748])
    muplus = 2.7003
    muminus = 0.89931
    mu = 1.7998

    mu_tmp, muplus_tmp, muminus_tmp = PyCircStat.mean(data, ci=0.95)
    assert_allclose(muplus, muplus_tmp, rtol=1e-4)
    assert_allclose(muminus, muminus_tmp, rtol=1e-4)
    assert_allclose(mu, mu_tmp, rtol=1e-4)


