from __future__ import absolute_import

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal

import PyCircStat

def test_median():
    alpha = np.array([
        [3.73153000, 1.63904879, 4.03175622, 3.90422402, 4.61029613, 4.04117818, 5.79313473, 5.50863002, 5.81530225, 2.44973903],
        [2.12868554, 0.09073566, 0.05581025, 5.10673712, 1.68712454, 3.72915575, 4.45439608, 4.70694685, 3.58470730, 2.49742028]
    ])
    m0 = np.array([2.93010777, 0.86489223, -1.09780942, -1.77770474, -3.13447497, -2.39801834, -1.15941990, -1.17539688, -1.58318053, 2.47357966])
    m1 = np.array([-2.24671810, -1.24910966])
    m11 = np.array([-2.24200713, -1.82878923])

    assert_allclose(PyCircStat.median(alpha, axis=1), m1)
    assert_allclose(PyCircStat.median(alpha[:,:-1], axis=1), m11)
    assert_allclose(PyCircStat.median(alpha, axis=0), m0)

def test_circular_distance():
    a = np.array([4.85065953, 0.79063862, 1.35698570])
    assert_allclose(PyCircStat.cdiff(a,a), np.zeros_like(a))

def test_pairwise_circular_distance():
    a = np.array([4.85065953, 0.79063862, 1.35698570])
    b = np.array([5.77091494, 2.02426471])
    ret = np.array([
        [-0.92025541, 2.82639482, ],
        [1.30290899, -1.23362610, ],
        [1.86925607, -0.66727901, ]
    ])
    assert_allclose(PyCircStat.pairwise_cdiff(a,b), ret)

def test_mean_constant_data():
    data = np.ones(10)

    # We cannot use `assert_equal`, due to numerical rounding errors.
    assert_allclose(PyCircStat.mean(data), 1.0)

def test_mean():
    data = np.array([1.80044838, 2.02938314, 1.03534016, 4.84225057, 1.54256458, 5.19290675, 2.18474784,
                      4.77054777, 1.51736933, 0.72727580])

    # We cannot use `assert_equal`, due to numerical rounding errors.
    assert_allclose(PyCircStat.mean(data), 1.35173983)

def test_mean_axial():
    data = np.array([1.80044838, 2.02938314, 1.03534016, 4.84225057, 1.54256458, 5.19290675, 2.18474784,
                      4.77054777, 1.51736933, 0.72727580])

    # We cannot use `assert_equal`, due to numerical rounding errors.
    assert_allclose(PyCircStat.mean(data, axial_correction=3), 0.95902619)

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

    mu_tmp, (muminus_tmp, muplus_tmp) = PyCircStat.mean(data, ci=0.95, axis=0)
    assert_allclose(muplus, muplus_tmp, rtol=1e-4)
    assert_allclose(muminus, muminus_tmp, rtol=1e-4)
    assert_allclose(mu, mu_tmp, rtol=1e-4)

def test_mean_ci_1d():
    data = np.array([0.88333, 2.22854, 3.06369, 1.49836, 1.51748])
    muplus = 2.7003
    muminus = 0.89931
    mu = 1.7998

    mu_tmp, (muminus_tmp, muplus_tmp) = PyCircStat.mean(data, ci=0.95)
    assert_allclose(muplus, muplus_tmp, rtol=1e-4)
    assert_allclose(muminus, muminus_tmp, rtol=1e-4)
    assert_allclose(mu, mu_tmp, rtol=1e-4)

def test_center():
    data = np.random.rand(1000)*2*np.pi
    assert_allclose(PyCircStat.mean(PyCircStat.center(data)), 0., rtol=1e-3, atol=1e-3)

def test_corrcc():
    data1 = np.random.rand(50000)*2*np.pi
    data2 = np.random.rand(50000)*2*np.pi
    assert_allclose(PyCircStat.corrcc(data1, data2), 0., rtol=1e-2, atol=1e-2)

def test_corrcc_ci():
    data1 = np.random.rand(200)*2*np.pi
    data2 = np.asarray(data1)
    exp = (1., PyCircStat.CI(1.,1.))
    assert_equal(PyCircStat.corrcc(data1, data2, ci=0.95), exp)

def test_corrcc_ci_2d():
    data1 = np.random.rand(2,200)*np.pi
    data2 = np.asarray(data1)

    out1, (out2, out3) = PyCircStat.corrcc(data1, data2, ci=0.95, axis=1)
    exp1, (exp2, exp3) = (np.ones(2), PyCircStat.CI(np.ones(2), np.ones(2)))
    assert_allclose(out1,exp1)
    assert_allclose(out2,exp2)
    assert_allclose(out3,exp3)

