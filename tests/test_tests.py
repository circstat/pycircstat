from __future__ import absolute_import
import warnings

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_true

import pycircstat
from pycircstat.tests import _sample_cdf


def test_rayleigh():
    data = np.array([
        -0.94904375, 0.26575165, -0.03226759, 1.98389239, 2.0084918,
        0.56965871, -0.19199522, -0.33080002, -0.03141245, 1.18560637,
        4.31138236, 4.84776075, 2.10336819, 3.45465972, 4.64572697,
        5.80976322, 6.27160926, 0.67979652, 1.33776004, 0.58059308])
    p, z = pycircstat.tests.rayleigh(data)
    assert_allclose(0.0400878, p, atol=0.001, rtol=0.001)
    assert_allclose(3.1678, z, atol=0.001, rtol=0.001)


def test_rayleightest2():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, z = pycircstat.tests.rayleigh(data, axis=0)
    assert_true(p.shape == (20, 5))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            p2, z2 = pycircstat.tests.rayleigh(data[:, i, j])
            assert_allclose(p[i, j], p2, atol=0.001, rtol=0.001)
            assert_allclose(z[i, j], z2, atol=0.001, rtol=0.001)


def test_rayleightest3():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, z = pycircstat.tests.rayleigh(data, axis=1)
    assert_true(p.shape == (10, 5))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            p2, z2 = pycircstat.tests.rayleigh(data[i, :, j])
            assert_allclose(p[i, j], p2, atol=0.001, rtol=0.001)
            assert_allclose(z[i, j], z2, atol=0.001, rtol=0.001)


def test_omnibus():
    data = np.array([
        -1.78277804, 0.20180845, -1.51291097, 0.57329272, 0.72195959,
        1.59947271, 1.4666837, -0.36532379, 1.4455209, 0.77365236,
        5.8678466, 2.58327349, 0.25429634, 1.74902778, 4.13215085,
        0.20612467, 1.38484181, 1.72546928, 3.33570062, 3.96191276])
    p, m = pycircstat.tests.omnibus(data)
    assert_allclose(0.295715, p, atol=0.001, rtol=0.001)
    assert_equal(5, m)


def test_omnibus2():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, m = pycircstat.tests.omnibus(data, axis=0)
    assert_true(p.shape == (20, 5))
    assert_true(m.shape == (20, 5))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            p2, m2 = pycircstat.tests.omnibus(data[:, i, j])
            assert_allclose(p[i, j], p2, atol=0.001, rtol=0.001)
            assert_allclose(m[i, j], m2, atol=0.001, rtol=0.001)


def test_omnibus3():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, m = pycircstat.tests.omnibus(data, axis=1)
    assert_true(p.shape == (10, 5))
    assert_true(m.shape == (10, 5))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            p2, m2 = pycircstat.tests.omnibus(data[i, :, j])
            assert_allclose(p[i, j], p2, atol=0.001, rtol=0.001)
            assert_allclose(m[i, j], m2, atol=0.001, rtol=0.001)


def test_raospacing():
    data = np.array([
        -1.78277804, 0.20180845, -1.51291097, 0.57329272, 0.72195959,
        1.59947271, 1.4666837, -0.36532379, 1.4455209, 0.77365236,
        5.8678466, 2.58327349, 0.25429634, 1.74902778, 4.13215085,
        0.20612467, 1.38484181, 1.72546928, 3.33570062, 3.96191276])
    p, U, Uc = pycircstat.tests.raospacing(data)
    assert_allclose(0.001, p, atol=0.0001, rtol=0.001)
    assert_allclose(233.7789, U, atol=0.001, rtol=0.001)
    assert_allclose(192.17, Uc, atol=0.001, rtol=0.001)


def test_raospacing2():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, U, Uc = pycircstat.tests.raospacing(data, axis=0)
    assert_true(p.shape == (20, 5))
    assert_true(U.shape == (20, 5))
    assert_true(Uc.shape == (20, 5))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            p2, U2, Uc2 = pycircstat.tests.raospacing(data[:, i, j])
            assert_allclose(p[i, j], p2, atol=0.001, rtol=0.001)
            assert_allclose(U[i, j], U2, atol=0.001, rtol=0.001)
            assert_allclose(Uc[i, j], Uc2, atol=0.001, rtol=0.001)


def test_raospacing3():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, U, Uc = pycircstat.tests.raospacing(data, axis=1)
    assert_true(p.shape == (10, 5))
    assert_true(U.shape == (10, 5))
    assert_true(Uc.shape == (10, 5))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            p2, U2, Uc2 = pycircstat.tests.raospacing(data[i, :, j])
            assert_allclose(p[i, j], p2, atol=0.001, rtol=0.001)
            assert_allclose(U[i, j], U2, atol=0.001, rtol=0.001)
            assert_allclose(Uc[i, j], Uc2, atol=0.001, rtol=0.001)


def test_vtest():
    data = np.array([
        -1.78277804, 0.20180845, -1.51291097, 0.57329272, 0.72195959,
        1.59947271, 1.4666837, -0.36532379, 1.4455209, 0.77365236,
        5.8678466, 2.58327349, 0.25429634, 1.74902778, 4.13215085,
        0.20612467, 1.38484181, 1.72546928, 3.33570062, 3.96191276])
    p, V = pycircstat.tests.vtest(data, 0)
    assert_allclose(0.1074, p, atol=0.001, rtol=0.001)
    assert_allclose(3.9230, V, atol=0.001, rtol=0.001)


def test_vtest2():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, V = pycircstat.tests.vtest(data, 0, axis=0)
    assert_true(p.shape == (20, 5))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            p2, V2 = pycircstat.tests.vtest(data[:, i, j], 0)
            assert_allclose(p[i, j], p2, atol=0.001, rtol=0.001)
            assert_allclose(V[i, j], V2, atol=0.001, rtol=0.001)


def test_vtest3():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, V = pycircstat.tests.vtest(data, 0, axis=1)
    assert_true(p.shape == (10, 5))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            p2, V2 = pycircstat.tests.vtest(data[i, :, j], 0)
            assert_allclose(p[i, j], p2, atol=0.001, rtol=0.001)
            assert_allclose(V[i, j], V2, atol=0.001, rtol=0.001)


def test_symtest():
    data = np.array([
        -1.78277804, 0.20180845, -1.51291097, 0.57329272, 0.72195959,
        1.59947271, 1.4666837, -0.36532379, 1.4455209, 0.77365236,
        5.8678466, 2.58327349, 0.25429634, 1.74902778, 4.13215085,
        0.20612467, 1.38484181, 1.72546928, 3.33570062, 3.96191276])
    p, T = pycircstat.tests.symtest(data)
    assert_allclose(0.295877, p, atol=0.001, rtol=0.001)


def test_symtest2():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, T = pycircstat.tests.symtest(data, axis=0)
    assert_true(p.shape == (20, 5))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            p2, T2 = pycircstat.tests.symtest(data[:, i, j])
            assert_equal(p[i, j], p2)
            assert_equal(T[i, j], T2)


def test_symtest3():
    data = np.random.rand(10, 20, 5) * np.pi * 2.
    p, T = pycircstat.tests.symtest(data, axis=1)
    assert_true(p.shape == (10, 5))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            p2, T2 = pycircstat.tests.symtest(data[i, :, j])
            assert_equal(p[i, j], p2)
            assert_equal(T[i, j], T2)


def test_watson_williams():
    dat1 = np.radians([135., 145, 125, 140, 165, 170])
    dat2 = np.radians([150, 130, 175, 190, 180, 220])
    dat3 = np.radians([140, 165, 185, 180, 125, 175, 140])
    p, T = pycircstat.watson_williams(dat1, dat2, dat3)
    assert_allclose(p, 0.1870637, atol=0.001, rtol=0.001)


def test_watson_williams_nd():
    dat1 = np.tile(np.radians([135., 145, 125, 140, 165, 170]), (3, 4, 1))
    dat2 = np.tile(np.radians([150, 130, 175, 190, 180, 220]), (3, 4, 1))
    dat3 = np.tile(np.radians([140, 165, 185, 180, 125, 175, 140]), (3, 4, 1))
    p, T = pycircstat.watson_williams(dat1, dat2, dat3, axis=2)
    assert_true(
        p.shape == (
            3,
            4),
        "return pvalue array does not have right shape")
    assert_allclose(p, 0.1870637, atol=0.0001, rtol=0.0001)


def test_sample_cdf():
    alpha = np.asarray([3.427109860970,
                        0.649035328217,
                        0.478105054362,
                        3.585976113724,
                        2.436564305605,
                        2.397389764489,
                        0.223860727758,
                        3.810441709350,
                        2.194680923763,
                        5.423136274255])

    pos, cdf = _sample_cdf(alpha, resolution=5)
    pos0 = np.asarray([0.000000000000,
                       1.256637061436,
                       2.513274122872,
                       3.769911184308,
                       5.026548245744])
    cdf0 = np.asarray([0.300000000000,
                       0.600000000000,
                       0.800000000000,
                       0.900000000000,
                       1.000000000000])

    assert_allclose(
        pos,
        pos0,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Error in evaluated positions.")
    assert_allclose(
        cdf,
        cdf0,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Error in evaluated cdfs.")


def test_kuiper_warning():
    alpha1 = np.asarray([0.291662278945,
                         5.899415544666,
                         5.402236718096,
                         3.728212505263,
                         5.303188109786,
                         3.737946900082,
                         3.850015526787,
                         4.902154536516,
                         3.631621444982,
                         5.341562525096])
    alpha2 = np.asarray([0.613650458799,
                         2.109660249330,
                         3.617555161298,
                         6.196794760548,
                         1.856071575830,
                         2.991480015107,
                         1.789200626487,
                         4.835921843822,
                         2.767491245457,
                         1.744565591973])
    p0 = 0.1
    k0 = 70
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p, k = pycircstat.kuiper(alpha1, alpha2)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "N=10 not found in table" in str(w[-1].message)


def test_kuiper():
    alpha1 = np.asarray([0.291662278945,
                         5.899415544666,
                         5.402236718096,
                         3.728212505263,
                         5.303188109786,
                         3.737946900082,
                         3.850015526787,
                         4.902154536516,
                         3.631621444982,
                         5.341562525096])
    alpha2 = np.asarray([0.613650458799,
                         2.109660249330,
                         3.617555161298,
                         6.196794760548,
                         1.856071575830,
                         2.991480015107,
                         1.789200626487,
                         4.835921843822,
                         2.767491245457,
                         1.744565591973])
    p0 = 0.1
    k0 = 70
    K0 = 67.395
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, k = pycircstat.kuiper(alpha1, alpha2)
    assert_allclose(p, p0, rtol=1e-4, atol=1e-4, err_msg="Error in p-values.")
    assert_allclose(
        k,
        k0,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Error in statistic k.")


def test_kuiper2():
    data1 = np.random.rand(10, 20, 30) * np.pi * 2.
    data2 = np.random.rand(10, 20, 30) * np.pi * 2.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, k = pycircstat.tests.kuiper(data1, data2, axis=0)
        assert_true(p.shape == (20, 30))
        for i in range(data1.shape[1]):
            for j in range(data1.shape[2]):
                p2, k2 = pycircstat.tests.kuiper(
                    data1[
                        :, i, j], data2[
                        :, i, j])
                assert_equal(p[i, j], p2)
                assert_equal(k[i, j], k2)


def test_kuiper3():
    data1 = np.random.rand(15, 20, 30) * np.pi * 2.
    data2 = np.random.rand(15, 20, 30) * np.pi * 2.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, k = pycircstat.tests.kuiper(data1, data2, axis=1)
        assert_true(p.shape == (15, 30))
        for i in range(data1.shape[0]):
            for j in range(data1.shape[2]):
                p2, k2 = pycircstat.tests.kuiper(
                    data1[
                        i, :, j], data2[
                        i, :, j])
                assert_equal(p[i, j], p2)
                assert_equal(k[i, j], k2)
