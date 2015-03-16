from __future__ import absolute_import, division

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_true
from pycircstat.regression import CL1stOrderRegression, CCTrigonometricPolynomialRegression


def test_circlinregression():
    alpha = np.random.rand(200)*np.pi*2
    a0 = np.random.rand()*2*np.pi
    A0 = np.abs(np.random.randn())
    m0 = np.random.randn()*10

    x = m0 + A0*np.cos(alpha - a0)

    reg = CL1stOrderRegression()
    reg.train(alpha, x)
    m = reg._coef[-1]
    a = np.arctan2(reg._coef[1], reg._coef[0]) % (2*np.pi)
    A = np.sqrt(reg._coef[1]**2 + reg._coef[0]**2)


    assert_allclose(A,A0,err_msg="amplitudes do not match")
    assert_allclose(m,m0,err_msg="offsets do not match")
    assert_allclose(a,a0,err_msg="base angles do not match")


def test_circlin_prediction():
    alpha = np.random.rand(200)*np.pi*2
    a0 = np.random.rand()*2*np.pi
    A0 = np.abs(np.random.randn())
    m0 = np.random.randn()*10

    x = m0 + A0*np.cos(alpha - a0)

    reg = CL1stOrderRegression()
    reg.train(alpha, x)

    x2 = reg(alpha)

    assert_allclose(x,x2,err_msg="predictions do not match")


def test_circlin_test():
    alpha = np.random.rand(200)*np.pi*2
    x = np.random.randn(200)

    reg = CL1stOrderRegression()
    res = reg.test(alpha, x)
    assert_true(res.ix['Liddell-Ord','p'] > 0.0001, 'p-value is smaller than 0.0001')



def test_circcirc_regression():
    alpha = np.random.rand(1000)*np.pi*2
    beta = np.cos(alpha + np.random.rand()*2*np.pi)*np.pi
    reg = CCTrigonometricPolynomialRegression(degree=10)
    reg.train(alpha, beta)
    beta2 = reg(alpha)

    assert_allclose(beta,beta2,err_msg="predictions do not match", atol=1e-4, rtol=1e-4)
