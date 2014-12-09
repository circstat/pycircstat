from __future__ import absolute_import

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_true

import pycircstat as circ

test_data_2d = np.array([
                    [0.58429, 0.88333],
                    [1.14892, 2.22854],
                    [2.87128, 3.06369],
                    [1.07677, 1.49836],
                    [2.96969, 1.51748],
                    ])


def test_kappa_basic():
    """circ.kappa:  test basic functionality of circ.distributions.kappa"""
    kappa = circ.distributions.kappa(test_data_2d)
    assert_allclose(kappa, 1.6221, rtol=1e-4)

def test_kappa_axis0():
    """circ.kappa:  test functionality of circ.distributions.kappa along axis=0"""
    kappa = circ.distributions.kappa(test_data_2d, axis=0)
    assert_allclose(kappa, [1.0536, 1.1514], rtol=1e-4)

def test_kappa_axis0():
    """circ.kappa:  test functionality of circ.distributions.kappa along axis=1"""
    kappa = circ.distributions.kappa(test_data_2d, axis=1)
    assert_allclose(kappa, [4.50679, 0.38271, 10.83801, 2.28470, 0.23442], rtol=1e-4)
