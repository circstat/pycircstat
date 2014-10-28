from __future__ import absolute_import

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_true

import pycircstat



def test_rayleigh():
    data = np.array([-0.94904375,  0.26575165, -0.03226759,  1.98389239,  2.0084918 ,
        0.56965871, -0.19199522, -0.33080002, -0.03141245,  1.18560637,
        4.31138236,  4.84776075,  2.10336819,  3.45465972,  4.64572697,
        5.80976322,  6.27160926,  0.67979652,  1.33776004,  0.58059308])
    p, z = pycircstat.tests.rayleigh(data)
    assert_allclose(0.0400878, p, atol=0.001, rtol=0.001)    
    assert_allclose(3.1678, z, atol=0.001, rtol=0.001)

def test_omnibus():
    data = np.array([-1.78277804,  0.20180845, -1.51291097,  0.57329272,  0.72195959,
        1.59947271,  1.4666837 , -0.36532379,  1.4455209 ,  0.77365236,
        5.8678466 ,  2.58327349,  0.25429634,  1.74902778,  4.13215085,
        0.20612467,  1.38484181,  1.72546928,  3.33570062,  3.96191276])
    p, m = pycircstat.tests.omnibus(data)
    assert_allclose(0.295715, p, atol=0.001, rtol=0.001)    
    assert_equal(5, m)





