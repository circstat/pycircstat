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






