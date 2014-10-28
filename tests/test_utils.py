from __future__ import absolute_import

import numpy as np

from numpy.testing import assert_allclose

from pycircstat import utils


def test_rad2ang_ang2rad():
    assert_allclose(np.pi, utils.ang2rad(utils.rad2ang(np.pi)), atol=0.001, rtol=0.001)    
    assert_allclose(180., utils.rad2ang(utils.ang2rad(180.)), atol=0.001, rtol=0.001)






