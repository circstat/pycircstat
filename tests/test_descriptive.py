from __future__ import absolute_import

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal

import PyCircStat

def test_mean_constant_data():
    data = np.ones(1000)

    # We cannot use `assert_equal`, due to numerical rounding errors.
    assert_allclose(PyCircStat.mean(data), 1.0)
