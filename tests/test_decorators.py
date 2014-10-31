from __future__ import absolute_import

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_true

from pycircstat.decorators import swap2zeroaxis


def test_swap2zeroaxis():

    @swap2zeroaxis(['x', 'y'], [0, 1])
    def dummy(x, y, z, axis=None):
        return np.mean(x[::2, ...], axis=0), np.mean(y[::2, ...], axis=0), z

    x = np.random.randn(3, 5, 7, 9)
    y = np.random.randn(3, 5, 7, 9)
    z = np.random.randn(3, 5, 7, 9)

    xx, yy, zz = dummy(x, y, z, axis=1)
    assert_allclose(xx, np.mean(x[:, ::2, ...], axis=1))
    assert_allclose(yy, np.mean(y[:, ::2, ...], axis=1))
    assert_allclose(zz, z)

    xx, yy, zz = dummy(x, y, z, 2)
    assert_allclose(xx, np.mean(x[:, :, ::2, ...], axis=2))
    assert_allclose(yy, np.mean(y[:, :, ::2, ...], axis=2))
    assert_allclose(zz, z)

    xx, yy, zz = dummy(x, y, z, axis=None)
    assert_allclose(xx, np.mean(x.ravel()[::2]))
    assert_allclose(yy, np.mean(y.ravel()[::2]))
    assert_allclose(zz, z)

    xx, yy, zz = dummy(x, y, z)
    assert_allclose(xx, np.mean(x.ravel()[::2]))
    assert_allclose(yy, np.mean(y.ravel()[::2]))
    assert_allclose(zz, z)

    @swap2zeroaxis(['x'], [0])
    def dummy(x, axis=None):
        return np.mean(x[::2, ...], axis=0)

    xx = dummy(x, axis=0)
    assert_allclose(xx, np.mean(x[::2, ...], axis=0))


if __name__ == "__main__":
    test_swap2zeroaxis()
