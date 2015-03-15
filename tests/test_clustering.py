from __future__ import absolute_import

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_true, assert_raises, raises

import pycircstat


def test_clustering():
    x = np.asarray([6.10599459, 0.14824723, 3.11272268, 3.45703846, 5.88211171, 3.53760218,
                    4.00392159, 2.76326071, 4.52222361, 4.05978276])
    # x = np.asarray([6.10599459, 0.14824723, 3.11272268, 3.45703846, 5.88211171])

    cl = pycircstat.clustering.AggCluster1D(numclust=4)
    _, ids = cl.train(x)

    assert_allclose(sorted(cl.centroids), sorted([6.139414042024, 2.937991695000, 3.497320320000, 4.193866918144]),
                    atol=1e-4, rtol=1e-4)
    _, testids = cl.test(x)
    assert_allclose(ids, testids)
if __name__ == "__main__":
    test_clustering()