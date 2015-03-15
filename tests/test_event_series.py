from __future__ import absolute_import

import numpy as np

from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_true

import pycircstat
from pycircstat import event_series as es


def test_vector_strength_spectrum():
    T = 3  # 2s
    sampling_rate = 10000.
    firing_rate = 10  # 1000Hz

    s = T * np.random.rand(np.random.poisson(firing_rate * T))

    w, vs_spec = es.vector_strength_spectrum(s, sampling_rate)

    F0 = []
    R = []
    lowcut, highcut = 500, 550
    idx = (w >= lowcut) & (w <= highcut)
    for i in np.where(idx)[0]:
        f0 = w[i]
        p0 = 1 / f0
        rho = pycircstat.resultant_vector_length((s % p0) / p0 * 2 * np.pi)

        F0.append(f0)
        R.append(rho)
    assert_allclose(R, vs_spec[idx])

def test_direct_vector_strength_spectrum():
    T = 3  # 2s
    sampling_rate = 10000.
    firing_rate = 10  # 1000Hz

    s = T * np.random.rand(np.random.poisson(firing_rate * T))

    w, vs_spec = es.vector_strength_spectrum(s, sampling_rate)
    lowcut, highcut = 500, 550
    idx = (w >= lowcut) & (w <= highcut)
    vs_2 = es.direct_vector_strength_spectrum(s, w[idx])
    assert_allclose(vs_2, vs_spec[idx])

def test_direct_vector_strength_spectrum_parallel():
    T = 3  # 2s
    sampling_rate = 10000.
    firing_rate = 10  # 1000Hz

    s = T * np.random.rand(np.random.poisson(firing_rate * T))

    w, vs_spec = es.vector_strength_spectrum(s, sampling_rate)
    lowcut, highcut = 1, 1400
    idx = (w >= lowcut) & (w <= highcut)
    vs_2 = es.direct_vector_strength_spectrum(s, w[idx])
    assert_allclose(vs_2, vs_spec[idx], rtol=1e-4, atol=1e-4)

