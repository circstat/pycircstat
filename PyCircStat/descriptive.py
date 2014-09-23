"""
Descriptive statistical functions
"""

import numpy as np

def mean(data, w=None, axis=0):
    """
    Compute mean direction of circular data.
    """

    if w is None:
        w = np.ones_like(data)

    if data.shape != w.shape:
        raise ValueError('Input dimensions do not match:', data.shape, alpha.shape)

    r = w * np.exp(1j*data).sum(axis=axis)

    mu = np.angle(r)

    return mu
