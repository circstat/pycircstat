"""
Descriptive statistical functions
"""

import numpy as np

def mean(data, w=None, axis=0):
    """
    Compute mean direction of circular data.

    :param data: circular data
    :param w: 	 weightings in case of binned angle data
    :param axis: compute along this dimension, default is 0
    :return: circular mean
    """

    if w is None:
        w = np.ones_like(data)

    if data.shape != w.shape:
        raise ValueError('Input dimensions do not match:', data.shape)

    r = (w * np.exp(1j*data)).sum(axis=axis)

    mu = np.angle(r)

    return mu
