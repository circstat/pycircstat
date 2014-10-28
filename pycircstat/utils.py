"""
Statistical tests
"""
import numpy as np

def rad2ang(alpha):
    """
    Converts data in radians to data in angles

    :param alpha:     sample of angles in radians
    :return alpha:    sample of angles in angles

    References:

    """

    alpha =  alpha / np.pi * 180
    return alpha


def ang2rad(alpha):
    """
    Converts data in angles to data in radians

    :param alpha:     sample of angles in angles
    :return alpha:    sample of angles in radians

    References:
    """

    alpha = alpha * np.pi / 180
    return alpha








