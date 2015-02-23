import os
import numpy as np
_data_path = '/'.join(os.path.realpath(__file__).split('/')[:-2] + ['data/'])


def load_kuiper_table():
    """
    Loads the lookup table for the kuiper test

    :return: table as numpy array
    """
    return np.load(_data_path + 'kuiper_table.npy')
