import numpy as np

def mod2pi(f):
    """
    Decorator to apply modulo 2*pi on the output of the function.

    The decorated function must either return a tuple of numpy.ndarrays or a
    numpy.ndarray itself.
    """
    def return_func(*args, **kwargs):
        ret = f(*args, **kwargs)

        if type(ret) == tuple:
            return tuple(r % (2*np.pi) for r in ret)
        elif type(ret) == np.ndarray:
            return ret % (2*np.pi)
        else:
            raise TypeError("Type not known!")

    return return_func
