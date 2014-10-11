from functools import wraps
import numpy as np
from pycircstat import CI


def mod2pi(f):
    """
    Decorator to apply modulo 2*pi on the output of the function.

    The decorated function must either return a tuple of numpy.ndarrays or a
    numpy.ndarray itself.
    """
    @wraps(f)
    def return_func(*args, **kwargs):
        ret = f(*args, **kwargs)

        if type(ret) == tuple:
            ret2 = []
            for r in ret:
                if type(r) == np.ndarray or np.isscalar(r):
                    ret2.append(r % (2*np.pi) )
                elif type(r) is CI:
                    ret2.append(CI(r.lower % (2*np.pi), r.upper % (2*np.pi)))
                else:
                    raise TypeError("Type not known!")
            return tuple(ret2)
        elif type(ret) == np.ndarray or np.isscalar(ret):
            return ret % (2*np.pi)
        else:
            raise TypeError("Type not known!")

    return return_func

