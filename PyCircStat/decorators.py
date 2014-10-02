import numpy as np

def mod2pi(f):

    def return_func(*args, **kwargs):
        ret = f(*args, **kwargs)

        if type(ret) == tuple:
            return tuple(r % (2*np.pi) for r in ret)
        elif type(ret) == np.ndarray:
            return ret % (2*np.pi)
        else:
            raise TypeError("Type not known!")

    return return_func