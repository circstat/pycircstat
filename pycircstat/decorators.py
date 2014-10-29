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

def get_var(f, varnames, args, kwargs, remove=False):
    fvarnames = f.__code__.co_varnames

    ret = {}
    for varname in varnames:
        if varname in fvarnames:
            var_pos = fvarnames.index(varname)
        else:
            raise ValueError('Function %s does not have variable %s.' % (f.__name__, varnames))

        if len(args) >= var_pos + 1:
            ret[varname] = args[var_pos]
            if remove:
                del args[var_pos]
        elif varname in kwargs:
            ret[varname] = kwargs[varname]
            if remove:
                del kwargs[varname]
        else:
            raise ValueError('%s was not specified in  %s.' % (varnames, f.__name__))

    return ret


class swap2zeroaxis:

    def __init__(self, inputs, out_idx):
        self.inputs = inputs
        self.out_idx = out_idx


    def __call__(self, f):

        @wraps(f)
        def wrapped_f(*args, **kwargs):
            to_swap = get_var(f, self.inputs, args, kwargs, remove=True)

            try:
                axis = get_var(f, ['axis'], args, kwargs)
            except ValueError:
                axis = None

            if axis is not None:
                for k, v in to_swap:
                    kwargs[k] = v.swapaxes(0, axis)
            else:
                for k, v in to_swap:
                    kwargs[k] = v.ravel()

            outputs = f(*args, **kwargs)

            # TODO: swap back
            return outputs

        return wrapped_f
