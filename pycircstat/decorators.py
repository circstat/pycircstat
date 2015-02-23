from __future__ import absolute_import

from functools import wraps
import numpy as np
from . import CI
from decorator import decorator


def mod2pi(f):
    """
    Decorator to apply modulo 2*pi on the output of the function.

    The decorated function must either return a tuple of numpy.ndarrays or a
    numpy.ndarray itself.
    """
    def wrapper(f, *args, **kwargs):
        ret = f(*args, **kwargs)

        if isinstance(ret, tuple):
            ret2 = []
            for r in ret:
                if isinstance(r, np.ndarray) or np.isscalar(r):
                    ret2.append(r % (2 * np.pi))
                elif isinstance(r, CI):
                    ret2.append(
                        CI(r.lower % (2 * np.pi), r.upper % (2 * np.pi)))
                else:
                    raise TypeError("Type not known!")
            return tuple(ret2)
        elif isinstance(ret, np.ndarray) or np.isscalar(ret):
            return ret % (2 * np.pi)
        else:
            raise TypeError("Type not known!")

    return decorator(wrapper, f)


def get_var(f, varnames, args, kwargs):
    fvarnames = f.__code__.co_varnames

    var_idx = []
    kwar_keys = []
    for varname in varnames:
        if varname in fvarnames:
            var_pos = fvarnames.index(varname)
        else:
            raise ValueError('Function %s does not have variable %s.'
                             % (f.__name__, varnames))
        if len(args) >= var_pos + 1:
            var_idx.append(var_pos)
        elif varname in kwargs:
            kwar_keys.append(varname)
        else:
            raise ValueError('%s was not specified in  %s.'
                             % (varnames, f.__name__))

    return var_idx, kwar_keys


class swap2zeroaxis:

    """
    This decorator is best explained by an example::

        @swap2zeroaxis(['x','y'], [0, 1])
        def dummy(x,y,z, axis=None):
            return np.mean(x[::2,...], axis=0), np.mean(y[::2, ...], axis=0), z

    This creates a new function that

    - either swaps the axes axis to zero for the arguments x and y if axis
      is specified in dummy or ravels x and y
    - swaps back the axes from the output arguments 0 and 1. Here it is
      assumed that the outputs lost one dimension during the function
      (e.g. like numpy.mean(x, axis=1) looses one axis).
    """

    def __init__(self, inputs, out_idx):
        self.inputs = inputs
        self.out_idx = out_idx

    def __call__(self, f):

        def _deco(f, *args, **kwargs):

            to_swap_idx, to_swap_keys = get_var(f, self.inputs, args, kwargs)
            args = list(args)

            # extract axis parameter
            try:
                axis_idx, axis_kw = get_var(f, ['axis'], args, kwargs)
                if len(axis_idx) == 0 and len(axis_kw) == 0:
                    axis = None
                else:
                    if len(axis_idx) > 0:
                        axis, args[axis_idx[0]] = args[axis_idx[0]], 0
                    else:
                        axis, kwargs[axis_kw[0]] = kwargs[axis_kw[0]], 0
            except ValueError:
                axis = None

            # adjust axes or flatten
            if axis is not None:
                for i in to_swap_idx:
                    if args[i] is not None:
                        args[i] = args[i].swapaxes(0, axis)
                for k in to_swap_keys:
                    if kwargs[k] is not None:
                        kwargs[k] = kwargs[k].swapaxes(0, axis)
            else:
                for i in to_swap_idx:
                    if args[i] is not None:
                        args[i] = args[i].ravel()
                for k in to_swap_keys:
                    if kwargs[k] is not None:
                        kwargs[k] = kwargs[k].ravel()

            # compute function
            outputs = f(*args, **kwargs)

            # swap everything back into place
            if len(self.out_idx) > 0 and axis is not None:
                if isinstance(outputs, tuple):
                    outputs = list(outputs)
                    for i in self.out_idx:
                        outputs[i] = outputs[i][np.newaxis, ...].\
                            swapaxes(0, axis).squeeze()

                    return tuple(outputs)
                else:
                    if self.out_idx != [0]:
                        raise ValueError("Single output argument and out_idx \
                                         != [0] are inconsistent!")
                    return outputs[np.newaxis, ...].swapaxes(0, axis).squeeze()
            else:
                return outputs

        return decorator(_deco, f)
