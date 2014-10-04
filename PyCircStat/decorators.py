import numpy as np
from PyCircStat import nd_bootstrap, CI
from PyCircStat import percentile as cpercentile


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
        elif type(ret) == np.ndarray or np.isscalar(ret):
            return ret % (2*np.pi)
        else:
            raise TypeError("Type not known!")

    return return_func

class bootstrap:
    def __init__(self, no_bootstrap, scale='linear'):
        self.no_boostrap = no_bootstrap
        self.scale = scale

    def _get_var(self, f, what, default, args, kwargs):
        varnames = f.func_code.co_varnames

        if what in varnames:
            what_idx = varnames.index(what)
        else:
            raise ValueError('Function %s does not have variable %s.' % (f.__name__, what))

        if len(args) >= what_idx+1:
            val = args[what_idx]
            args[what_idx] = default
        elif what in kwargs:
            val = kwargs.pop(what, default)
        else:
            val = default

        return val


    def __call__(self, f):

        def wrapped_f(*args, **kwargs):
            ci = self._get_var(f, 'ci', None, args, kwargs)
            bootstrap_iter = self._get_var(f, 'bootstrap_iter', None, args, kwargs)
            axis = self._get_var(f, 'axis', 0, args, kwargs) # TODO: change that to None if decided

            alpha = args[:self.no_boostrap]
            args0 = args[self.no_boostrap:]

            if ci is not None:
                r = [f(*(a+args0), **kwargs) for a in nd_bootstrap(alpha, bootstrap_iter, axis=axis)]

            r0 = f(*(alpha+args0), **kwargs)
            if self.scale == 'linear':
                ci_low, ci_high = np.percentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100], axis=0)
            elif self.scale == 'circular':
                ci_low, ci_high = cpercentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100],
                                              q0=(r0+np.pi)%(2*np.pi), axis=0)
            else:
                raise ValueError('Scale %s not known!' % (self.scale, ))
            return r0, CI(ci_low, ci_high)


        return wrapped_f
