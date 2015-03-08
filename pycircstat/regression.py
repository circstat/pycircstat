from __future__ import division
from pycircstat import CI
from pycircstat.iterators import index_bootstrap
import numpy as np

class BaseRegressor(object):
    def __init__(self):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError(u"{0:s}.train not implemented".format(self.__class__.__name__))

    def test(self, *args, **kwargs):
        raise NotImplementedError(u"{0:s}.test not implemented".format(self.__class__.__name__))

    def loss(self, x, y, lossfunc, ci=None, bootstrap_iter=1000):
        """
        Computes loss function between the predictions f(x) and the true y.

        :param x: inputs in radians. If multidimensional, each row must
                  be a specimen and each column a feature.
        :param y: desired outputs in radians. If multidimensional, each
                  row must be a specimen and each column a feature.
        :param lossfunc: loss function, must take an array of input and outputs and compute the loss.
        :param ci: confidence interval in [0,1]. If not None, bootstrapping is performed.
        :param bootstrap_iter: number of bootstrap iterations if
        :return: loss as computed by the loss function.
        """

        if ci is not None:
            yhat = self.predict(x)
            l = [lossfunc(y[idx], yhat[idx]) for idx in index_bootstrap(x.shape[0], bootstrap_iter)]
            mu = np.mean(l)
            q = 1-ci
            return mu, CI(np.percentile(l, q/2.*100), np.percentile(l, 1-q/2.*100))
        return lossfunc(y, self.predict(x))

    def predict(self, *args, **kwargs):
        raise NotImplementedError(u"{0:s}.predict not implemented".format(self.__class__.__name__))

s
