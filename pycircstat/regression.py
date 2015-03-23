from __future__ import division
import warnings
from pycircstat import CI
from pycircstat.iterators import index_bootstrap
import numpy as np
from scipy import stats
import pandas as pd

class BaseRegressor(object):
    """
    Basic regressor object. Mother class to all other regressors.

    Regressors support indexing which is passed to the coefficients.

    Regressors also support calling. In this case the prediction function is called.
    """

    def __init__(self):
        self._coef = None

    def istrained(self):
        """
        Returns whether the regressor is trained of not.

        :return: True if trained
        """
        return self._coef is not None

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
            q = 1 - ci
            return mu, CI(np.percentile(l, q / 2. * 100), np.percentile(l, 1 - q / 2. * 100))
        return lossfunc(y, self.predict(x))

    def predict(self, *args, **kwargs):
        raise NotImplementedError(u"{0:s}.predict not implemented".format(self.__class__.__name__))

    def __getitem__(self, item):
        return self._coef.__getitem__(item)

    def __setitem__(self, key, value):
        return self._coef.__getitem__(key, value)

    def __call__(self, *args, **kwargs):
        assert self.istrained(), "Regressor must be trained first."
        return self.predict(*args, **kwargs)


class CL1stOrderRegression(BaseRegressor):
    """
    Implements a circular linear regression model of the form

    .. math::
        x = m + a \\cos(\\alpha - \\alpha_0)

    The actual model is equivalently implemented as

    .. math::
        x = c_1 \\cos(\\alpha) + c_2 \\sin(\\alpha) + m

    References: [Jammalamadaka2001]_

    """

    def __init__(self):
        super(CL1stOrderRegression, self).__init__()

    def train(self, alpha, x):
        """
        Estimates the regression coefficients. Only works for 1D data.

        :param alpha: independent variable, angles in radians
        :param x: dependent variable
        """
        assert alpha.shape == x.shape, "x and alpha need to have the same shape"
        assert len(alpha.shape) == 1, "regression only implemented for 1D data"
        assert len(x.shape) == 1, "regression only implemented for 1D data"

        X = np.c_[np.cos(alpha), np.sin(alpha), np.ones_like(alpha)]
        c = np.dot(np.linalg.pinv(X), x)
        self._coef = c

    def predict(self, alpha):
        """
        Predicts linear values from the angles.

        :param alpha: inputs, angles in radians
        :return: predictions
        """
        X = np.c_[np.cos(alpha), np.sin(alpha), np.ones_like(alpha)]

        return np.dot(X, self._coef)

    def test(self, alpha, x):
        """
        Tests whether alpha and x are significantly correlated.
        The test assumes that x is normally distributed. The test
        function uses a Shapiro-Wilk test to test this assumption.

        :param alpha: independent variable, angles in radians
        :param x: dependent variable
        :return: test results of Shapiro-Wilk and Liddell-Ord test
        :rtype: pandas.DataFrame

        References: [Jammalamadaka2001]_
        """
        w, psw = stats.shapiro(x)
        if psw < 0.05:
            warnings.warn("This test requires Gaussian distributed x")

        rxc, rxs, rcs = np.corrcoef(x, np.cos(alpha))[0,1], np.corrcoef(x, np.sin(alpha))[0,1], \
                        np.corrcoef(np.cos(alpha), np.sin(alpha))[0,1]
        n = len(alpha)
        r2 = (rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1 - rcs**2)
        f = (n-3)*r2/(1-r2)
        p = stats.f.sf(f, 2, n-3)

        df = pd.DataFrame(dict(
            test = ['Shapiro-Wilk','Liddell-Ord'],
            statistics = [w, f],
            p = [psw, p],
            dof = [None, (2, n-3)]
        )).set_index('test')
        return df

class CCTrigonometricPolynomialRegression(BaseRegressor):
    """
    Implements a circular circular regression model of the form

    .. math::
        \\cos(\\beta) = a_0 + \\sum_{k=1}^d a_k \\cos(k\\alpha) + b_k \\sin(k\\alpha)

        \\sin(\\beta) = c_0 + \\sum_{k=1}^d c_k \\cos(k\\alpha) + d_k \\sin(k\\alpha)

    The angles :math:`\\beta` are estimated via :math:`\\hat\\beta = atan2(\\sin(\\beta), \\cos(\\beta))`



    :param degree: degree d of the trigonometric polynomials

    References: [Jammalamadaka2001]_
    """

    def __init__(self, degree=3):
        super(CCTrigonometricPolynomialRegression, self).__init__()
        self.degree = degree

    def train(self, alpha, beta):
        """
        Estimates the regression coefficients. Only works for 1D data.

        :param alpha: independent variable, angles in radians
        :param beta: dependent variable, angles in radians
        """
        X = np.vstack([np.ones_like(alpha)] + [np.cos(alpha*k) for k in np.arange(1., self.degree+1)] \
                                  + [np.sin(alpha*k) for k in np.arange(1., self.degree+1)]).T
        self._coef = np.c_[np.dot(np.linalg.pinv(X), np.cos(beta)),
                           np.dot(np.linalg.pinv(X), np.sin(beta))]

    def predict(self, alpha):
        """
        Predicts linear values from the angles.

        :param alpha: inputs, angles in radians
        :return: predictions, angles in radians
        """
        X = np.vstack([np.ones_like(alpha)] + [np.cos(alpha*k) for k in np.arange(1., self.degree+1)] \
                                  + [np.sin(alpha*k) for k in np.arange(1., self.degree+1)]).T
        beta = np.dot(X, self._coef)
        return np.arctan2(beta[:,1], beta[:,0])

