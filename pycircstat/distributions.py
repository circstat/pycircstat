from __future__ import absolute_import
from scipy import stats

from scipy.stats import rv_continuous
import numpy as np
import sys
from .decorators import swap2zeroaxis
from .descriptive import resultant_vector_length


@swap2zeroaxis(['alpha'], [0])
def kappa(alpha, w=None, axis=None):
    """
    Computes an approximation to the ML estimate of the concentration
    parameter kappa of the von Mises distribution.


    :param alpha: angles in radians OR alpha is length resultant
    :param w: number of incidences in case of binned angle data
    :param axis: kappa is computed along this axis
    :return: estimated value of kappa

    References: [Fisher1995]_ p. 88
    """

    if w is None:
        w = np.ones_like(alpha)
    else:
        assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    n = alpha.shape[axis]

    if n > 1:
        R = resultant_vector_length(alpha, w, axis=axis)
    else:
        R = alpha
    R = np.atleast_1d(R)

    kappa = np.asarray(0 * R)

    idx = R < 0.53

    kappa[idx] = 2. * R[idx] + R[idx]**3. + 5 * R[idx]**5. / 6

    idx = (R >= 0.53) & (R < 0.85)
    kappa[idx] = -.4 + 1.39 * R[idx] + 0.43 / (1. - R[idx])

    idx = R > 0.85
    kappa[idx] = 1. / (R[idx]**3. - 4. * R[idx]**2. + 3. * R[idx])

    if n < 15 and n > 1:
        idx = kappa < 2.
        kappa[idx] = kappa[idx] - 2 * (n * kappa[idx])**-1.
        idx0 = kappa < 0
        kappa[idx & idx0] = 0

        kappa[~idx] = (n - 1)**3 * kappa[~idx] / (n**3. + n)

    return kappa


class cardioid_gen(rv_continuous):

    """

    Cardioid distribution of a single random variable.

    .. math::

        p(x) = \\frac{1 + 2 \\rho \\cos(x - \\mu)}{2\\pi}


    :param mu: mean (in [0, 2*pi])
    :param rho: concentration parameter (in [-0.5,0.5])



    **Note:**

        - To use the distribution, use *cardioid*, not *cardioid_gen*.
        - See scipy.stats how to use distributions.

    **Example:**

    ::

        from pycircstat.distributions import cardioid
        import matplotlib.pyplot as plt
        import numpy as np
        mu, rho = np.pi, .2
        t = np.linspace(0,2*np.pi,1000)
        x = cardioid.rvs(mu, rho, size=5000)
        plt.plot(t, cardioid.pdf(t, mu, rho))
        plt.plot(t, cardioid.cdf(t, mu, rho))
        plt.hist(x, bins=50, normed=True)
        plt.show()

    References: [Jammalamadaka2001]_

    """

    def _argcheck(self, mu, rho):
        return (-.5 <= rho <= .5) and (0 <= mu <= 2. * np.pi)

    def _stats(self, mu, rho):
        return mu, None, None, None

    def _pdf(self, x, mu, rho):
        x = x % (2 * np.pi)
        return (1 + 2 * rho * np.cos(x - mu)) / 2. / np.pi

    def _cdf(self, x, mu, rho):
        # x = (x - mu - np.pi) % (2*np.pi)
        # return (-mu + 2*rho*np.sin(x-mu) + x + np.pi)/2/np.pi
        x = x % (2 * np.pi)
        return (2 * rho * np.sin(x - mu) + x + 2 * rho * np.sin(mu)) / \
            2 / np.pi

# hack for problems with numpy missing in readthedocs and mock
if not 'sphinx' in sys.modules:
    cardioid = cardioid_gen(name='cardioid', shapes="mu, rho")


class triangular_gen(rv_continuous):

    """

    triangular distribution of a single random variable.

    .. math::

        p(x) = \\frac{1}{8\\pi} (4-\\pi^2\\rho + 2\\pi\\rho |\\pi - x|)


    :param rho: concentration parameter (in [-0.5,0.5])



    **Note:**

        - To use the distribution, use *triangular*, not *triangular_gen*.
        - See scipy.stats how to use distributions.

    **Example:**

    ::

         from pycircstat.distributions import triangular
         import matplotlib.pyplot as plt
         import numpy as np
         rho = .4
         t = np.linspace(0,2*np.pi,1000)
         x = triangular.rvs(rho, size=5000)
         plt.plot(t, triangular.pdf(t, rho))
         plt.plot(t, triangular.cdf(t, rho))
         plt.hist(x, bins=50, normed=True)
         plt.show()


    References: [Jammalamadaka2001]_

    """

    def _argcheck(self, rho):
        return 0 <= rho <= 4. / np.pi ** 2.

    def _stats(self, rho):
        return 0, None, None, None

    def _pdf(self, x, rho):
        x = x % (2 * np.pi)
        return ((4 - np.pi ** 2. * rho + 2. * np.pi * rho * np.abs(np.pi - x))
                / 8. / np.pi)

    def _cdf(self, x, rho):
        x = x % (2 * np.pi)
        ret = 0 * x
        idx = (x < np.pi)
        ret[idx] = -0.125 * rho[idx] * x[idx] ** 2 + x[idx] * \
            (0.125 * np.pi ** 2 * rho[idx] + 0.5) / np.pi
        ret[~idx] = 0.125 * rho[~idx] * x[~idx] ** 2 + 0.25 * \
            np.pi ** 2 * rho[~idx] - x[~idx] * \
            (0.375 * np.pi ** 2 * rho[~idx] - 0.5) / np.pi
        return ret

# hack for problems with numpy missing in readthedocs and mock
if not 'sphinx' in sys.modules:
    triangular = triangular_gen(name='triangular', shapes="rho")

# wrapper for von Mises
vonmises = stats.vonmises
