from scipy.stats import rv_continuous
import numpy as np
from pycircstat import mod2pi


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

    ..code::

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
        return (-.5 <= rho <= .5) and (0 <= mu <= 2 * np.pi)

    def _stats(self, mu, rho):
        return mu, None, None, None

    def _pdf(self, x, mu, rho):
        x = x % (2 * np.pi)
        return (1 + 2 * rho * np.cos(x - mu)) / 2. / np.pi

    def _cdf(self, x, mu, rho):
        # x = (x - mu - np.pi) % (2*np.pi)
        # return (-mu + 2*rho*np.sin(x-mu) + x + np.pi)/2/np.pi
        x = x % (2 * np.pi)
        return (2 * rho * np.sin(x - mu) + x + 2 * rho * np.sin(mu)) / 2 / np.pi


cardioid = cardioid_gen(name='cardioid', a=0, b=2 * np.pi, shapes="mu, rho")


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

    ..code::

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
        return (4 - np.pi ** 2 * rho + 2 * np.pi * rho * np.abs(np.pi - x)) / 8. / np.pi

    def _cdf(self, x, rho):
        x = x % (2 * np.pi)
        ret = 0 * x
        idx = (x < np.pi)
        ret[idx] = -0.125 * rho[idx] * x[idx]**2 + x[idx] * (0.125 * np.pi**2 * rho[idx] + 0.5) / np.pi
        ret[~idx] = 0.125 * rho[~idx] * x[~idx]**2 + 0.25 * np.pi**2 * rho[~idx] - x[~idx] * \
                        (0.375 * np.pi ** 2 * rho[~idx] - 0.5) / np.pi
        return ret

triangular = triangular_gen(name='triangular', a=0, b=2 * np.pi, shapes="rho")

