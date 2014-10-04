from scipy.stats import rv_continuous
import numpy as np
from PyCircStat import mod2pi


class cardioid_gen(rv_continuous):
    """

    Cardioid distribution of a single random variable.

    :param rho: concentration parameter (-0.5 <= rho <= 0.5)

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
        #return (-mu + 2*rho*np.sin(x-mu) + x + np.pi)/2/np.pi
        x = x % (2 * np.pi)
        return (2 * rho * np.sin(x - mu) + x + 2* rho * np.sin(mu)) / 2 / np.pi


cardioid = cardioid_gen(name='cardioid', a=0, b=2 * np.pi)