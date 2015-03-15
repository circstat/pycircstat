import itertools
import numpy as np
from scipy import stats
from pycircstat import var


def convolve_dirac_gauss(t, trial, sigma=1.):
    """
    Convolves event series represented as time points of Dirac deltas with
    the pdf of a Gaussian

    :param t: time points at which the convolution will be computed
    :param trial: array of event times
    :param sigma: std of the Gaussian convolution filter
    :returns: convolved event train
    """
    ret = 0 * t
    for st in trial:
        ret[:] += stats.norm.pdf(t, loc=st, scale=sigma)
    return ret


def vector_strength_spectrum(event_times, sampling_rate, time=None):
    """
    Computes the vector strength (resultant vector length) between a series of events and a
    sinusoid of many frequencies. The resolution in frequency space is determines by the
    sampling rate.

    :param event_times: event times in seconds
    :param sampling_rate: sampling rate in Hz
    :param time: np.array of time points or two values that denote a (right open) time range
    :return: frequencies and vector strength between the events and sinusoids at these frequencies

    **Example**

    ::

        T = 3 # three seconds
        n = 20
        sampling_rate = 10000. # sampling rate in Hz
        events = T*np.random.rand(n)
        w, vs_spec = es.vector_strength_spectrum(events, sampling_rate)

    """
    dt = 1. / sampling_rate
    if time is not None:
        if len(time) == 2:
            t = np.arange(time[0], time[1], dt)
        else:
            assert np.abs(
                dt - (time[1] - time[0])) < 1e-6, "Sampling rate and dt in time do not agree."
            t = time
    else:
        t = np.arange(
            np.amin(event_times) -
            50. /
            sampling_rate,
            np.amax(event_times) +
            50. /
            sampling_rate,
            dt)

    w = np.fft.fftfreq(len(t), d=dt)
    sigma = 1. / 2. / np.pi / sampling_rate * 8

    x = convolve_dirac_gauss(t, event_times, sigma=sigma)

    a = np.abs(np.fft.fft(x)) * dt / len(event_times)
    a[w == 0] = np.NaN
    gf = np.exp(-2 * np.pi**2 * sigma**2 * w**2)
    return w, a / gf

def _vector_strength(param):
    event_times, w = param
    return 1-var( (event_times % (1./w) )*w*2*np.pi )

def direct_vector_strength_spectrum(event_times, frequencies):
    """
    Computes the direct vector strength spectrum for the given frequencies.

    :param event_times: event times in seconds
    :param frequencies: locking frequencies in Hz
    :return: vector strength spectrum
    """
    ret = np.asarray([1-var( (event_times % (1./w) )*w*2*np.pi ) for w in frequencies])

    return ret