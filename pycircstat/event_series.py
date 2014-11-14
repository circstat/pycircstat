import numpy as np
from scipy import stats


def concolve_dirac_gauss(t, trial, sigma=1.):
    """
    Convolves event series represented as time points of Dirac deltas with
    the pdf of a Gaussian

    :param t: time points at which the convolution will be computed
    :param trial: array of event times
    :param sigma: std of the Gaussian convolution filter
    :returns: convolved event train
    """
    ret = 0*t
    for st in trial:
        ret[:] += stats.norm.pdf(t, loc=st, scale=sigma)
    return ret


def vector_strength_spectrum(event_times, sampling_rate):
    """
    Computes the vector strength (resultant vector length) between a series of events and a
    sinusoid of many frequencies. The resolution in frequency space is determines by the
    sampling rate.

    :param event_times: event times in seconds
    :param sampling_rate: sampling rate in Hz
    :return: frequencies and vector strength between the events and sinusoids at these frequencies

    **Example**

    ::

        T = 3 # three seconds
        n = 20
        sampling_rate = 10000. # sampling rate in Hz
        events = T*np.random.rand(n)
        w, vs_spec = es.vector_strength_spectrum(events, sampling_rate)

    """
    dt = 1./sampling_rate
    t = np.arange(np.amin(event_times)-50./sampling_rate, np.amax(event_times)+50./sampling_rate, dt)

    w = np.fft.fftfreq(len(t), d=dt)
    sigma = 1./2./np.pi/sampling_rate*8

    x = concolve_dirac_gauss(t,  event_times, sigma=sigma)

    a = np.abs(np.fft.fft(x))*dt / len(event_times)
    gf = np.exp(-2 * np.pi**2 * sigma**2 * w**2)

    return w, a/gf



