import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def lag_finder(y1, y2, fs):
    n = len(y1)

    corr = sp.signal.correlate(y2, y1, mode='same') / np.sqrt(sp.signal.correlate(y1, y1, mode='same')[int(n/2)] * sp.signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5 * n / fs, 0.5 * n / fs, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')

    plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 6)*1000) + ' ms')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()

