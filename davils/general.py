import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import colors
from colorsys import hls_to_rgb


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

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def is_diag(x):
    return np.allclose(x, np.diag(np.diagonal(x)))

def loop_cmap(cmap, max_colors):
    """
    Wraps a colormap to make it loop after max_colors.

    Parameters:
    - cmap: the original colormap (e.g., plt.get_cmap('tab20'))
    - max_colors: the number of colors in the original colormap

    Returns:
    - A function that takes an index and returns a color from the looping colormap
    """

    def looped(index):
        return cmap(index % max_colors)

    return looped


def loop_cmap_listed(cmap, max_colors):
    """
    Wraps a colormap to make it loop after max_colors.

    Parameters:
    - cmap: the original colormap (e.g., plt.get_cmap('tab20'))
    - max_colors: the number of colors in the original colormap

    Returns:
    - A looping ListedColormap that repeats after max_colors
    """
    colors_list = [cmap(i % max_colors) for i in range(max_colors * 10)]  # Loop through ten as many colors for safety
    return colors.ListedColormap(colors_list)


def exp_dist(arr1, arr2=None, var=1.0, l_scale=0.1, squared=False):
    if arr2 is None:
        arr2 = arr1
    arr1, arr2 = np.meshgrid(arr1, arr2)
    if squared:
        # dist = var ** 2 * np.exp(-((arr1 - arr2) / (arr1 + arr2)) ** 2 / (2 * l_scale ** 2))
        dist = var ** 2 * np.exp(-(np.log(arr1) - np.log(arr2)) ** 2 / (2 * l_scale ** 2))
    else:
        dist = var ** 2 * np.exp(-np.abs((arr1 - arr2) / (arr1 + arr2)) / l_scale)
    dist = np.nan_to_num(dist)
    return dist


def colorize_complex(z, sat=None):
    """
    Converts a complex array to an RGB image using hue, lightness, and saturation.

    Parameters:
    - z (np.ndarray): Complex input array.
    - sat (float, optional): Saturation value. If None, it is computed adaptively.

    Returns:
    - np.ndarray: RGB image array.
    """
    r = np.abs(z)
    arg = np.angle(z)
    h = (arg + np.pi) / (2 * np.pi) + 0.5  # Hue maps phase [-pi, pi] to [0,1]
    r_max = np.percentile(r, 95)  # Adaptive scale based on data spread
    l = np.tanh(r / r_max)  # Smooth lightness variation

    if sat:
        s = sat
    else:
        s = 0.8 + 0.2 * (r / (r + 1))  # Adaptive saturation
        # s = np.clip(1 - np.exp(-r / r_max), 0, 1)  # Exponential response
        # s = 1 / (1 + np.exp(-5 * (r / r_max - 0.5)))  # Sigmoid-controlled saturation
        # s = np.tanh(5 * (r / r_max - 0.5)) / 2 + 0.5  # tanh-based saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)
    c = np.array(c).swapaxes(0, 2)
    return c
