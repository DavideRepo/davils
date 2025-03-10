import numpy as np
import scipy as sp
import numba as nb
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import mplcursors
from .dynamics import mpcval as mpc
from scipy.linalg import svd
import jax.scipy as jsp


def moving_average(data, n=3):

    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n


def get_poly(coeff, x):

    deg = np.shape(coeff)[0] - 1
    y = np.zeros(np.shape(x))

    for i in range(np.shape(coeff)[0]):
        y = y + coeff[i] * x ** (deg - i)

    return y


def find_those_peaks(x, y, SlopeThreshold=None, AmpThreshold=None, smoothwidth=None, smoothtype=2):

    FWHM = 0.5 / (x[1] - x[0])

    if SlopeThreshold is None:
        FWHM = 0.5 / (x[1] - x[0])
        SlopeThreshold = 0.7 * FWHM ** (-4)
        print(f'Slope Threshold = {SlopeThreshold}')

    if AmpThreshold is None:
        AmpThreshold = 3*np.average(y)
        print(f'Amplitude Threshold = {AmpThreshold}')

    if smoothwidth is None:
        smoothwidth = FWHM

    def deriv(a):
        # First derivative of vector using 2-point central difference.
        n = len(a)
        d = np.zeros_like(a)
        d[0] = a[1] - a[0]
        d[n-1] = a[n-1] - a[n-2]

        for j in range(1, n-1):
            d[j] = (a[j+1] - a[j-1]) / 2

        return d

    def fastsmooth(Y, w, type, ends):

        def sa(Y, smoothwidth, ends):
            w = int(smoothwidth)
            SumPoints = np.sum(Y[:w])
            s = np.zeros_like(Y)
            halfw = int(w / 2)
            L = len(Y)

            for k in range(1, L-w):
                s[k+halfw-1] = SumPoints
                SumPoints = SumPoints - Y[k]
                SumPoints = SumPoints + Y[k+w]

            s[k+halfw] = np.sum(Y[L-w+1:L])
            SmoothY = s / w

            if ends == 1:
                startpoint = int((smoothwidth + 1) / 2)
                SmoothY[0] = (Y[0] + Y[1]) / 2

                for k in range(1, startpoint):
                    SmoothY[k] = np.mean(Y[0:(2*k-1)])
                    SmoothY[L-k+1] = np.mean(Y[L-2*k+2:L])

                SmoothY[L] = (Y[L] + Y[L-1]) / 2

            return SmoothY

        if type == 1:
            SmoothY = sa(Y, w, ends)

        elif type == 2:
            SmoothY = sa(sa(Y, w, ends), w, ends)

        elif type == 3:
            SmoothY = sa(sa(sa(Y, w, ends), w, ends), w, ends)

        return SmoothY

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if smoothwidth < 1:
        smoothwidth = 1

    smoothwidth = int(round(smoothwidth))

    if smoothwidth > 1:
        dy = deriv(y)
        d = fastsmooth(dy, smoothwidth, smoothtype, ends=0)
    else:
        d = deriv(y)

    P = np.zeros((0, 3))
    peak = int(1)

    for j in range(2 * int(smoothwidth / 2) - 1, len(y)-smoothwidth-1):
        if np.sign(d[j]) > np.sign(d[j+1]):
            if d[j] - d[j+1] > SlopeThreshold:
                if y[j] > AmpThreshold:
                    PeakY = np.amax(y[j-5 : j+6])
                    PeakX = x[np.argmax(y[j-5 : j+6])+j-5]
                    P = np.vstack((P, [peak, PeakX, PeakY]))
                    peak += int(1)

    print(P)
    return P

def find_those_peaks2(x, y, n_peaks, smooth_window=None, smooth_passes=2):

    def get_max(array, n):

        indices = np.argpartition(array, -n)[-n:]  # Indices of the top n maximum values
        values = array[indices]  # Top n maximum values
        sorted_indices = indices[np.argsort(values)][::-1]  # Sort indices based on values in descending order
        sorted_values = values[np.argsort(values)][::-1]  # Sort values in descending order

        return sorted_values, sorted_indices


    def fastsmooth(y, w, type, ends):

        def sa(y, smoothwidth, ends):
            w = int(smoothwidth)
            SumPoints = np.sum(y[:w])
            s = np.zeros_like(y)

            halfw = int(w / 2)
            L = len(y)

            for k in range(1, L-w):
                s[k+halfw-1] = SumPoints
                SumPoints = SumPoints - y[k]
                SumPoints = SumPoints + y[k + w]

            s[k+halfw] = np.sum(y[L - w + 1:L])
            SmoothY = s / w

            if ends == 1:
                startpoint = int((smoothwidth + 1) / 2)
                SmoothY[0] = (y[0] + y[1]) / 2

                for k in range(1, startpoint):
                    SmoothY[k] = np.mean(y[0:(2 * k - 1)])
                    SmoothY[L-k+1] = np.mean(y[L - 2 * k + 2:L])

                SmoothY[L] = (y[L] + y[L - 1]) / 2

            return SmoothY

        if type == 1:
            SmoothY = sa(y, w, ends)

        elif type == 2:
            SmoothY = sa(sa(y, w, ends), w, ends)

        elif type == 3:
            SmoothY = sa(sa(sa(y, w, ends), w, ends), w, ends)

        return SmoothY


    if smooth_window is None:
        smooth_window = int(len(x)/4)

    smooth_y = fastsmooth(y, smooth_window, smooth_passes, 0)
    plt.figure()
    plt.plot(x, smooth_y)

    peaksY, peaks_index = get_max(y - smooth_y, n_peaks)
    peaksX = x[peaks_index]
    peaksN = np.arange(1, len(peaks_index)+1)

    P = np.column_stack([peaksN, peaksX, peaksY])
    print(P)
    return P


def rdn_noise(nodes, dofs_in_nodes, t, std_scale, dist_corr=[False, None],  rotation=False):

    LoadCovMat = np.zeros([nodes.shape[0], nodes.shape[0]])

    if dist_corr[0]:
        for j in range(nodes.shape[0]):
            for i in range(nodes.shape[0]):
                if i != j and sp.spatial.distance.euclidean(nodes[i, 1:], nodes[j, 1:]) != 0:
                    LoadCovMat[i][j] = std_scale**2 * np.exp(-(sp.spatial.distance.euclidean(nodes[i, 1:], nodes[j, 1:]) ** 2) /
                                                             (2 * dist_corr[1] ** 2)) * 0.6 + 0.4
                else:
                    LoadCovMat[i][j] = std_scale**2

        lam1, v1 = sp.linalg.eig(LoadCovMat)  # Solve eigenvalue problem using scipy
        LoadCovMat_modal = np.matmul(np.matmul(v1.T, LoadCovMat), v1)  # Transform cov. matrix to uncorrelated space
        U_rid = np.zeros((nodes.shape[0], t.shape[0]))
        for i in range(nodes.shape[0]):
            U_rid[i, :] = np.random.normal(0, LoadCovMat_modal[i, i] ** 0.5, t.shape[0])

        X_rid = np.matmul(np.linalg.inv(v1.T), U_rid)  # Transform to correlated space

    else:
        LoadCovMat = np.diag(np.full(nodes.shape[0],std_scale**2))
        LoadCovMat_modal = LoadCovMat
        X_rid = np.zeros((nodes.shape[0], t.shape[0]))
        for i in range(nodes.shape[0]):
            X_rid[i, :] = np.random.normal(0, LoadCovMat_modal[i, i] ** 0.5, t.shape[0])

    X = np.repeat(X_rid, repeats=dofs_in_nodes, axis=0)

    if not rotation:
        X[2::3, :] = np.zeros((1, t.shape[0]))

    return X, LoadCovMat, LoadCovMat_modal


def PSD_matrix(data, window, f_s, zero_pad=None, plot=False, plotOverlay=False, f_n=None, f_plot=None):
    """
    This function computes the (Cross) Power Spectral Density Matrix of the signal data, according to Welch method.

    Returns:
      f_CSD     - [numpy array], array of the discretized frequencies after Welch smoothening
      data_CSD  - [numpy array, complex] (nch*nch*len(f_CSD)), the (Cross) Power Spectral Density Matrix of the signal(s)
      peaks     - if findPeaks=True, [numpy array] with as columns: peak number, peak index, peak value of the overlayed
                (normalized) cross power spectra plot ("peak-picking" method)

    Requires as input:
      data      - [List] or [numpy array] (nch*l), with signal(s) on one dimensions and number of channels in the other
      window    - [windows class], window function used for Welch method
      f_s       - [int], sampling freq. of the signal(s)
    """

    data = np.array(data)
    if data.shape[0] > data.shape[1]: # Transpose data if channels are in columns
        data = data.T

    nch = data.shape[0]
    f_CSD, _ = sp.signal.csd(data[0, :], data[0, :], f_s, window, nfft=zero_pad)
    data_CSD = np.zeros((nch, nch, len(f_CSD)), dtype=complex)

    for i in range(nch):
        for j in range(nch):
            _, data_CSD[i, j, :] = sp.signal.csd(data[i, :], data[j, :], f_s, window, nfft=zero_pad)

    if plot:
        if not f_plot:
            f_plot = f_CSD[-1]
        fig, axs = plt.subplots(nch, nch, figsize=(20, 15))
        for i in range(nch):
            for k in range(nch):
                axs[i, k].plot(f_CSD[f_CSD <= f_plot], np.real(data_CSD[i, k, f_CSD <= f_plot]))
                axs[i, k].plot(f_CSD[f_CSD <= f_plot], np.imag(data_CSD[i, k, f_CSD <= f_plot]))
                axs[i, k].grid(True)
        for k in range(nch):
            axs[i, k].set_xlabel("Freq. [Hz]")
        plt.tight_layout()

    return f_CSD, data_CSD


@nb.njit(parallel=True)
def PSD_matrix_numba(data, window, f_s, zero_pad=None):
    """
    Computes the cross-spectral density (CSD) matrix for multi-channel data using Welch's method.

    Parameters:
    -----------
    data : numpy.ndarray
        2D array with shape (nch, l), where nch is the number of channels and l is the signal length.
    window : array-like or callable
        Window function to be used in Welch's method.
    f_s : int
        Sampling frequency of the signal.
    zero_pad : int, optional
        Length of the FFT used, if different from the signal length (for zero-padding). Defaults to None.

    Returns:
    --------
    f_PSD : numpy.ndarray
        Frequency values corresponding to the PSD estimates.
    PSD : numpy.ndarray
        3D array of shape (nch, nch, len(f_PSD)) containing the cross-spectral density matrix for each channel pair.
    """
    nch = data.shape[0]

    # Compute frequency values using the first signal
    f_PSD, _ = sp.signal.csd(data[0, :], data[0, :], f_s, window, nfft=zero_pad, dtype=float)

    # Preallocate PSD matrix
    PSD = np.zeros((nch, nch, len(f_PSD)), dtype=complex)

    # Fill PSD matrix with cross-spectral density for each channel pair
    for i in nb.prange(nch):
        for j in range(nch):
            _, PSD[i, j, :] = sp.signal.csd(data[i, :], data[j, :], f_s, window, nfft=zero_pad)

    return f_PSD, PSD


def FDD(data_PSD, f_PSD, f_s, vec=None, plot=False, plotLim=None, plotLog=False, findPeaks=False, f_n=None, n_sval=None):
    """
    This function performs the Frequency Domain Decomposition of data in input.

    Returns:
      S_val     - singular values [numpy array]
      S_vec_dx  - right singular vectors [list of 2D numpy arrays]
      S_vec_sx  - right singular vectors [list of 2D numpy arrays]
      peaks     - if findPeaks=True, [2D numpy array] with as columns: peak number, peak index, peak value

    Requires as input:
      data_PSD  - [List] or [numpy array] (n*n*l), with the Power Spectral Densities of n channels for l freq. steps
      f_PSD     - [List] or [numpy array] (1*l), array of the discretized frequencies
      f_s       - [int], sampling freq. of the signals
    """

    PSD_matrix = np.array(data_PSD)

    nch = len(data_PSD)
    nf = len(data_PSD[0][0])

    S_val = np.zeros((nch, nf))

    S_vec_sx = np.zeros((nch, nch, nf), dtype='complex')
    S_vec_dx = np.zeros((nch, nch, nf), dtype='complex')
    # S_vec_sx = [np.zeros((nch, nch)) for _ in range(nf)]
    # S_vec_dx = [np.zeros((nch, nch)) for _ in range(nf)]

    # Performing SVD
    for i in range(nf):
        U, S, V = np.linalg.svd(PSD_matrix[:, :, i])
        # S = np.diag(S)
        S_val[:,i] = np.sqrt(S)
        S_vec_sx[:,:,i] = U
        S_vec_dx[:,:,i] = V

    if findPeaks:
        # peaks_index, _ = spsi.find_peaks(np.abs(S_val[0, 0]), distance=20, height=0.2*np.max(S_val[0,0]),
        #                                 prominence=0.2*(np.max(S_val[0,0])-np.mean(S_val[0,0])))
        peaks = find_those_peaks(f_PSD, S_val[0,:],SlopeThreshold=None,AmpThreshold=None,smoothwidth=None,smoothtype=2)
        # peaks = find_those_peaks2(f_PSD, S_val[0,:], 4)

    elif not findPeaks:
        peaks = []

    if plot:
        if plotLim is None:
            plotLim = f_PSD[-1]

        plt.figure(figsize=(10,5))
        plt.title(fr'Singular values of a PSD($\omega$) matrix')
        plt.semilogy(f_PSD, S_val[0, :], label="sv1", linewidth=1.5)
        plt.semilogy(f_PSD, S_val[1, :], label="sv2", linewidth=1.5)
        plt.semilogy(f_PSD, S_val[2, :], label="sv3", linewidth=1.5)

        if findPeaks:
            plt.plot(peaks[:,1], peaks[:,2], 'o', linestyle='')

        if plotLog:
            plt.yscale("log")

        if f_n is not None:
            plt.vlines(f_n, np.zeros(len(f_n)), np.ones(len(f_n)) * np.max(S_val[:3, f_PSD <= plotLim]),
                       colors="grey", linestyles="dashed")

        plt.xlim(0, plotLim)
        plt.xlabel('Frequency [Hz]')
        plt.grid()
        plt.legend()
        # plt.title("Singular values")
        plt.tight_layout()
        plt.show()

    return S_val[:n_sval,:], S_vec_sx[:,:n_sval,:], S_vec_dx[:n_sval,:,:]


@nb.njit(parallel=True)
def FDD_numba(PSD_matrix, f_PSD, f_s, n_sval=1):
    """
    Perform Frequency Domain Decomposition (FDD) on the given PSD matrix.

    Parameters:
    -----------
    PSD_matrix : numpy.ndarray
        3D array of shape (nch, nch, nf) representing the cross-spectral density matrix for each frequency.
    f_PSD : numpy.ndarray
        1D array of frequency values corresponding to the PSD matrix.
    f_s : int
        Sampling frequency of the signal.
    n_sval : int, optional
        Number of singular values and vectors to return. Defaults to 1.

    Returns:
    --------
    sval : numpy.ndarray
        2D array of shape (n_sval, nf) containing the singular values at each frequency.
    svec_sx : numpy.ndarray
        3D array of shape (nch, n_sval, nf) containing the left singular vectors at each frequency.
    svec_dx : numpy.ndarray
        3D array of shape (n_sval, nch, nf) containing the right singular vectors at each frequency.
    """
    nch = len(PSD_matrix)
    nf = len(PSD_matrix[0][0])

    sval = np.zeros((n_sval, nf), dtype='float')
    svec_sx = np.zeros((nch, n_sval, nf), dtype='complex')
    svec_dx = np.zeros((n_sval, nch, nf), dtype='complex')

    # Parallelize over the frequency bins (nf)
    for i in nb.prange(nf):  # nb.prange allows Numba to parallelize this loop
        U, S, Vt = np.linalg.svd(PSD_matrix[:, :, i], full_matrices=False)

        sval[:, i] = np.sqrt(S[:n_sval])
        svec_sx[:, :n_sval, i] = U[:, :n_sval]
        svec_dx[:n_sval, :, i] = Vt[:n_sval, :]

    return sval, svec_sx, svec_dx


@nb.njit(parallel=True, fastmath=True)
def FDD_numba_alt(PSD_matrix, f_PSD, f_s, n_sval=1):
    nch = len(PSD_matrix)
    nf = PSD_matrix.shape[2]

    sval = np.zeros((n_sval, nf), dtype=np.float64)
    svec_sx = np.zeros((nch, n_sval, nf), dtype=np.complex128)
    svec_dx = np.zeros((n_sval, nch, nf), dtype=np.complex128)

    for i in nb.prange(nf):
        U, S, Vh = svd(PSD_matrix[:, :, i], full_matrices=False, overwrite_a=True, lapack_driver='gesdd')

        sval[:, i] = np.sqrt(S[:n_sval])
        svec_sx[:, :n_sval, i] = U[:, :n_sval]
        svec_dx[:n_sval, :, i] = Vh[:n_sval, :]

    return sval, svec_sx, svec_dx


def FDD_modes(S_val, S_vec, peaks_index=None, plot=None, model=None):
    """
        This function returns the mode shapes at automatically detected peaks of the first singular values,
        if not provided, some user-defined frequencies are required as console input.
    """

    if peaks_index is None:
        peaks_index = np.array([int(item) for item in input("Enter the freq. at peaks (Hz) > ").split()])

    modes_fdd = np.array(S_vec.shape[0], peaks_index.shape[0])

    for i in peaks_index:
        modes_fdd[:, i] = S_vec[:][0][i]
        modes_fdd[:, i] = modes_fdd[:, i] / np.max(np.real(modes_fdd[:, i]))

    return modes_fdd, peaks_index


def FDD_plot_modeshapes(f_CSD, S_vec_sx, scale, x, y, z, dropout=None):
    """
        This function plots & returns the mode shapes at some user-defined frequncies.
        Requires geometry of the problem as input.

        !!!! ---> The function plots results according to some problem-specific, built-in information (for now!)
        !!!! ---> To be included in FDD_modes
    """

    FDD_freq_ID = np.array(list(map(float,(input('Type selected frequencies (space-delimited): ').split(' ')))))
    FDD_mode_shapes_ID = [S_vec_sx[i][:,0] for i in range(len(S_vec_sx)) if np.any(np.isclose(f_CSD[i], FDD_freq_ID))]

    xpos=np.array([-12,-6,-3.6,0,3.2,6.4,9.6,12.8,16.6,16.6,19.2,22.4,25.6,32,38])

    fig, axs = plt.subplots(len(FDD_mode_shapes_ID), 2)

    for i in range(len(FDD_mode_shapes_ID)):
        axs[i, 0].plot(xpos, np.real(FDD_mode_shapes_ID[i][::2]))
        axs[i, 1].plot(xpos, np.real(FDD_mode_shapes_ID[i][1::2]))

    axs[0, 0].title.set_text('horizontal')
    axs[0, 1].title.set_text('vertical')

    # Define the grid points and corresponding z-values

    for i in range(len(FDD_mode_shapes_ID)):
        xs = x
        ys = y + np.real(FDD_mode_shapes_ID[i][::2])*scale
        zs = z + np.real(FDD_mode_shapes_ID[i][1::2])*scale
        xs_ = np.delete(xs, dropout)
        ys_ = np.delete(ys, dropout)
        zs_ = np.delete(zs, dropout)

        # Create the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, c='r', marker='o')

        # Customize the plot & adding line
        ax.plot(xs_, ys_, zs_, '-o', c='b')
        ax.set_xlim3d(min(xs), max(xs))
        ax.set_ylim3d(-(max(xs) - min(xs))/2, (max(xs) - min(xs))/2)
        ax.set_zlim3d(min(xs), max(xs))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()


def plot_deformed_dofs(nodes, elements, u, dof_u, skd):
    hor_size = np.max(nodes[:, 1]) - np.min(nodes[:, 1])
    hor_mid = 1 / 2 * (np.max(nodes[:, 1]) + np.min(nodes[:, 1]))

    vert_size = np.max(nodes[:, 2]) - np.min(nodes[:, 2])
    vert_mid = 1 / 2 * (np.max(nodes[:, 2]) + np.min(nodes[:, 2]))

    max_dim = np.max([hor_size, vert_size]) * 1.3

    nodes_deformed = np.copy(nodes)

    for i in range(len(dof_u)):
        nodes_deformed[dof_u[i]//3, dof_u[i] % 3+1] = nodes_deformed[dof_u[i]//3, dof_u[i] % 3+1] + skd * u[i]

    fig = plt.figure()

    plt.plot(nodes_deformed[np.unique(dof_u//3), 1], nodes_deformed[np.unique(dof_u//3), 2], "o")

    hor_size = np.max(nodes[:, 1]) - np.min(nodes[:, 1])
    hor_mid = 1 / 2 * (np.max(nodes[:, 1]) + np.min(nodes[:, 1]))
    vert_size = np.max(nodes[:, 2]) - np.min(nodes[:, 2])
    vert_mid = 1 / 2 * (np.max(nodes[:, 2]) + np.min(nodes[:, 2]))
    max_dim = np.max([hor_size, vert_size]) * 1.1

    #plt.plot(nodes[:, 1], nodes[:, 2], 'o')

    for k in range(elements.shape[0]):
        x1 = [nodes[nodes[:, 0] == elements[k, 1], 1], nodes[nodes[:, 0] == elements[k, 2], 1]]
        x2 = [nodes[nodes[:, 0] == elements[k, 1], 2], nodes[nodes[:, 0] == elements[k, 2], 2]]

        plt.plot(x1, x2)

    plt.xlim([hor_mid - max_dim / 2, hor_mid + max_dim / 2])
    plt.ylim([vert_mid - max_dim / 2, vert_mid + max_dim / 2])
    plt.grid()

    return fig


def FDD_get_modeshapes(f_CSD, S_vec_sx, scale=None, x=None, y=None, z=None, dropout=None, plot=False):
    """
        1D replica of FDD_plot_modeshapes, to be deprecated
    """
    FDD_freq_ID = np.array(list(map(float,(input('Type selected frequencies (space-delimited): ').split(' ')))))
    FDD_mode_shapes_ID = [S_vec_sx[i][:,0] for i in range(len(S_vec_sx)) if np.any(np.isclose(f_CSD[i], FDD_freq_ID))]

    if plot:
        xpos=np.array([-12,-6,-3.6,0,3.2,6.4,9.6,12.8,16.6,16.6,19.2,22.4,25.6,32,38])


        fig, axs = plt.subplots(len(FDD_mode_shapes_ID), 1)

        plt.figure()
        for i in range(len(FDD_mode_shapes_ID)):
            axs[i].plot(xpos, np.real(FDD_mode_shapes_ID[i]))

        axs[0].title.set_text('vertical')

        # Define the grid points and corresponding z-values

        for i in range(len(FDD_mode_shapes_ID)):
            xs = x + np.real(FDD_mode_shapes_ID[i])*scale
            ys = y + np.real(FDD_mode_shapes_ID[i])*scale
            zs = z + np.real(FDD_mode_shapes_ID[i])*scale
            xs_ = np.delete(xs, dropout)
            ys_ = np.delete(ys, dropout)
            zs_ = np.delete(zs, dropout)

            # Create the 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs, c='r', marker='o')

            # Customize the plot & adding line
            ax.plot(xs_, ys_, zs_, '-o', c='b')
            ax.set_xlim3d(min(xs), max(xs))
            ax.set_ylim3d(-(max(xs) - min(xs))/2, (max(xs) - min(xs))/2)
            ax.set_zlim3d(min(xs), max(xs))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Show the plot
            plt.show()

    return FDD_freq_ID, FDD_mode_shapes_ID


def koma_stabplot_matplot(lambd, orders, frequency_unit='rad/s', damped_freq=False, flim=None, overlay=None, f_n=None, overlaylabel=None, plottitle=None, mpcgradient=False, phi=None, save=False, filename=None):
    """

    Parameters
    ----------
    This function is a modified version of 'stabplot' in the KOMA package (which works with JupiterNotebook only). It
    returns the stabilization polt using 'matplotlib' and 'mplcursors' (both must be installed in the Python environment
    and imported in the script).

    lambd:          Lambdas from find_stable_poles
    orders:         Orders from find_stable_poles
    frequency_unit  Freq. unit, 'Hz' or 'rad/s'
    damped_freq     If damped freqs are being used
    flim            Freq. limit in the plot (xlim)
    overlay         You can overaly a signal on top of the SSI stabilization plot (like for exaple singular values from
                    FFD). 'List': first element is the signal (1-dimensional array), second element is its sampling
                    frequency (i.e., the f_max in the freq. domain)
    f_n             You can overlay known values of natural frequencies as vertical lines ('List' or 'np array')

    Returns         Matplotlib figure class.
    -------

    """
    # Create suffix and frequency value depending on whether damped freq. is requested or not
    if damped_freq:
        omega = np.abs(np.imag(lambd))
    else:
        omega = np.abs(lambd)
    
    if frequency_unit.lower() == 'rad/s':
        x = omega
        xlabel = f'$\omega_{"d" if damped_freq else "n"} \; [{frequency_unit}]$'
    elif frequency_unit.lower() == 'hz':
        x = omega/2/np.pi
        xlabel = f'$f_{"d" if damped_freq else "n"} \; [{frequency_unit}]$'
        
    # Create frequency/period axis and corresponding labels
    xi = -np.real(lambd) / np.abs(lambd)
    ylabel = r'Model order $n$'

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 9))
    if mpcgradient:
        x_grad = mpc(phi)
        cmap = LinearSegmentedColormap.from_list("my_colormap", ["red", "blue"])
        scatter = ax.scatter(x, orders, c=x_grad**2, cmap=cmap, vmin=0, vmax=1, alpha=0.6, s=20, label='SSI poles')  # power-2 to make it more visible
        text = [f'xi = {xi_i*100:.2f}% mpc = {x_grad*100:.2f}% \nfreq = {x_i:.2f} {frequency_unit} idx = {ix}' for ix, xi_i, x_grad, x_i in zip(range(len(xi)), xi, x_grad, x)]
    else:    
        scatter = ax.scatter(x, orders, c='blue', alpha=0.6, s=20, label='SSI poles')
        text = [f'xi = {xi_i*100:.2f}% freq = {x_i:.2f} {frequency_unit} idx = {ix}' for ix, xi_i, x_i in zip(range(len(xi)), xi, x)]

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.label.set_size(16)  # Set xlabel font size
    ax.yaxis.label.set_size(16)  # Set ylabel font size
    ax.set_title('Cov-SSI Stabilization Plot \n Poles', fontsize=16)
    ax.set_xticks(np.arange(0,25+1))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)

    if f_n is not None:
        plt.vlines(f_n, np.zeros(len(f_n)), np.ones(len(f_n)) * ax.get_ylim()[1],
                   colors="grey", linestyles="dashed")
        #plt.xlim(0, plotLim)
        plt.grid()

    if overlay is not None:
        for idx, signal in enumerate(overlay):
            signal_x = np.linspace(0, flim, len(signal))
            if idx == 0:
                signal_max = np.max(signal)
            signal_scaling_factor = 0.9 * ax.get_ylim()[1] / signal_max
            signal_scaled = signal * signal_scaling_factor
            
            if overlaylabel is None:
                label = 'overlay' + f' {idx}'
            else:
                label = overlaylabel + f' ${idx+1}$'
            ax.plot(signal_x, signal_scaled, alpha=0.75, label=label)

    if flim is not None:
        ax.set_xlim(0, flim)
    
    if plottitle is not None:
        ax.set_title(plottitle, fontsize=16)
    
    plt.legend(fontsize=14)
    plt.tight_layout()

    if save:
        if filename:
            plt.savefig(filename, dpi=150)
        else:
            plt.savefig('SSI_stabilization_plot.png', dpi=150)
        return

    # mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(text[int(sel.index)]))
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(text[int(sel.index)]))
    # cursor.connect("remove", lambda sel: sel.annotation.remove())

    plt.show()
    return fig
