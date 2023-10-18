import numpy as np
import scipy as sp


def signal_preprocess(data, fs_orig, fs_new, hipass_freq=None, hipass_order=4, lowpass_filter='iir',
                    detrend_type='linear', axis=0):
    """

    Args:
        data: Signals to be pre-processed, can be a List
        fs_orig: Original sampling frequency
        fs_new: New (desired) sampling frequency
        hipass_freq: Hi-pass filter frequency ('None' for no hi-pass filtering)
        hipass_order: Specifies Butterworth filter order (defaults to 4)
        lowpass_filter: Type of low-pass filter (defaults to 'iir' filter)
        detrend_type:Type of desired detrending (defaults to 'linear', 'None' for no detrending)
        axis: Axis along which signal propagates

    Returns: List of pre-processed signals

    """

    if isinstance(data, np.ndarray):
        data = [data]

    if hipass_freq:
        sos = sp.signal.butter(hipass_order, hipass_freq, 'hp', fs=fs_orig, output='sos')

        for i in range(len(data)):
            data[i] = sp.signal.sosfilt(sos, data[i], axis=axis)

    data_res = []
    for i in range(len(data)):
        data_res.append(sp.signal.decimate(data[i], int(fs_orig/fs_new), n=None, ftype=lowpass_filter, axis=axis))

    if detrend_type is not None:
        for i in range(len(data)):
            data_res[i] = sp.signal.detrend(data_res[i], axis=axis, type=detrend_type, overwrite_data=True)

    return data_res
