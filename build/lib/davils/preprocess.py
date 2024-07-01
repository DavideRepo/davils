import numpy as np
import scipy as sp


def signal_preprocess(data, fs_orig, fs_new, hipass_freq=None, hipass_order=4, lowpass_filter='iir',
                    detrend_type='linear', axis=0):
    """
    Noted: It is not suggested to go above 0.25 Hz when hi-passing the signal, otherwise it will start to distort.
           Also it is not suggested to go above 4-order for the buttehrworth filter
    
    Args:
        data: Signals to be pre-processed, can be a List, a Dictionary or a numpy array
        fs_orig: Original sampling frequency
        fs_new: New (desired) sampling frequency
        hipass_freq: Hi-pass filter frequency ('None' for no hi-pass filtering)
        hipass_order: Specifies Butterworth filter order (defaults to 4)
        lowpass_filter: Type of low-pass filter (defaults to 'iir' filter)
        detrend_type:Type of desired detrending (defaults to 'linear', 'None' for no detrending)
        axis: Axis along which signal propagates

    Returns: List of pre-processed signals

    """
    if isinstance(data, dict):
        data_tmp = [row for row in data.values()]
    elif isinstance(data, list):
        data_tmp = [row for row in data]
    else:
        if len(data.shape) == 2:
            data_tmp = [row for row in data]
        elif len(data.shape) == 1:
            data_tmp = [data] 
        else:
            raise ValueError("Input data must have 1 or 2 dimensions, not more.")
    
    if detrend_type is not None:
        for i in range(len(data_tmp)):
            data_tmp[i] = sp.signal.detrend(data_tmp[i], axis=axis, type=detrend_type, overwrite_data=True)
    
    if hipass_freq is not None:
        sos = sp.signal.butter(hipass_order, hipass_freq, 'highpass', fs=fs_orig, output='sos')
        for i in range(len(data_tmp)):
            data_tmp[i] = sp.signal.sosfilt(sos, data_tmp[i], axis=axis)

    if fs_new is not None:
        for i in range(len(data_tmp)):
            data_tmp[i] = sp.signal.decimate(data_tmp[i], int(fs_orig/fs_new), n=None, ftype=lowpass_filter, axis=axis)

    if isinstance(data, dict):
        return {key: data_tmp[i] for i, key in enumerate(data.keys())}
    elif isinstance(data, list):
        return data_tmp
    else:
        return data_tmp[0]

