import numpy as np
# import scipy as sp
import pickle
import h5py
import dwdatareader as dw
from davils.preprocess import signal_preprocess
import logging

logger = None


def setup_logger(filename='app.log'):
    global logger
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


def load_variables(file_name):
    """
    Load variables from pickle file.

    Arguments
    ---------------------------
    file_name : str
        name of the file where variables are stored
    as_locals : 'True', optional
        if True, variables are loaded as local variables in Python console

    Returns
    ---------------------------
    var_dict : dict
        dictionary of loaded variables
    """
    with open(file_name, 'rb') as file:
       var_dict = pickle.load(file)
    # except FileNotFoundError:
    #     with open(file_name+'.p', 'rb') as file:
    #        var_dict = pickle.load(file)
        
    return var_dict


def save_variables(variables, file_name, var_names=None):
    """
    Save variables to file with pickle.

    Arguments
    ---------------------------
    variables : list
        list of 'variables names
    file_name : str
        name of the file where variables will be stored
    var_names : 'none', optional
        list of variables to be saved. If 'None', variables are fetched from local variables in Python console,
        according to names in 'var_names'

    Returns
    ---------------------------
    var_dict : dict
        dictionary of loaded variables
    """
    var_dict = {}
    if not var_names:
        for idx,var in enumerate(variables):
            var_dict['var_'+str(idx)] = var
    else:
        for idx,var in enumerate(variables):
            var_dict[var_names[idx]] = var

    with open(file_name+'.pkl', 'wb') as file:
        # A new file will be created

        pickle.dump(var_dict, file)


def load_norsenga_h5(filename, ch=['x', 'y', 'z', 'T'], north=True, south=True):

    with h5py.File(filename, "r") as f:
        print(list(f['AI'].keys()))
        print(list(f['Plugins'].keys()))

        accels = {}
        strains = {}
        weather = {}
        temps = {}
        
        for i in range(1, 11):
            for k in ch:
                try:
                    if k == 'T':
                        if south:
                            temps[f'as{i:02d}'] = np.array(f['AI'][f'AS{i:02d}-{k}'])
                        if north:
                            temps[f'an{i:02d}'] = np.array(f['AI'][f'AN{i:02d}-{k}'])
                    else:
                        if south:
                            accels[f'as{i:02d}{k}'] = np.array(f['AI'][f'AS{i:02d}-{k}'])
                        if north:
                            accels[f'an{i:02d}{k}'] = np.array(f['AI'][f'AN{i:02d}-{k}'])
                except KeyError:
                    pass

        for i in range(1, 13):
            try:
                if south:
                    strains[f'ss{i:02d}'] = np.array(f['AI'][f'SS{i:02d}'])
                if north:
                    strains[f'sn{i:02d}'] = np.array(f['AI'][f'SN{i:02d}'])
            except KeyError:
                pass

        for key in f['Plugins']:
            weather[key] = np.array(f['Plugins'][key])

    return accels, strains, temps, weather


def load_norsenga_dwd(filename, ch=['x', 'y', 'z', 'T'], north=True, south=True, verbose=True):
    """
    :param filename: Name of .dxd file to read
    :param ch: what channels you want to import from the accelerometers recordings
    :param north: False if you want to exclude the north truss
    :param south: False if you want to exclude the south truss
    :return: data ordered in 4 different dictionaries (accels: accelerometers time series, strains: strain guages time
             series, temps: thermocouples time series (at acceleromentes), weather: weather station data time series.
             Accelerometers are names as "as01-x", strain guages as "ss01", thermocouples as "as05" in their respective
             dictionaries.
    """

    with dw.open(filename) as f:
        if verbose:
            print(f.info)
            print(list(f.keys()))

        duration = f.info.duration

        acc = {}
        str = {}
        wth = {}
        tmp = {}

        for i in range(1, 11):
            for k in ch:
                if k == 'T':
                    try:
                        tmp[f'as{i:02d}'] = np.array(f[f'AS{i:02d}-{k}'].series())
                    except KeyError:
                        pass
                    try:
                        tmp[f'an{i:02d}'] = np.array(f[f'AN{i:02d}-{k}'].series())
                    except KeyError:
                        pass
                else:
                    try:
                        if south:
                            acc[f'as{i:02d}{k}'] = np.array(f[f'AS{i:02d}-{k}'].series())
                        if north:
                            acc[f'an{i:02d}{k}'] = np.array(f[f'AN{i:02d}-{k}'].series())
                    except KeyError:
                        pass

        for i in range(1, 13):
            try:
                if south:
                    str[f'ss{i:02d}'] = np.array(f[f'SS{i:02d}'].series())
                if north:
                    str[f'sn{i:02d}'] = np.array(f[f'SN{i:02d}'].series())
            except KeyError:
                pass

        for key in ['Wind direction', 'Wind speed', 'Compass heading (north)', 'Barometric pressure',
                    'Relative humidity', 'Air temperature', 'Dewpoint', 'Supply voltage',
                    '5 min average wind direction', '5 min average wind speed', 'Gust direction', '3 s wind gust']:
            wth[key] = np.array(f[key].series())

    return acc, str, tmp, wth, duration


class LoadNorsenga:
    def __init__(self, filename, ch_group=['x', 'y', 'z', 'str', 'tmp', 'wth'], slicing=None, pp_options=None, verbose=False, transform=True):
        self.filename = filename
        self.ch_group = ch_group
        self.verbose = verbose
        self.slicing = slicing
        self.pp_options = pp_options

        self.load_data()
        if transform:
            self.transform_acc()

    def load_data(self):
        with dw.open(self.filename) as f:
            if self.verbose:
                print(f.info)
                print(list(f.keys()))

            self.info = f.info
            self.acc = {}
            self.str = {}
            self.tmp = {}
            self.wth = {}

            if self.slicing is None:
                slicing = np.arange(0, int(self.info.duration * self.info.sample_rate))
            else:
                slicing = np.arange(int(self.slicing[0] * self.info.sample_rate),
                                    int(self.slicing[1] * self.info.sample_rate))

            acc_labels = [label for label in f.keys() if (label[:2] == 'AS' or label[:2] == 'AN') and label[-1] != 'T']
            str_labels = [label for label in f.keys() if label[:2] == 'SS' or label[:2] == 'SN']
            tmp_labels = [label for label in f.keys() if label[-1] == 'T']
            wth_labels = ['Wind direction', 'Wind speed', 'Compass heading (north)', 'Barometric pressure',
                          'Relative humidity', 'Air temperature', 'Dewpoint', 'Supply voltage',
                          '5 min average wind direction', '5 min average wind speed', 'Gust direction', '3 s wind gust']

            for label in acc_labels:
                for axis in ['x', 'y', 'z']:
                    if axis in self.ch_group and label[-1]==axis:
                        if self.pp_options is None:
                            self.acc[label.lower()] = f[label].series().to_numpy()[slicing].copy()
                        if self.pp_options is not None:
                            self.acc[label.lower()] = signal_preprocess(f[label].series().to_numpy()[slicing].copy(),
                                                                        **self.pp_options)

            if 'str' in self.ch_group:
                for label in str_labels:
                    self.str[label.lower()] = f[label].series().to_numpy()[slicing].copy()

            if 'tmp' in self.ch_group:
                for label in tmp_labels:
                    if self.pp_options is None:
                        self.tmp[label[:-2].lower()] = f[label].series().to_numpy()[slicing].copy()
                    if self.pp_options is not None:
                        self.tmp[label[:-2].lower()] = signal_preprocess(f[label].series().to_numpy()[slicing].copy(),
                                                                    **self.pp_options)

            if 'wth' in self.ch_group:
                for label in wth_labels:
                    self.wth[label.lower()] = f[label].series()

    def preprocess(self, data_to_process, fs_orig, fs_new, hipass_freq=None, lopass_freq=None, hipass_order=4, lopass_order=4, lowpass_filter='iir',
                    detrend_type=None, axis=0):
        for attr in data_to_process:
            if hasattr(self, attr):
                data = getattr(self, attr)
                processed_data = signal_preprocess(data, fs_orig, fs_new, hipass_freq, lopass_freq, hipass_order, lopass_order, lowpass_filter,
                    detrend_type, axis)
                setattr(self, attr, processed_data)
            else:
                print(f"Attribute {attr} not found in the class.")

    def transform_acc(self):
        if self.verbose:
            print('Accelerations are transformed to global coordinates. '
                  '\n\tUx: along-bridge, towards N/E direction (cabinet) \n\tUy: across-bridge, towards N/W direction \n\tUz: vertical')
        for l in list(set(label[:-2] for label in self.acc.keys())):
            if l[1] == 'n':
                self.acc[l+'-x'], self.acc[l+'-y'], self.acc[l+'-z'] = self.acc[l+'-y'], self.acc[l+'-z'], self.acc[l+'-x']
            elif l[1] == 's':
                self.acc[l+'-x'], self.acc[l+'-y'], self.acc[l+'-z'] = -self.acc[l+'-y'], -self.acc[l+'-z'], self.acc[l+'-x']

