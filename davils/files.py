# import numpy as np
# import scipy as sp
import pickle
import h5py


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
    with open(file_name+'.pkl', 'rb') as file:
       var_dict = pickle.load(file)

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


def load_norsenga_h5(filename):

    with h5py.File("data_2023_10_10_180000.h5", "r") as f:
        print(list(f['AI'].keys()))
        print(list(f['Plugins'].keys()))

        accels = {}
        strains = {}
        weather = {}
        for i in range(2, 11):
            if i in [5, 9]:
                for k in ['x', 'y', 'z', 'T']:
                    accels[f'as{i:02d}{k}'] = np.array(f['AI'][f'AS{i:02d}-{k}'])
            else:
                for k in ['x', 'y', 'z']:
                    accels[f'as{i:02d}{k}'] = np.array(f['AI'][f'AS{i:02d}-{k}'])

        for i in range(1, 13):
            if i == 7:
                continue
            strains[f'ss{i:02d}'] = np.array(f['AI'][f'SS{i:02d}'])

        for key in f['Plugins']:
            weather[key] = np.array(f['Plugins'][key])

    return accels, strains, weather
