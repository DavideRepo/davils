import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def get_corr_matrix(cov,plot=False):
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std,std)
    if plot:
        plt.figure(constrained_layout=True)
        C = plt.imshow(corr, cmap=None, interpolation=None)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(range(len(cov)),range(len(cov)))
        plt.yticks(range(len(cov)),range(len(cov)))
        plt.colorbar(C)
        plt.title('Correlation Matrix')
    return corr


def get_maha_dist(train_set, val_set, test_set, train_set_msd=True):

    mean = np.mean(train_set, axis=1)
    cov = np.cov(train_set)
    std = np.diag(cov)**0.5
    cov_inv = np.linalg.inv(cov)

    msd_train = []
    if train_set_msd is True:
        for el in np.transpose(train_set):
            msd_train.append(sp.spatial.distance.mahalanobis(el, mean, cov_inv) ** 2)

    msd_val = []
    if val_set is not None:
        for el in np.transpose(val_set):
            msd_val.append(sp.spatial.distance.mahalanobis(el, mean, cov_inv) ** 2)

    msd_test = []
    if test_set is not None:
        for el in np.transpose(test_set):
            msd_test.append(sp.spatial.distance.mahalanobis(el, mean, cov_inv) ** 2)

    return msd_test, msd_val, msd_train


