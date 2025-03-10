import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def cov_dist(position, mean, std, l_scale, plot_kernel=False):

    cov = np.zeros([position.shape[0], position.shape[0]])
    if plot_kernel:
        plt.figure()
        plt.plot(np.linspace(0, 100, 100), np.exp(-(np.linspace(0, 100, 100) ** 2) / (2 * l_scale ** 2)) * 0.7 + 0.3)
        plt.xlabel('Distance [m]')
        plt.ylabel('Kernel function [-]')
        plt.title('Kernel (correlation) function over distance')
        
    for j in range(position.shape[0]):
        for i in range(position.shape[0]):
            if i != j and sp.spatial.distance.euclidean(position[i, 1:], position[j, 1:]) != 0:
                cov[i][j] = std ** 2 * (np.exp(-(sp.spatial.distance.euclidean(position[i, 1:], position[j, 1:]) ** 2) /
                                              (2 * l_scale ** 2)) * 0.7 + 0.3)
            else:
                cov[i][j] = std ** 2

        lam1, v1 = np.linalg.eig(cov)  # Solve eigenvalue problem using scipy
        cov_mod = v1.T @ cov @ v1  # Transform cov. matrix to uncorrelated space
        # var_mod = np.random.normal(np.full(len(position), mean), np.diag(cov_mod) ** 0.5)
        # var = np.linalg.inv(v1.T) @ var_mod  # Transform to correlated space        
        var = np.random.multivariate_normal(np.full(len(position), mean), cov)
    return var, cov, cov_mod


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


def get_maha_dist_deprecated(train_set, val_set=None, test_set=None):
    
    mean = np.mean(train_set, axis=1)
    cov = np.cov(train_set)
    std = np.diag(cov)**0.5
    cov_inv = np.linalg.inv(cov)

    msd_train = []
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



class MahalanobisDistance:
    """
    Class to compute the Mahalanobis sqaured distance for a given dataset.
    Datasets should be a 2D numpy array with samples in rows and features in columns.
    """
    def __init__(self):
        self.mean = None
        self.cov = None
        self.cov_inv = None

    def fit(self, dataset):
        """
        Fit the Mahalanobis distance model to the given dataset.
        :param dataset: 2D numpy array with samples in rows and features in columns.
        :return: None
        """
        self.mean = np.mean(dataset, axis=0)
        self.cov = np.cov(dataset, rowvar=False)
        self.cov_inv = np.linalg.inv(self.cov)

    def get_dist(self, dataset):
        """
        Compute the Mahalanobis distance for the given dataset.
        :param dataset: 2D numpy array with samples in rows and features in columns.
        :return: List of Mahalanobis distances for each sample in the dataset.
        """
        return [sp.spatial.distance.mahalanobis(sample, self.mean, self.cov_inv) ** 2 for sample in dataset]

