# this script contains the main tracking algorithm - implementation 2 (first version)
# it has been tested on the Norsenga dataset

# it searches fot the reference modes (passed manually or discovered automatically in the seed data) though the singular
# vectors of the power spectral density matrix of the response.

import numpy as np
import scipy as sp
from .dynamics import mpcval
from .dynamics import xmacmat_alt
from .general import exp_dist
from .general import loop_cmap_listed
from .general import loop_cmap

import time
# import pickle
# import h5py
import hdbscan

from matplotlib import pyplot as plt
from matplotlib import colors


class FDDTracker:
    """
    FDDTracker is a class for tracking modes in a dataset using the Frequency Domain Decomposition (FDD) method.

    Attributes:
        chunk_idx (list): List of chunk indexes.
        fn_list (dict): Dictionary of lists of natural frequencies of the modes.
        phi_list (dict): Dictionary of lists of mode shapes of the modes.
        xi_list (dict): Dictionary of lists of damping ratios of the modes (not yet implemented).
        svid_list (dict): Dictionary of lists of singular vectors order of the identified modes.
        MACs (dict): Dictionary of lists of MAC values of the modes vs. the reference modes, at moment of detection.
        STM (dict): Short-term memory containing 'fn', 'phi', 'fn_all', 'phi_all', and 'weights'.
        num_modes (int): Number of tracking modes.
        lag (int): Number of previous detections to keep into memory (memory length).
        fs (float): Frequency resolution in Hz (1/bin_w) (legacy).
        bin_w (float): Width of the frequency bin in Hz (1/fs).
        f_lim (list): Limits of the frequency range to track.
        n_bins (int): Number of frequency bins.
        track_metric (str): Metric used to track the modes ('mac' or 'mac_by_sval').
        track_metric_weights (dict): Weights for the weighted average metric.
        use_avg (bool): Use the average of the shapes in the memory when computing the MAC.
        max_mem_check (float): Percentage of memory elements to check when computing the MAC (deprecated).
        threshold (float): MAC threshold for detection.
        f_scale (float): Width of the Tukey window centered at the expected natural frequency in percentage.
        MAC_weighted (bool): Use MAC-weighted averaging when computing the average modal parameters.
        verbose (bool): Print additional information.
        run_time (dict): Run time statistics containing 'total', 'update', and 'track'.
        sval_seed (list): Seed singular values.
        svec_seed (list): Seed singular vectors.
        labels_split (list): Labels of the discovered modes.
        labels_keep_split (list): Labels of the discovered modes kept for tracking.
        debug (bool): Enable debug mode.
    """

    metric_options = {
        'mac': lambda mac, sval, f_win: mac * f_win,
        'mac*sval': lambda mac, sval, f_win: mac * sval * f_win,
        'logmac*sval': lambda mac, sval, f_win: -1 / np.log(mac) * sval * f_win,  # experimental
        'wavg': lambda mac, sval, f_win:
            self.track_metric_weights['mac'] * mac * f_win + self.track_metric_weights['sval'] * (sval * f_win) /
            (sval * f_win).max(axis=-1, keepdims=True) if self.track_metric_weights else None
    }

    dist_metric_options = {
        'mac+freq': lambda sval_seed, l_scale, mac_conc: exp_dist([range(len(sval_seed[0][0]))] * len(sval_seed),
                                                                  [range(len(sval_seed[0][0]))] * len(sval_seed), var=1,
                                                                  l_scale=l_scale, squared=True) * mac_conc,
        'mac': lambda sval_seed, l_scale, mac_conc: mac_conc
        }

    def __init__(self, lag, bin_w, f_lim, track_metric='mac_by_sval', threshold=0.85, f_scale=0.2, use_avg=False,
                 mac_weighted=False, track_metric_weights=None, verbose=False):
        """
        Initialize the FDDTracker class with the given parameters.

        Parameters:
        lag (int): Number of previous detections to keep in memory (memory length).
        bin_w (float): Width of the frequency bin in Hz (1/fs).
        f_lim (list): Limits of the frequency range to track.
        track_metric (str, optional): Metric used to track the modes ('mac' or 'mac_by_sval'). Default is 'mac_by_sval'.
        threshold (float, optional): MAC threshold for detection. Default is 0.85.
        f_scale (float, optional): Width of the Tukey window centered at the expected natural frequency in percentage. Default is 0.2.
        use_avg (bool, optional): Use the average of the shapes in the memory when computing the MAC. Default is False.
        mac_weighted (bool, optional): Use MAC-weighted averaging when computing the average modal parameters. Default is False.
        track_metric_weights (dict, optional): Weights for the weighted average metric. Default is None.
        verbose (bool, optional): Print additional information. Default is False.
        """

        self.chunk_idx = []     # list of chunk indexes (as long as fn_list)
        self.fn_list = {}       # dictionary of lists of natural frequencies of the modes
        self.phi_list = {}      # dictionary of lists of mode shapes of the modes
        self.xi_list = {}       # dictionary of lists of damping ratios of the modes (not yet implemented)
        self.svid_list = {}     # dictionary of lists of singular vectors order of the identified modes
        self.MACs = {}          # dictionary of lists of MAC values of the modes vs. the reference modes, at moment of detection
        self.STM = {'fn': {}, 'phi': {}, 'fn_all': {}, 'phi_all': {}, 'weights': {}}    # short-term memory
        self.num_modes = len(self.fn_list.keys())   # number of tracking modes
        self.lag = lag          # number of previous detections to keep into memory (memory length)
        self.fs = 1/bin_w       # frequency resolution in Hz (1/bin_w) (legacy)
        self.bin_w = bin_w      # width of the frequency bin in Hz (1/fs)
        self.f_lim = f_lim      # limits of the frequency range to track
        self.n_bins = None      # number of frequency bins
        self.track_metric = track_metric  # metric used to track the modes ('mac' or 'mac_by_sval')
        self.track_metric_weights = track_metric_weights  # weights for the weighted average metric
        self.use_avg = use_avg  # use the average of the shapes in the memory when computing the MAC
        self.max_mem_check = None  # percentage of memory elements to check when computing the MAC (deprecated)
        self.threshold = threshold  # MAC threshold for detection
        self.f_scale = f_scale  # width of the Tukey window (alpha=2/3) centered at the expected natural frequency in percentage.
        self.MAC_weighted = mac_weighted    # use MAC-weighted averaging when computing the average modal parameters
        self.verbose = verbose  # print additional information
        self.run_time = {'total': 0, 'update': 0, 'track': 0}  # run time statistics
        self.sval_seed = []     # seed singular values
        self.svec_seed = []     # seed singular vectors
        self.labels_split = []  # labels of the discovered modes
        self.labels_keep_split = []  # labels of the discovered modes kept for tracking
        self.debug = False       # enable debug mode
        self.metric_func = self.metric_options.get(track_metric)
        if self.metric_func is None:
            raise ValueError(
                f'Invalid track_metric: {track_metric}. Please select a valid metric ("mac", "mac_by_sval", "logmac_by_sval" or "weighted_avg").'
                f'If using "weighted_avg", please provide the metric weights in "track_metric_weights" (Use {"mac": 1, "sval": 1} for equal weights.)')

    @staticmethod
    def _discover_modes(sval_seed, svec_seed, mode, min_cluster_size, l_scale, dist_metric_func):
        """
        Discover modes in the seed data using the specified metric and clustering mode.

        Parameters:
        - sval_seed (list of np.ndarray): List of singular values for the seed data.
        - svec_seed (list of np.ndarray): List of singular vectors for the seed data.
        - metric (str): Metric used for distance calculation ('mac' or 'mac+freq').
        - mode (str): Clustering mode to use ('HDBSCAN').
        - min_cluster_size (int): Minimum cluster size for HDBSCAN.
        - l_scale (float): Length scale for the distance metric.

        Returns:
        - labels (np.ndarray): Cluster labels for each data point.
        - probs (np.ndarray): Cluster membership probabilities for each data point.
        - (mac_conc, dist) (tuple of np.ndarray): Tuple containing the MAC matrix and the distance matrix.
        """
        svec_conc = np.concatenate(svec_seed, axis=-1)
        mac_conc = xmacmat_alt(svec_conc[0], svec_conc[0])

        metric = dist_metric_func(sval_seed, l_scale, mac_conc)
        dist = 1 - metric  # Classical distance metric, bounded between 0 and 1
        # dist = - np.log(dist_metric)  # Alternative distance metric, bounded between 0 and +inf
        dist[dist < 0] = 0

        if mode == 'HDBSCAN':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None, metric='precomputed')
            clusterer.fit(dist)
            labels = clusterer.labels_
            probs = clusterer.probabilities_
        else:
            raise ValueError(
                f'Invalid mode: {mode}. Please select a valid mode (e.g., "HDBSCAN"). Other modes are not yet implemented.')
        return labels, probs, (mac_conc, dist)

    @staticmethod
    def _filter_modes(labels, probs, sval_seed, svec_seed, mac_lim, mode_consistency, sval_prominence, mpc_lim, band_reject, metric_func):
        """
        Filters the discovered modes based on various criteria such as MAC values, mode consistency, singular value prominence, MPC values, and proximity to band edges.

        Parameters:
        - labels (np.ndarray): Cluster labels for each data point.
        - probs (np.ndarray): Cluster membership probabilities for each data point.
        - sval_seed (list of np.ndarray): List of singular values for the seed data.
        - svec_seed (list of np.ndarray): List of singular vectors for the seed data.
        - mac_lim (float): Threshold for average MAC values of the cluster.
        - mode_consistency (float): Minimum percentage of samples in which a mode must appear.
        - sval_prominence (float): Threshold for average singular values of the modes.
        - mpc_lim (float): Threshold for average MPC values of the modes.
        - band_reject (float): Threshold for rejecting modes close to the band edges.

        Returns:
        - labels_split (list of np.ndarray): List of arrays with the filtered cluster labels for each dataset.
        - labels_keep_split (list of np.ndarray): List of arrays with the kept cluster labels for each dataset.
        """
        svec_conc = np.concatenate(svec_seed, axis=-1)
        sval_conc = np.concatenate(sval_seed, axis=-1)

        labels_split = np.split(labels, len(sval_seed))
        labels_keep = np.copy(labels)
        labels_remove = []

        if mac_lim:  # remove the modes with average MAC values of the cluster below the threshold
            mac_labels = [xmacmat_alt(svec_conc[0, :, labels == i].T, svec_conc[0, :, labels == i].T) for i
                          in np.unique(labels)]
            mac_labels_avg = [m[m > np.median(m)].mean() for m in mac_labels]
            labels_remove.extend([i for i, m in zip(np.unique(labels), mac_labels_avg) if m < mac_lim])

        if mode_consistency:  # remove the modes that show up in less than a certain percentage of the samples
            for i in np.unique(labels):
                c = 0
                for sp in labels_split:
                    if i in sp:
                        c += 1
                if c < len(sval_seed) * mode_consistency:
                    labels_remove.append(i)

        if sval_prominence:  # remove the modes with average singular values below the threshold
            for i in np.unique(labels):
                lab_sval_avg = np.mean(sval_conc[0, labels == i])
                i_sval_avg = np.mean(np.concatenate([s[0] for s, l in zip(sval_seed, labels_split) if i in l]))
                if lab_sval_avg < sval_prominence * i_sval_avg:
                    labels_remove.append(i)

        if mpc_lim:  # remove the modes with average MPC values below the threshold
            for i in np.unique(labels):
                lab_mpc_avg = np.mean(mpcval(svec_conc[0, :, labels == i].T))
                if lab_mpc_avg < mpc_lim:
                    labels_remove.append(i)

        if band_reject:  # remove the modes that are close to the band edges
            for i in np.unique(labels):
                fn = np.concatenate([np.where(lab == i)[0] for lab in labels_split]).mean()
                if fn < band_reject * len(sval_seed[0][0]) or fn > (1 - band_reject) * len(sval_seed[0][0]):
                    labels_remove.append(i)

        labels_keep[np.isin(labels_keep, labels_remove)] = -1
        labels_keep_split = np.split(labels_keep, len(sval_seed))

        # # For each dataset, keep only the f_bin with the highest average MAC in the cluster
        # for i, lab in enumerate(labels_keep_split):
        #     for j in np.unique(lab):
        #         if j != -1:
        #             idxs = np.where(lab == j)[0]
        #             if len(idxs) > 1:
        #                 avg_MAC_idxs = []
        #                 for idx in idxs:
        #                     svec_i = svec_seed[i][0, :, idx]
        #                     avg_MAC_idxs.append(np.mean(
        #                         [xmacmat_alt(svec_i, svec_seed[i][0, :, idx_])[0] for idx_ in idxs if
        #                          idx_ != idx]))
        #                 labels_keep_split[i][idxs[np.argmax(avg_MAC_idxs)]] = j
        #                 labels_keep_split[i][idxs[idxs != idxs[np.argmax(avg_MAC_idxs)]]] = -1

        # # Alternative strategy: for each dataset, keep only the f_bin with the highest cluster membership score (probability)
        # for i, lab in enumerate(labels_keep_split):
        #     for j in np.unique(lab):
        #         if j != -1:
        #             idxs = np.where(lab == j)[0]
        #             if len(idxs) > 1:
        #                 for idx in idxs:
        #                     if probs[idx] < np.max(probs[idxs]):
        #                         labels_keep_split[i][idx] = -1
        #                     else:
        #                         labels_keep_split[i][idx] = j
        #             idxs = np.where(lab == j)[0]  # Update idxs after filtering by max probability
        #             if len(idxs) > 1:  # if there are still multiple detections in the same f_bin, keep only the idx in the middle
        #                 labels_keep_split[i][idxs] = -1
        #                 labels_keep_split[i][idxs[len(idxs) // 2]] = j

        # # Second alternative strategy: for each dataset, keep only the f_bin with the highest sval value
        # for i, lab in enumerate(labels_keep_split):
        #     for j in np.unique(lab):
        #         if j != -1:
        #             idxs = np.where(lab == j)[0]
        #             if len(idxs) > 1:
        #                 for idx in idxs:
        #                     if sval_seed[i][0, idx] < np.max(sval_seed[i][0, idxs]):
        #                         labels_keep_split[i][idx] = -1
        #                     else:
        #                         labels_keep_split[i][idx] = j
        #             idxs = np.where(lab == j)[0]  # Update idxs after filtering by max sval
        #             if len(idxs) > 1:  # if there are still multiple detections in the same f_bin, keep only the idx in the middle
        #                 labels_keep_split[i][idxs] = -1
        #                 labels_keep_split[i][idxs[len(idxs) // 2]] = j

        # Third alternative strategy: for each dataset, keep only the f_bin where mac_by_sval is the highest
        for i, lab in enumerate(labels_keep_split):
            for j in np.unique(lab):
                if j != -1:
                    idxs = np.where(lab == j)[0]
                    if len(idxs) > 1:
                        mac = np.median(xmacmat_alt(svec_seed[i][0][:, idxs], svec_seed[i][0]), axis=0)
                        metric = metric_func(mac, sval_seed[i][0], np.ones_like(sval_seed[i][0]))
                        max_id = idxs[np.argmax(metric[idxs])]

                        labels_keep_split[i][idxs] = -1
                        labels_keep_split[i][max_id] = j

                    idxs = np.where(lab == j)[0]  # Update idxs after filtering by max sval
                    if len(idxs) > 1:  # if there are still multiple detections in the same f_bin, keep only the idx in the middle
                        print('multiple detections in the same f_bin', idxs)
                        labels_keep_split[i][idxs] = -1
                        labels_keep_split[i][idxs[len(idxs) // 2]] = j

        # change the numeration of the labels to increase according to the (mean) natural frequency
        unique_labels = np.unique(np.concatenate(labels_keep_split))
        valid_labels = unique_labels[unique_labels != -1]
        mean_freqs = np.array([np.mean(np.concatenate([np.where(lab == i)[0] for lab in labels_keep_split])) for i in valid_labels])
        sorted_labels = valid_labels[np.argsort(mean_freqs)]
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
        labels_keep_split = [np.vectorize(label_mapping.get)(lab) for lab in labels_keep_split]
        labels_keep_split = [np.where(lab == None, -1, lab) for lab in labels_keep_split]

        # Combine mappings for `labels_split`, keeping original mappings for existing labels.
        unique_labels_split = np.unique(np.concatenate(labels_split))
        valid_labels_split = unique_labels_split[
            (unique_labels_split != -1) & (~np.isin(unique_labels_split, valid_labels))]
        start_label = max(label_mapping.values()) + 1 if label_mapping else 0
        label_mapping_split = {old_label: new_label for new_label, old_label in
                               zip(range(start_label, start_label + len(valid_labels_split)), valid_labels_split)}
        combined_label_mapping = {**label_mapping, **label_mapping_split}
        labels_split = [np.vectorize(combined_label_mapping.get)(lab) for lab in labels_split]
        labels_split = [np.where(lab == None, -1, lab) for lab in labels_split]

        return labels_split, labels_keep_split

    def _set_bins(self, sval):
        """
        Set the number of frequency bins based on the length of the input singular values (sval).

        Parameters:
        - sval (list of np.ndarray): List of singular values for the current dataset.

        Raises:
        - ValueError: If the calculated number of bins does not match the expected value based on the frequency limits (f_lim) and bin width (bin_w).
        """
        if self.n_bins is None:
            self.n_bins = len(sval[0])
            print(f'Number of bins set to {self.n_bins}.')
        if self.n_bins != (self.f_lim[1] - self.f_lim[0]) / self.bin_w + 1:
            raise ValueError(
                f'Number of bins ({self.n_bins}) is not equal to (f_lim[1]-f_lim[0])/bin_w+1 ({(self.f_lim[1] - self.f_lim[0]) / self.bin_w + 1}).')

    def update_stm(self):   # update the short-term memory
        """
        Update the short-term memory (STM) with the latest detected modes.

        This function updates the STM with the latest detected modes, including their natural frequencies, mode shapes, and weights.
        It also computes the weighted or unweighted average of the modal parameters based on the `MAC_weighted` attribute.

        The STM is updated for each mode in `phi_list`. If a mode is not already in the STM, it initializes the STM entries for that mode.
        If the mode was detected in the latest update, it appends the new values to the STM and keeps only the last `lag` entries.

        The function also updates the runtime statistics for the update process.

        Attributes:
            self.STM (dict): Short-term memory containing 'fn', 'phi', 'fn_all', 'phi_all', and 'weights'.
            self.phi_list (dict): Dictionary of lists of mode shapes of the modes.
            self.fn_list (dict): Dictionary of lists of natural frequencies of the modes.
            self.MACs (dict): Dictionary of lists of MAC values of the modes vs. the reference modes, at moment of detection.
            self.lag (int): Number of previous detections to keep into memory (memory length).
            self.MAC_weighted (bool): Use MAC-weighted averaging when computing the average modal parameters.
            self.run_time (dict): Run time statistics containing 'total' and 'update'.
        """
        t_upd = time.time()
        for mode in self.phi_list.keys():
            if mode not in self.STM['fn']:
                self.STM['fn_all'][mode] = np.array([f for f in self.fn_list[mode] if f is not None])
                self.STM['phi_all'][mode] = np.array([p for p in self.phi_list[mode] if p is not None]).T
                self.STM['weights'][mode] = np.array([w for w in self.MACs[mode] if w is not None])
            else:
                if self.fn_list[mode][-1] is not None:  # if the mode was detected
                    self.STM['fn_all'][mode] = np.append(self.STM['fn_all'][mode], self.fn_list[mode][-1])[-self.lag:]
                    self.STM['phi_all'][mode] = np.append(self.STM['phi_all'][mode], self.phi_list[mode][-1][:, np.newaxis], axis=1)[:,-self.lag:]
                    self.STM['weights'][mode] = np.append(self.STM['weights'][mode], self.MACs[mode][-1])[-self.lag:]

            if self.MAC_weighted:
                self.STM['fn'][mode] = np.average(self.STM['fn_all'][mode], axis=0, weights=self.STM['weights'][mode])
                self.STM['phi'][mode] = np.average(self.STM['phi_all'][mode], axis=-1, weights=self.STM['weights'][mode])
            else:
                self.STM['fn'][mode] = np.average(self.STM['fn_all'][mode], axis=0)
                self.STM['phi'][mode] = np.average(self.STM['phi_all'][mode], axis=-1)

        self.run_time['update'] += time.time() - t_upd
        self.run_time['total'] += time.time() - t_upd

    def initialize(self, sval_seed, svec_seed, chunk_idx=None, mode='HDBSCAN', dist_metric='mac+freq', min_cluster_size=None,
                   l_scale=0.2, band_reject=0.1, sval_prominence=0.5, mode_consistency=0.5, mac_lim=None, mpc_lim=None,
                   return_init=False):  # initialize the tracker
        """
        Initialize the tracker with the given parameters.

        Parameters:
        - sval_seed (list of np.ndarray): List of singular values for the seed data.
        - svec_seed (list of np.ndarray): List of singular vectors for the seed data.
        - chunk_idx (list, optional): List of chunk indexes. Default is None.
        - mode (str, optional): Clustering mode to use ('HDBSCAN' or 'precomputed'). Default is 'HDBSCAN'.
        - metric (str, optional): Metric used for distance calculation ('mac' or 'mac+freq'). Default is 'mac+freq'.
        - min_cluster_size (int, optional): Minimum cluster size for HDBSCAN. Default is None.
        - l_scale (float, optional): Length scale for the distance metric. Default is 0.2.
        - band_reject (float, optional): Threshold for rejecting modes close to the band edges. Default is 0.1.
        - sval_prominence (float, optional): Threshold for average singular values of the modes. Default is 0.5.
        - mode_consistency (float, optional): Minimum percentage of samples in which a mode must appear. Default is 0.5.
        - mac_lim (float, optional): Threshold for average MAC values of the cluster. Default is None.
        - mpc_lim (float, optional): Threshold for average MPC values of the modes. Default is None.
        - return_init (bool, optional): If True, return the initial labels, probabilities, and distances. Default is False.

        Raises:
        - ValueError: If the length of sval_seed is greater than lag.
        - ValueError: If the number of bins does not match the expected value based on the frequency limits and bin width.

        Returns:
        - If return_init is True, returns a tuple (labels, probs, distances).
        """
        # if the modes are precomputed, sval_seed is interpreted as the precomputed natural frequencies and svec_seed
        # as the precomputed mode shapes counterparts
        self.sval_seed = sval_seed
        self.svec_seed = svec_seed

        # discover the modes in the seed data or use the provided modes (mode='precomputed')
        if mode == 'precomputed':  # if the modes are precomputed, directly assign them to the tracker
            self.fn_list = {k: [f] for k, f in enumerate(sval_seed[0])}
            self.phi_list = {k: [p] for k, p in enumerate(svec_seed[0].T)}
            self.MACs = {k: [None] for k, _ in enumerate(sval_seed[0])}
            self.svid_list = {k: [None] for k, _ in enumerate(sval_seed[0])}
        else:
            self.dist_metric_func = self.dist_metric_options.get(dist_metric)
            if self.dist_metric_func is None:
                raise ValueError(
                    f'Invalid metric: {dist_metric}. Please select a valid metric ("mac" or "mac+freq"). Other metrics are not yet implemented.')

            if len(sval_seed) > self.lag:
                raise ValueError(
                    f'Length of sval_seed ({len(sval_seed)}) is greater than lag ({self.lag}). Please adjust lag or sval_seed.')

            if chunk_idx is None:
                self.chunk_idx.extend(np.arange(len(sval_seed)))
            else:
                self.chunk_idx.extend(chunk_idx)

            self._set_bins(sval_seed[0])

            if min_cluster_size is None:
                min_cluster_size = int(len(sval_seed) * mode_consistency * self.n_bins * 0.01)  # default value

            labels, probs, distances = self._discover_modes(self.sval_seed, self.svec_seed, mode, min_cluster_size,
                                                            l_scale, self.dist_metric_func)
            labels[probs < 0.5] = -1

            self.labels_split, self.labels_keep_split = self._filter_modes(labels, probs, sval_seed, svec_seed,
                                                                           mac_lim, mode_consistency, sval_prominence,
                                                                           mpc_lim, band_reject, self.metric_func)

            for k, i in enumerate(np.unique(self.labels_keep_split)[np.unique(self.labels_keep_split) != -1]):
                self.fn_list[k] = []
                self.phi_list[k] = []
                self.MACs[k] = []
                self.svid_list[k] = []
                for j, lbl in enumerate(self.labels_keep_split):
                    if i in lbl:
                        self.fn_list[k].append(np.where(lbl == i)[0][0] * self.bin_w)
                        self.phi_list[k].append(svec_seed[j][0, :, np.where(lbl == i)[0]][0])
                        self.MACs[k].append(None)
                        self.svid_list[k].append(0)
                    else:
                        self.fn_list[k].append(None)
                        self.phi_list[k].append(None)
                        self.MACs[k].append(None)
                        self.svid_list[k].append(None)

            if self.verbose:
                print(f'Minimum cluster size: {min_cluster_size}\n')
                self.plot_discovery()

            if self.debug:
                self.plot_initdistances((distances[0], (1-distances[1])/distances[0], distances[1]), 3)

        self.num_modes = len(self.fn_list.keys())
        self.update_stm()

        if self.verbose:
            print(f'Initialization completed in {self.run_time["total"]:.2f}s.\nTracked modes:\n')
            print(f'{"Mode":<5}{"Frequency [Hz]":<20}{"MPC":<20}')
            for i in self.STM['fn'].keys():
                print(f'{i:<5}{self.STM["fn"][i]:<20.2f}{np.mean(mpcval(self.STM["phi_all"][i])):<20.2f}')

        if return_init:
            return labels, probs, distances

    def track(self, sval, svec, chunk_name=None):
        """
        Track the modes in the given dataset using the provided singular values and vectors.

        Parameters:
        - sval (np.ndarray): Singular values for the current dataset.
        - svec (np.ndarray): Singular vectors for the current dataset.
        - chunk_name (str, optional): Name of the current data chunk. Default is None.

        This function updates the tracking of modes by computing the MAC values and applying the specified tracking metric.
        It also updates the short-term memory (STM) with the latest detected modes and their parameters.
        """
        self._set_bins(sval)

        t_trk = time.time()
        if chunk_name is None:
            self.chunk_idx.append(self.chunk_idx[-1] + 1)
        else:
            self.chunk_idx.append(chunk_name)

        for mode in self.phi_list.keys():
            start = max(0, round(self.STM['fn'][mode] // self.bin_w * (1 - self.f_scale)))
            end = min(round(self.STM['fn'][mode] // self.bin_w * (1 + self.f_scale)), self.n_bins)

            tk_win = sp.signal.windows.tukey(end-start, alpha=0.5)
            f_win = np.zeros(sval.shape[-1])
            f_win[start: end] = tk_win
            f_win /= np.max(f_win)

            mac = np.zeros_like(sval)
            if self.use_avg:
                for i, svec_i in enumerate(svec):
                    mac[i, start: end] = xmacmat_alt(self.STM['phi'][mode], svec_i[:, start: end])[0]
            else:
                # idxs = np.random.choice(len(self.STM['phi_all'][mode]), min(int(self.max_mem_check*self.lag), len(self.STM['phi_all'][mode])), replace=False)
                for i, svec_i in enumerate(svec):
                    # mac[i, start: end] = np.average([xmacmat_alt(phi_i, svec_i[:, start: end])[0] for phi_i in np.array(self.STM['phi_all'][mode])[idxs]], axis=0)
                    mac[i, start: end] = np.median(xmacmat_alt(self.STM['phi_all'][mode], svec_i[:, start: end]), axis=0)

            metric_options = {
                'mac': lambda mac, sval, f_win: mac * f_win,
                'mac_by_sval': lambda mac, sval, f_win: mac * sval * f_win,
                'logmac_by_sval': lambda mac, sval, f_win: -1 / np.log(mac) * sval * f_win,  # experimental
                'weighted_avg': lambda mac, sval, f_win:
                self.track_metric_weights['mac'] * mac * f_win + self.track_metric_weights['sval'] * (sval * f_win) /
                (sval * f_win).max(axis=-1, keepdims=True) if self.track_metric_weights else None}

            metric_func = metric_options.get(self.track_metric)
            if metric_func is None:
                raise ValueError(
                    f'Invalid track_metric: {self.track_metric}. Please select a valid metric ("mac", "mac_by_sval", "logmac_by_sval" or "weighted_avg").'
                    f'If using "weighted_avg", please provide the metric weights in "track_metric_weights" (Use {"mac": 1, "sval": 1} for equal weights.)')

            metric = metric_func(mac, sval, f_win)

            max_mac_id = np.unravel_index(np.argmax(mac*f_win), mac.shape)
            max_mac = mac[max_mac_id[0], max_mac_id[1]]
            max_id = np.argmax(metric[max_mac_id[0], :])

            if self.debug:
                print(f'start: {start}, end: {end}, fn_avg: {self.STM["fn"][mode]:.4f} | max_mac: {max_mac:.4f}, max_mac_id: {max_mac_id}, max_id: {max_id}')
                fig, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
                for i, data in enumerate([sval.T, mac.T, metric.T]):
                    axs[i].plot(data)
                    axs[i].grid()
                axs[1].plot(f_win)
                axs[1].scatter(max_mac_id[1], mac[max_mac_id[0], max_mac_id[1]], color='lime', s=40, edgecolors='b',
                               linewidths=.5)
                axs[2].plot(metric[max_mac_id[0], :], color='red', linewidth=6, alpha=.3)
                axs[2].scatter(max_id, metric[max_mac_id[0], max_id], color='red', s=40, edgecolors='b', linewidths=.5)
                for ax in axs:
                    ax.axvline(self.STM['fn'][mode] // self.bin_w, color='red', linestyle='--')
                plt.xlim(0, sval.shape[-1])
                plt.tight_layout()

            if max_mac > self.threshold:
                # self.fn_list[mode].append(max_id/self.fs)
                self.fn_list[mode].append(max_id*self.bin_w)
                # self.phi_list[mode].append(svec[max_mac_id[0], :, max_id])
                self.phi_list[mode].append(svec[max_mac_id[0], :, max_mac_id[1]])
                self.MACs[mode].append(mac[max_mac_id[0], max_mac_id[1]])
                self.svid_list[mode].append(max_mac_id[0])
            else:
                self.fn_list[mode].append(None)
                self.phi_list[mode].append(None)
                self.MACs[mode].append(None)
                self.svid_list[mode].append(None)

        self.run_time['track'] += time.time() - t_trk
        self.run_time['total'] += time.time() - t_trk
        self.update_stm()

    def plot_discovery(self, save_name=None):
        """
        Plots the discovered modes and the tracked modes.

        Parameters:
        - save_name (str, optional): The name of the file to save the plot, if provided. Default is None.

        The function creates a plot with two subplots:
        - The first subplot shows the discovered modes (after clustering).
        - The second subplot shows the tracked modes (after filtering).
        """
        tab10_no_gray = colors.ListedColormap(plt.get_cmap('tab10').colors[:-3] + plt.get_cmap('tab10').colors[-2:])
        cmap = loop_cmap_listed(tab10_no_gray, tab10_no_gray.N)  # Now this returns a valid ListedColormap

        sval_data = np.array([s[0] for s in self.sval_seed])
        label_array = np.full_like(sval_data.T, fill_value=-1, dtype=int)
        for i, sv_lab in enumerate(self.labels_split):
            label_array[:, i] = sv_lab
        label_masked = np.ma.masked_where(label_array == -1, label_array)

        fig, axs = plt.subplots(1, 2, figsize=(6, 9), sharey=True)
        ax = axs[0]
        ax.imshow(sval_data.T, aspect='auto', origin='lower',
                  extent=[0, len(self.sval_seed), self.f_lim[0], self.f_lim[1]], cmap='binary', alpha=1,
                  zorder=0,
                  norm=colors.LogNorm())
        ax.imshow(label_masked, aspect='auto', origin='lower', interpolation='nearest',
                  extent=[0, len(self.sval_seed), self.f_lim[0], self.f_lim[1]], cmap=cmap,
                  alpha=0.7, zorder=10, vmax=len(cmap.colors)-1)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Discovered')

        ax = axs[1]
        ax.imshow(sval_data.T, aspect='auto', origin='lower',
                  extent=[0, len(self.sval_seed), self.f_lim[0], self.f_lim[1]], cmap='binary', alpha=1,
                  norm=colors.LogNorm())
        for i, fn in self.fn_list.items():
            plt.plot(np.arange(len(fn)) + .5, fn, color=cmap(i), marker='o', markersize=4, linewidth=2, alpha=1)
        ax.set_title('Tracked')
        for ax in axs:
            ax.set_xticks(np.arange(.5, len(self.sval_seed) + .5, 2))
            ax.set_xticklabels(np.arange(1, len(self.sval_seed) + 1, 2))
            ax.set_xlabel('Dataset Index')

        for i in self.fn_list.keys():
            if i != -1:
                ax.scatter([], [], color=cmap(i), label=f'Mode {i + 1}', s=40, marker='s')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name, dpi=300)
        
    def plot_tracking(self, sval_all=None, xticks_spacing=100, save_name=None):
        """
        Plots the tracking results of the modes over the datasets.

        Parameters:
        - sval_all (list of np.ndarray, optional): List of singular values for all datasets. If provided, the singular values
          will be plotted as a background image. Default is None.
        - xticks_spacing (int, optional): Spacing between x-ticks on the plot. Default is 100.
        - save_name (str, optional): The name of the file to save the plot, if provided. Default is None.

        The function creates a plot showing the tracked modes over the datasets. Each mode is represented by a different color.
        The function also plots the singular values as a background image if `sval_all` is provided.
        """
        plt.figure(figsize=(len(self.fn_list[0]) // 400 + 6, 12))
        if sval_all is not None:
            plt.imshow(np.array([s[0] / s[0].max() for s in sval_all]).T, aspect='auto', origin='lower',
                       extent=[0, len(sval_all), 0, self.sval_seed[0].shape[1] * self.bin_w], cmap='binary', alpha=1,
                       norm=colors.LogNorm())
        else:
            plt.axes().set_facecolor('gray')

        tab10_no_gray = colors.ListedColormap(plt.get_cmap('tab10').colors[:-3] + plt.get_cmap('tab10').colors[-2:])
        cmap = loop_cmap(tab10_no_gray, tab10_no_gray.N)

        for i in list(self.fn_list.keys())[::-1]:
            if i != -1:
                plt.scatter([], [], color=cmap(i), label=f'Mode {i + 1}', s=24)
        plt.legend(scatterpoints=5, loc='upper left', bbox_to_anchor=(1.09, 1), handletextpad=0.5, handlelength=2)

        for i, (fn, sv_lab) in enumerate(zip(self.fn_list.values(), self.svid_list.values())):
            plt.scatter(range(len(fn)), np.where(np.where(np.array(sv_lab) == 0, True, False), np.array(fn), None),
                        label=f'Mode {i}', s=12, color=cmap(i), linewidths=0, alpha=.8)
            plt.scatter(range(len(fn)), np.where(np.where(np.array(sv_lab) == 1, True, False), np.array(fn), None),
                        label=f'Mode {i}', s=12, color=cmap(i), edgecolors='w', linewidths=1, alpha=1)
            plt.scatter(range(len(fn)), np.where(np.where(np.array(sv_lab) == 2, True, False), np.array(fn), None),
                        label=f'Mode {i}', s=12, color=cmap(i), edgecolors='k', linewidths=1, alpha=1)

        plt.xticks(list(range(0, len(self.chunk_idx), int(xticks_spacing))),
                   [self.chunk_idx[i] for i in range(0, len(self.chunk_idx), int(xticks_spacing))])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Dataset Index')
        plt.grid()
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)

    def plot_initdistances(self, distances, n, save_name=None):
        """
        Plots the initial distances used for mode discovery.

        Parameters:
        - distances (tuple of np.ndarray): A tuple containing the distance matrices to be plotted.
        - n (int): The number of datasets to plot.
        - save_name (str, optional): The name of the file to save the plot, if provided. Default is None.

        The function creates a plot with three subplots showing the MAC Metric, k_SE Metric, and D Matrix.
        """
        gs_kw = dict(width_ratios=[1, 1, 1, .04], height_ratios=[1])  # use last axis for colorbar
        fig, axs = plt.subplots(1, 4, figsize=(9, 3.5), gridspec_kw=gs_kw)
        titles = ['MAC Metric', r'$k_{SE}$ Metric', 'D Matrix']
        for i, ax in enumerate(axs[:-1]):
            ax.imshow(distances[i][:self.n_bins * n, :self.n_bins * n], origin='lower', cmap='Blues')
            ax.set_title(titles[i])
            # ax.set_xticks(list(range(0, (self.n_bins - 1) * n, self.fs * 25)))
            # ax.set_xticklabels(list(range(0, self.n_bins // self.fs, 25)) * n)
            # ax.set_yticks(list(range(0, (self.n_bins - 1) * n, self.fs * 25)))
            # if i == 0:
                # ax.set_yticklabels(list(range(0, self.n_bins // self.fs, 25)) * n)
                # ax.set_ylabel('Frequency [Hz]')
            if i == 2:
                ax.imshow(-distances[i][:self.n_bins * n, :self.n_bins * n], origin='lower', cmap='Blues')
            else:
                ax.imshow(distances[i][:self.n_bins * n, :self.n_bins * n], origin='lower', cmap='Blues')
            ax.set_title(titles[i])
            if i == 0:
                ax.set_ylabel('Frequency [Hz]')
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('Frequency [Hz]')
            for j in range(1, n):
                ax.axvline(j * self.n_bins, color='red', linestyle='--')
                ax.axhline(j * self.n_bins, color='red', linestyle='--')
        cbar = fig.colorbar(axs[0].images[0], cax=axs[3], orientation='vertical')
        axs[3].set_aspect(20)
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)


########################################################################################################################
# Example usage

# filename = 'F:/norsenga_si/norsenga_si.hdf5'
# seed_len = 18
# start_datetime = '2024_03_01/00/00'
# end_datetime = '2024_06_30/24/40'
#
# with h5py.File(filename, 'r') as f:
#     chunk_names = [f'{d}/{h}/{m}' for d in f for h in f[d] for m in f[d][h]]
#     chunk_names = chunk_names[chunk_names.index(start_datetime):chunk_names.index(end_datetime) + 1]
#
#     sval_seed, svec_seed = zip(*[(f[chunk]['sval'][..., :1001], f[chunk]['svec'][..., :1001].swapaxes(0, 1))
#                                  for chunk in chunk_names[:seed_len]])
#     sval_all = list(sval_seed)
#
#     # Initialize the tracker
#     tracker = FDDTracker(lag=504, bin_w=0.01, f_lim=[0, 10], track_metric='logmac_by_sval', threshold=0.85, mac_weighted=False,
#                          verbose=True, f_scale=0.15, use_avg=False)
#     tracker.initialize(sval_seed=sval_seed, svec_seed=svec_seed, chunk_idx=chunk_names[:seed_len], mode='HDBSCAN',
#                        min_cluster_size=45, metric='mac+freq', band_reject=0.1, sval_prominence=0.5, mode_consistency=0.5,
#                        mac_lim=None, mpc_lim=0.6, l_scale=0.2, return_init=False)
#
#     td = time.time()
#     for i, chunk in enumerate(chunk_names[seed_len:]):
#         sval, svec = f[chunk]['sval'][..., :1001], f[chunk]['svec'][..., :1001].swapaxes(0, 1)
#         sval_all.append(sval)
#
#         # Track the modes
#         tracker.track(sval, svec, chunk)
#
#         if i % 72 == 0:
#             print(f'Processed day: {chunk.split("/")[0]} | {time.time() - td:.2f} s')
#             td = time.time()
#
# with open('tracker_norsenga_mac_2.1_final_logmac.pkl', 'wb') as f:
#     pickle.dump(tracker, f)
#
# load the tracker class from a file with pickle
# with open('tracker_norsenga_mac_2.1_final_logmac.pkl', 'rb') as f:
#     tracker = pickle.load(f)

