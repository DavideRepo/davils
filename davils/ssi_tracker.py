import numpy as np
import scipy as sp
from .dynamics import mpcval
from .dynamics import xmacmat_alt
from .general import exp_dist
from .general import loop_cmap_listed
from .general import loop_cmap

import time

import h5py
import hdbscan

from matplotlib import pyplot as plt
from matplotlib import colors

from datetime import datetime, timedelta


class SSITracker:
    """
    Class for tracking modes in Structural Health Monitoring (SHM) data using streaming clusters of
    Stochastic Subspace Identification (SSI) output poles.
    """

    def __init__(self, consistence=0.5, f_lim=[0,10], mem_len=100, l_scale=0.2, metric='mac+freq'):
        self.consistence = consistence
        self.f_lim = f_lim
        self.mem_len = mem_len
        self.l_scale = l_scale
        self.metric = metric

    @staticmethod
    def _get_distances(data1, data2=None, metric='mac+freq', l_scale=0.2, condensed=False, plot=False):
        """
        Compute the distances between the data1 and data2 using the specified metric.
        """

        f1, phi1 = data1
        if data2:
            f2, phi2 = data2
        else:
            f2, phi2 = f1, phi1

        mac = xmacmat_alt(phi1.T, phi2.T)

        dist_metric_options = {
            'mac+freq': lambda: exp_dist(f1, f2, var=1, l_scale=l_scale, squared=True).T * mac,
            'freq': lambda: exp_dist(f1, f2, var=1, l_scale=l_scale, squared=True).T,
            'mac': lambda: mac}

        dist_metric_func = dist_metric_options.get(metric)
        if dist_metric_func is None:
            raise ValueError(
                f'Invalid metric: {metric}. Please select a valid metric ("mac", "freq", "mac+freq"). Other metrics are not yet implemented.')

        dist_metric = dist_metric_func()
        dist = 1 - dist_metric  # Classical distance metric, bounded between 0 and 1
        # dist = - np.log(dist_metric)  # Alternative distance metric, bounded between 0 and +inf
        dist[dist < 0] = 0

        if plot:
            freq_dist = exp_dist(f1, f2, var=1, l_scale=l_scale, squared=True)

            lim = np.array([20, 28, 26, 8])
            tick_lines = np.cumsum(lim)[:-1] - 0.5  # Keep tick lines unchanged
            tick_labels = np.cumsum(lim) - lim / 2  # Shift labels to the middle

            fig, axs = plt.subplots(1, 3, figsize=(6, 3), sharey=True)
            for ax, data, title in zip(axs, [freq_dist, mac, 1-dist], [r'K$_{SE}$ matrix', 'MAC matrix', 'Distance matrix']):
                ax.set_title(title)
                im = ax.imshow(data, cmap='PuBuGn')
                ax.set_xlim([0, np.sum(lim)])
                ax.set_ylim([0, np.sum(lim)])
                ax.set_xticks(tick_lines)
                ax.set_xticklabels([])
                ax.set_yticks(tick_lines)
                ax.set_yticklabels([])
                for i, label in enumerate([f'seg. {i + 1}' for i in range(len(lim) - 1)]):
                    ax.text(tick_labels[i], -2, label, ha='center', va='top')
                    if title == r'K$_{SE}$ matrix':
                        ax.text(-2, tick_labels[i], label, ha='right', va='center', rotation=90)
                for line in tick_lines:
                    ax.axvline(line, color='r', linestyle='--')
                    ax.axhline(line, color='r', linestyle='--')
            # plt.tight_layout(rect=[0, 0, 0.9, 1])
            # cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.5])
            # fig.colorbar(im, cax=cbar_ax)
            plt.tight_layout(rect=[0, 0.12, 1, 1])
            cbar_ax = fig.add_axes([0.3, 0.12, 0.4, 0.02])
            fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            plt.savefig('distances.png', dpi=300)

        if condensed:
            if data2 is not None:
                raise ValueError("Compact form is only supported for square distance matrices (data2 must be None).")
            dist = sp.spatial.distance.squareform(dist, checks=False)  # Convert to compact form

        return dist

    @staticmethod
    def _preprocess_poles(lam, phi, f_lim):
        """
        Preprocess the poles (lambdas, phis) by filtering out the poles outside the frequency limits.
        """

        if isinstance(lam[0], (float, complex)):
            lam = list([lam])
            phi = list([phi])

        fd = [np.abs(np.imag(lam_val)) / 2 / np.pi for lam_val in lam]

        filtered_lam = []
        filtered_phi = []
        filtered_fd = []

        for ff, ll, pp in zip(fd, lam, phi):
            valid_idx = np.where((ff >= f_lim[0]) & (ff <= f_lim[1]))[0]
            filtered_lam.append(ll[valid_idx])
            filtered_phi.append(pp[valid_idx])
            filtered_fd.append(ff[valid_idx])

        return filtered_fd, filtered_lam, filtered_phi

    def update_stm(self, new_labels, new_fd, new_lam, new_phi, new_dist):
        """
        Update the short-term memory (STM) with the new detections. Compute the distances.
        """

        for mode in self.stm.keys():
            if mode in np.unique(new_labels[new_labels != -1]):
                # Update the STM with the new detections
                self.stm[mode]['fd'] = np.concatenate([self.stm[mode]['fd'][-self.mem_len:], new_fd[new_labels == mode]])
                self.stm[mode]['lam'] = np.concatenate([self.stm[mode]['lam'][-self.mem_len:], new_lam[new_labels == mode]])
                self.stm[mode]['phi'] = np.concatenate([self.stm[mode]['phi'][-self.mem_len:], new_phi[new_labels == mode]])

                # Update the distances in the STM
                full_dist = np.pad(sp.spatial.distance.squareform(self.stm[mode]['dist']), pad_width=((0, 1), (0, 1)), mode='constant')
                new_pdist = new_dist[mode]['dist'][:, new_labels == mode].ravel()
                full_dist[:-1, -1] = full_dist[-1, :-1] = new_pdist
                full_dist = full_dist[-len(self.stm[mode]['fd']):, -len(self.stm[mode]['fd']):]
                self.stm[mode]['dist'] = sp.spatial.distance.squareform(full_dist, checks=False)
                # self.stm[mode]['dist'] = self._get_distances((self.stm[mode]['fd'], self.stm[mode]['phi']), metric=self.metric, l_scale=self.l_scale, condensed=True)
                self.stm[mode]['dist_mean'].append(np.mean(self.stm[mode]['dist'], axis=0))
                self.stm[mode]['dist_quantiles'] = np.concatenate([self.stm[mode]['dist_quantiles'], np.quantile(self.stm[mode]['dist'], q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])[..., np.newaxis]], axis=1)
                self.stm[mode]['age'] = max(self.stm[mode]['age'] - 10, 0)
            else:
                # Increase the age of the modes that were not detected
                self.stm[mode]['age'] += 1

    def initialize_clusters(self, lam_seed, phi_seed, min_cluster_size, min_prob=0.5, idx=None):
        """
        Initialize the clusters using HDBSCAN clustering on the seed data.
        """

        # Preprocess the seed data
        fd_seed, lam_seed, phi_seed = self._preprocess_poles(lam_seed, phi_seed, self.f_lim)

        self.data = {'lam': lam_seed, 'phi': phi_seed, 'fd': fd_seed}
        self.seed_len = len(lam_seed)
        self.min_cluster_size = min_cluster_size

        # Concatenate the seed data
        fd_conc = np.concatenate(self.data['fd'], axis=0)
        lam_conc = np.concatenate(self.data['lam'], axis=0)
        phi_conc = np.concatenate(self.data['phi'], axis=0)

        # Compute the distances between the seed data and cluster using the specified metric
        dist = self._get_distances((fd_conc, phi_conc), metric=self.metric, l_scale=self.l_scale, plot=False)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None, metric='precomputed')
        clusterer.fit(dist)
        self.clusterer = clusterer

        labels = clusterer.labels_
        probs = clusterer.probabilities_

        # Sort the labels by the average frequency of the modes
        unique_labels = np.unique(labels[labels != -1])  # Exclude noise points (-1)
        avg_freqs = np.array([np.mean(fd_conc[np.where(labels == mode)]) for mode in unique_labels])
        sorted_labels = unique_labels[np.argsort(avg_freqs)]
        label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
        labels = np.array([label_map[label] if label in label_map else -1 for label in labels])

        # Filter out labels with low probability and low consistence
        if min_prob:
            labels[probs < min_prob] = -1

        valid_labels, counts = np.unique(labels, return_counts=True)
        self.valid_labels = valid_labels[counts >= self.consistence * self.seed_len]

        if np.any(counts < self.consistence * self.seed_len):
            print(f'Rejected labels: {valid_labels[counts < self.consistence * self.seed_len]}')

        # Initialize the short-term memory (STM) with the seed data
        self.stm = {mode: {'fd': fd_conc[labels == mode][-self.mem_len:], 'lam': lam_conc[labels == mode][-self.mem_len:],
                           'phi': phi_conc[labels == mode][-self.mem_len:]} for mode in self.valid_labels[self.valid_labels != -1]}

        for mode in self.stm.keys():
            self.stm[mode]['dist'] = self._get_distances((self.stm[mode]['fd'], self.stm[mode]['phi']), metric=self.metric, l_scale=self.l_scale, condensed=True)
            self.stm[mode]['dist_quantiles'] = np.quantile(self.stm[mode]['dist'], q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])[..., np.newaxis]
            self.stm[mode]['age'] = 0
            self.stm[mode]['dist_mean'] = [np.mean(self.stm[mode]['dist'], axis=0)]
            self.stm[mode]['threshold'] = []

        # Add the seed data to the labels and probabilities
        self.data['labels'] = np.split(labels, np.cumsum([len(l) for l in self.data['fd']])[:-1])
        self.data['probs'] = np.split(probs, np.cumsum([len(l) for l in self.data['fd']])[:-1])
        if idx:
            self.data['idx'] = idx if isinstance(idx, list) else list(idx)
        else:
            self.data['idx'] = list(range(self.seed_len))

    def track(self, new_lam, new_phi, new_idx=None, q=0.8, d_max=0.5, d_min=0.2, averaging='mean', eld_rate=1.001):
        """
        Track the modes in the given dataset (lambdas, phis) using hdbscan clusters and assigning new detections by
        similarity measure.
        """

        # Preprocess the new data
        new_fd, new_lam, new_phi = (out[0] for out in self._preprocess_poles(new_lam, new_phi, self.f_lim))

        # Compute the distances between the new data and the existing modes
        new_dist = {}
        for mode in self.stm.keys():
            dist = self._get_distances((self.stm[mode]['fd'], self.stm[mode]['phi']), data2=(new_fd, new_phi),
                                       metric=self.metric, l_scale=self.l_scale)
            new_dist[mode] = {'dist': dist, 'mean': np.mean(dist, axis=0), 'median': np.median(dist, axis=0)}

        # Gather all valid pairs (mode, detection_index, distance) in a list
        valid_pairs = []
        for mode in self.stm.keys():
            mode_threshold = np.min([np.max([np.quantile(self.stm[mode]['dist'], q=q)*eld_rate**self.stm[mode]['age'], d_min]), d_max])
            self.stm[mode]['threshold'].append(mode_threshold)
            for i in range(len(new_fd)):
                dist_val = new_dist[mode][averaging][i]
                if dist_val < mode_threshold:
                    valid_pairs.append((mode, i, dist_val))

        # Sort by distance so that the smallest-dist pairs get assigned first
        valid_pairs.sort(key=lambda x: x[2])

        # Store the final assignments in new_labels (length = number of detections).
        new_labels = -1 * np.ones(len(new_fd), dtype=int)

        # Keep track of which modes and detections have already been used
        assigned_modes = set()
        assigned_detections = set()

        # Greedy assignment over valid pairs
        for (mode, i, dist_val) in valid_pairs:
            if (mode not in assigned_modes) and (i not in assigned_detections):
                new_labels[i] = mode
                assigned_modes.add(mode)
                assigned_detections.add(i)

        # Add the new detections to the labels and probabilities
        if new_idx:
            self.data['idx'].append(new_idx)
        else:
            self.data['idx'].append(len(self.data['idx']) + 1)
        self.data['fd'].append(new_fd)
        self.data['lam'].append(new_lam)
        self.data['phi'].append(new_phi)
        self.data['labels'].append(new_labels)
        self.data['probs'].append(np.ones_like(new_labels))

        self.update_stm(new_labels, new_fd, new_lam, new_phi, new_dist)