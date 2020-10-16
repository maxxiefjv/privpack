"""
DataGenerate module is used to generate data for theoretical experiments.
Currently the following types of data are supported:

- Gaussian Data: Completely Uncorrelated, from PPAN Paper [1]
- Bivariate Binary Data: Completely Uncorrelated, Completely Correlated, From a random distribution


References
==========
[1] Tripathy, Ardhendu, Ye Wang, and Prakash Ishwar (2017). “Privacy-Preserving Ad-versarial Networks.

Module Contents
----------------
"""

import random
import math
import numpy as np
import torch

from scipy.stats import multivariate_normal

class DataGenerator():
    """
    DataGenerator is a simple class used to generate different types of data mostly used for experiments.

    Includes:

    - Gaussian Data: Completely Uncorrelated, from PPAN Paper [1]
    - Bivariate Binary Data: Completely Uncorrelated, Completely Correlated, From a random distribution


    [1] Tripathy, Ardhendu, Ye Wang, and Prakash Ishwar (2017). “Privacy-Preserving Ad-versarial Networks.
    """

    def get_completely_correlated_dist():
        '''
        Return numpy array with: 2x2 completely correlated distrubion + accumulated distribution
        '''

        dist = np.array([
            [0.5, 0],
            [0, 0.5]
        ]).flatten()

        accumulated_norm_dist = np.array(
            [sum(dist[:i + 1]) for i in range(dist.size)])
        return (dist.reshape(2, 2), accumulated_norm_dist)

    def get_completely_uncorrelated_dist():
        '''
        Return numpy array with: 2x2 completely uncorrelated distrubion + accumulated distribution
        '''

        dist = np.array([
            [0.25, 0.25],
            [0.25, 0.25]
        ]).flatten()

        accumulated_norm_dist = np.array(
            [sum(dist[:i + 1]) for i in range(dist.size)])
        return (dist.reshape(2, 2), accumulated_norm_dist)

    def random_binary_dist():
        '''
        Return numpy array with: 2x2 random distrubion + accumulated distribution
        '''

        np.random.seed(0)
        dist = np.random.uniform(0, 1, size=(2, 2))
        norm_dist = (dist / dist.sum()).flatten()
        accumulated_norm_dist = np.array(
            [sum(norm_dist[:i + 1]) for i in range(norm_dist.size)])

        return (norm_dist.reshape(2, 2), accumulated_norm_dist)

    def generate_binary_data(amount, acc_dist):
        '''
        Returns pytorch tensor with @amount number of entries.
        '''

        torch.manual_seed(0)
        data = np.random.uniform(0, 1, size=(amount))
        return torch.Tensor([(int(i / 2), i % 2) for i in np.digitize(data, acc_dist)])

    def generate_gauss_mixture_data(mu, cov, seed=None, num_samples=10000, train_ratio=0.8):
        '''
        Returns pytorch tensor with @amount number of entries.
        '''

        np.random.seed(seed)
        mixture_data = np.random.multivariate_normal(mu, cov, num_samples)
        num_train_samples = int(num_samples * train_ratio)
        gm_train_data = torch.Tensor(mixture_data[:num_train_samples])
        gm_test_data = torch.Tensor(mixture_data[num_train_samples:])

        return (gm_train_data, gm_test_data)

    def get_ppan_distribution_params(x_dim, y_dim):

        correlation_coefficients = torch.Tensor([0.47, 0.24, 0.85, 0.07, 0.66])[:x_dim]
        cov_top = torch.cat(
            (torch.eye(x_dim), torch.diag(correlation_coefficients)), dim=1)
        cov_bot = torch.cat(
            (torch.diag(correlation_coefficients), torch.eye(x_dim)), dim=1)
        cov = torch.cat((cov_top, cov_bot))
        mu = torch.zeros(x_dim * 2)
        return (mu, cov)

    def get_completely_uncorrelated_distribution_params(x_dim, y_dim):

        cov = torch.eye(x_dim + y_dim)
        mu = torch.zeros(x_dim + y_dim)
        return (mu, cov)
