import random
import math
import numpy as np
import torch

from scipy.stats import multivariate_normal

'''
  Return numpy array with: 2x2 completely correlated distrubion + accumulated distribution
'''
def get_completely_correlated_dist():
    print('correlated')
    dist = np.array([
        [0.5, 0],
        [0, 0.5]
    ]).flatten()

    accumulated_norm_dist = np.array(
        [sum(dist[:i + 1]) for i in range(dist.size)])
    return (dist.reshape(2, 2), accumulated_norm_dist)


'''
  Return numpy array with: 2x2 completely uncorrelated distrubion + accumulated distribution
'''
def get_completely_uncorrelated_dist():
    print('uncorrelated')
    dist = np.array([
        [0.25, 0.25],
        [0.25, 0.25]
    ]).flatten()

    accumulated_norm_dist = np.array(
        [sum(dist[:i + 1]) for i in range(dist.size)])
    return (dist.reshape(2, 2), accumulated_norm_dist)


'''
  Return numpy array with: 2x2 random distrubion + accumulated distribution
'''
def random_binary_dist():
    print('random')
    np.random.seed(0)
    dist = np.random.uniform(0, 1, size=(2, 2))
    norm_dist = (dist / dist.sum()).flatten()
    accumulated_norm_dist = np.array(
        [sum(norm_dist[:i + 1]) for i in range(norm_dist.size)])

    return (norm_dist.reshape(2, 2), accumulated_norm_dist)


'''
  Returns pytorch tensor with @amount number of entries.
'''
def generate_binary_data(amount, acc_dist):
    torch.manual_seed(0)
    data = np.random.uniform(0, 1, size=(amount))
    return torch.Tensor([(int(i / 2), i % 2) for i in np.digitize(data, acc_dist)])


'''
  Returns pytorch tensor with @amount number of entries.
'''
def generate_gauss_mixture_data(mu, cov):
    np.random.seed(0)
    mixture_data = np.random.multivariate_normal(mu, cov, 10000)
    gm_train_data = torch.Tensor(mixture_data[:8000])
    gm_test_data = torch.Tensor(mixture_data[8000:])

    return (gm_train_data, gm_test_data)

def get_ppan_distribution_params(x_dim, y_dim):
    print("Running ppan distribution params for data generation")
    correlation_coefficients = torch.Tensor([0.47, 0.24, 0.85, 0.07, 0.66])
    cov_top = torch.cat(
        (torch.eye(x_dim), torch.diag(correlation_coefficients)), dim=1)
    cov_bot = torch.cat(
        (torch.diag(correlation_coefficients), torch.eye(x_dim)), dim=1)
    cov = torch.cat((cov_top, cov_bot))
    mu = torch.zeros(x_dim + y_dim)
    return (mu, cov)

def get_completely_uncorrelated_distribution_params(x_dim, y_dim):
    print("Running distribution params with completely independent data")
    cov = torch.eye(x_dim + y_dim)
    mu = torch.zeros(x_dim + y_dim)
    return (mu, cov)
