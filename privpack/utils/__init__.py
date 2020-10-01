"""
Utility module containing the most used utilities in this library.
"""
from .data_generator import DataGenerator
from .statistics import (
    PartialBivariateBinaryMutualInformation, PartialMultivariateGaussianMutualInformation, ComputeDistortion
)

import torch

def compute_released_data_statistics(released_data, data, statistics):
    """
    Utiltity function exectuing every provided statistic to the supploed `released_data`
    and `data`.

    Parameters:

    - `released_data`: Ideally data as output from some privatizer mechanisms
    - `data`: Original data obtained from some source
    """
    statistics_report = {}
    for statistic in statistics:
        data_stats = statistic(released_data.detach(), data)
        statistics_report[statistic.name] = data_stats

    return statistics_report

# Bivariate Binary related Utility functions
def get_likelihood_xi_given_z(adversary_out, Xi):
    res = adversary_out  # P(Xi = 1 | z)
    # If Xi = 0 -> res = 1 - res; else (if Xi = 1) -> res = res
    res = (1 - Xi) * (1 - res) + Xi * res

    # if Xi == 0:
    #     res = 1 - adversary_out ## 1 - P(Xi = 1 | z) = P(Xi = 0 | z)

    return res  # P(x | z)

def hamming_distance(actual, expected):
    return (actual != expected).to(torch.int)

def elementwise_mse(released, expected):
    return torch.square(torch.norm(expected - released, p=None, dim=1))

# (Multivariate) Gaussian related Utility functions
def sample_from_network(privatizer, entry, k):
    sampled_data = torch.Tensor([])
    entry = entry.to(torch.float)
    for j in range(k):
        privatized_sample = privatizer(entry)
        privatized_sample = privatized_sample.unsqueeze(0)
        sampled_data = torch.cat((sampled_data, privatized_sample), 0)

    return sampled_data
