"""
Utility module containing the most used utilities in this library.
"""
from typing import List
from .data_generator import DataGenerator
from .metrics import (
    PartialBivariateBinaryMutualInformation, PartialMultivariateGaussianMutualInformation, ComputeDistortion, Metric
)

import torch

def compute_released_data_metrics(released_data, data, metrics: List[Metric]):
    """
    Utiltity function exectuing every provided statistic to the supploed `released_data`
    and `data`.

    Parameters:

    - `released_data`: Ideally data as output from some privatizer mechanisms
    - `data`: Original data obtained from some source
    """
    metrics_report = {}
    for metric in metrics:
        data_stats = metric(released_data.detach(), data)
        metrics_report[metric.name] = data_stats

    return metrics_report

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
