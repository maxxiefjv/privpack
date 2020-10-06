from privpack.utils import DataGenerator
from privpack.utils.statistics import MultivariateGaussianMutualInformation

import torch
import numpy as np
import pytest


@pytest.fixture
def train_data():
    """
    The generated data consists of 10 columns, generated using the DataGenerator.get_ppan_distribution_params function.
    """
    return np.loadtxt('tests/data/static_train_data.csv', delimiter=',')


@pytest.fixture
def test_data():
    """
    The generated data consists of 10 columns, generated using the DataGenerator.get_ppan_distribution_params function.
    """
    return np.loadtxt('tests/data/static_test_data.csv', delimiter=',')


def test_compute_schur_complement_static_train_data(train_data):
    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')

    train_cov = torch.Tensor(np.cov(train_data.T))
    schur_complement = multivariate_gauss_statistic._compute_schur_complement(train_cov, 5)

    assert schur_complement.size() == torch.Size([5, 5])
    assert True


def test_multivariate_gaussian_mi(train_data, test_data):
    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')
    # mi = multivariate_gauss_statistic(train_x, train_y)
    assert True

def test_compute_schur_complement(train_data):
    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')
    train_cov = torch.Tensor(np.cov(train_data.T))
    schur_complement = multivariate_gauss_statistic._compute_schur_complement(train_cov, 5)

    assert schur_complement.size() == torch.Size([5, 5])
    