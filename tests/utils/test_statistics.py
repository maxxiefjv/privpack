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

@pytest.fixture
def get_random_covariance_matrix():
    def _create_random_covariance_matrix(n):
        return np.power(np.random.uniform(0, 1, (n, n)), 2) * 10

    return _create_random_covariance_matrix

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

def test_compute_schur_complement(train_data, get_random_covariance_matrix):
    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')

    for i in range(10):
        cov_matrix = torch.Tensor(get_random_covariance_matrix(10))
        schur_complement = multivariate_gauss_statistic._compute_schur_complement(cov_matrix, 5)

        print(i, cov_matrix)
        assert schur_complement.size() == torch.Size([5, 5])
        assert torch.det(schur_complement) > 0
