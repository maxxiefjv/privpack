from privpack.utils import DataGenerator
from privpack.utils.metrics import MultivariateGaussianMutualInformation

import torch
import numpy as np
import pytest


@pytest.fixture
def get_random_covariance_matrix():
    def _create_random_covariance_matrix(n):
        return np.power(np.random.normal(size=(n, n)), 2) * 5

    return _create_random_covariance_matrix

def test_compute_schur_complement_fixed_train_data(fixed_train_data):
    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')

    train_cov = torch.Tensor(np.cov(fixed_train_data.T))
    schur_complement = multivariate_gauss_statistic._compute_schur_complement(train_cov, 5)

    assert schur_complement.size() == torch.Size([5, 5])
    assert torch.det(schur_complement) > 0

def test_compute_schur_complement__on_100_random_train_data_samples(get_random_train_data):
    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')

    for i in range(100):
        train_cov = torch.Tensor(np.cov(get_random_train_data().T))
        schur_complement = multivariate_gauss_statistic._compute_schur_complement(train_cov, 5)

        assert schur_complement.size() == torch.Size([5, 5])
        assert torch.det(schur_complement) > 0


@pytest.mark.skip(reason="Test is currently failing. Implementation should be solved.")
def test_compute_schur_complement_on_10_random_cov_matrices(fixed_train_data, get_random_covariance_matrix):
    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')

    for i in range(10):
        cov_matrix = torch.Tensor(get_random_covariance_matrix(10))
        schur_complement = multivariate_gauss_statistic._compute_schur_complement(cov_matrix, 5)

        assert schur_complement.size() == torch.Size([5, 5])
        assert torch.det(schur_complement) > 0

# def test_multivariate_gaussian_mi(fixed_train_data, fixed_test_data):
#     multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')
#     # mi = multivariate_gauss_statistic(train_x, train_y)
#     assert True
