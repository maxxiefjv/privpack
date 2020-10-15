from privpack.utils import DataGenerator
from privpack.utils.metrics import MultivariateGaussianMutualInformation, ComputeDistortion
from privpack.utils import hamming_distance

import torch
import numpy as np
import pytest


@pytest.fixture
def get_random_covariance_matrix():
    def _create_random_covariance_matrix(n):
        return np.power(np.diag(np.random.normal(size=(n))), 2)
        # A = np.diag(np.random.normal(size=(n)))
        # return A * A.T

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

def test_compute_schur_complement_on_10_random_cov_matrices(fixed_train_data, get_random_covariance_matrix):
    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')

    for i in range(100):
        cov_matrix = torch.Tensor(get_random_covariance_matrix(10))
        schur_complement = multivariate_gauss_statistic._compute_schur_complement(cov_matrix, 5)

        assert schur_complement.size() == torch.Size([5, 5])
        assert torch.det(schur_complement) > 0

def test_hamming_distance_half_bad():
    hamm = ComputeDistortion('E[hamm(x,y)]', 1).set_distortion_function(lambda x, y: hamming_distance(x, y).to(torch.float64))

    W = torch.Tensor( [
        [0, 0],
        [0, 0],
        [0, 1],
        [0, 1],
    ])

    yhat = torch.Tensor( [
        1,
        0,
        0,
        1,
    ])

    assert 0.5 == hamm(yhat, W)


def test_hamming_distance_all_bad():
    hamm = ComputeDistortion('E[hamm(x,y)]', 1).set_distortion_function(lambda x, y: hamming_distance(x, y).to(torch.float64))

    W = torch.Tensor( [
        [0, 0],
        [0, 0],
        [0, 1],
        [0, 1],
    ])

    yhat = torch.Tensor( [
        1,
        1,
        0,
        0,
    ])

    assert 1 == hamm(yhat, W)


def test_hamming_distance_none_bad():
    hamm = ComputeDistortion('E[hamm(x,y)]', 1).set_distortion_function(lambda x, y: hamming_distance(x, y).to(torch.float64))

    W = torch.Tensor( [
        [0, 0],
        [0, 0],
        [0, 1],
        [0, 1],
    ])

    yhat = torch.Tensor( [
        0,
        0,
        1,
        1,
    ])

    assert 0 == hamm(yhat, W)
# def test_multivariate_gaussian_mi(fixed_train_data, fixed_test_data):
#     multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')
#     # mi = multivariate_gauss_statistic(train_x, train_y)
#     assert True
