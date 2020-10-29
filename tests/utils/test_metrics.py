from privpack.utils import DataGenerator
from privpack.utils.metrics import MultivariateGaussianMutualInformation, ComputeDistortion
from privpack.utils import hamming_distance, elementwise_mse

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

def test_mse_distortion_1d():
    mse = ComputeDistortion('E[mse(x,y)]', [0]).set_distortion_function(elementwise_mse)

    mock_released_samples = torch.Tensor([
        [0.25],
        [0.75],
        [0],
        [1]
    ])

    mock_expected_samples = torch.Tensor([
        [0.3],
        [0.5],
        [0.9],
        [0.3],
    ])

    expected_out = torch.Tensor([
        (0.25 - 0.3) ** 2,
        (0.75 - 0.5) ** 2,
        (0 - 0.9) ** 2,
        (1 - 0.3) ** 2,
    ]).mean().item()

    actual_out = mse(mock_released_samples, mock_expected_samples)
    
    print(actual_out, expected_out)
    assert np.isclose(expected_out, actual_out)


@pytest.mark.xfail(raises=RuntimeError)
def test_mse_distortion_dimension_error():
    mse = ComputeDistortion('E[mse(x,y)]', 0).set_distortion_function(elementwise_mse)

    mock_released_samples = torch.Tensor([
        [0.25],
        [0.75],
        [0],
        [1]
    ])

    mock_expected_samples = torch.Tensor([
        [0.3],
        [0.5],
        [0.9],
        [0.3],
    ])

    expected_out = torch.Tensor([
        (0.25 - 0.3) ** 2,
        (0.75 - 0.5) ** 2,
        (0 - 0.9) ** 2,
        (1 - 0.3) ** 2,
    ]).mean().item()

    actual_out = mse(mock_released_samples, mock_expected_samples)
    assert np.isclose(expected_out, actual_out)


def test_mse_distortion_5d():
    mse = ComputeDistortion('E[mse(x,y)]', [0,1,2]).set_distortion_function(elementwise_mse)

    mock_released_samples = torch.Tensor([
        [0.25, 0.5, 0.75],
        [0.75, 0.75, 0.75],
        [0, 0, 0],
        [1, 1, 1]
    ])

    mock_expected_samples = torch.Tensor([
        [0.3, 0.5, 0.5],
        [0.5, 0.6, 0.4],
        [0.9, 0.4, 0.3],
        [0.3, 0.7, 0.6],
    ])

    expected_out = torch.Tensor([
        (0.25 - 0.3) ** 2 + (0.5 - 0.5) ** 2 + (0.75 - 0.5) ** 2,
        (0.75 - 0.5) ** 2 + (0.75 - 0.6) ** 2 + (0.75 - 0.4) ** 2,
        (0 - 0.9) ** 2 + (0 - 0.4) ** 2 + (0 - 0.3) ** 2,
        (1 - 0.3) ** 2 + (1 - 0.7) ** 2 + (1 - 0.6) ** 2
    ]).mean().item()

    actual_out = mse(mock_released_samples, mock_expected_samples)
    
    assert np.isclose(expected_out, actual_out)


def test_mse_distortion_example():
    data = torch.Tensor([
        [-1.5870],
        [-0.3276],
        [-2.0638],
        [-0.9552],
        [ 0.2117],
        [ 0.2597],
        [-0.6986],
        [-0.3355],
        [-1.4931],
        [-0.5350]
    ])

    release = torch.Tensor([
        [-0.1676],
        [ 0.0000],
        [ 0.0000],
        [ 0.1059],
        [ 0.4314],
        [-0.0876],
        [-0.0558],
        [ 0.0000],
        [ 0.0000],
        [ 0.0420]
    ])

    actual = elementwise_mse(release, data)
    expected = torch.Tensor([
        (-1.5870 - -0.1676) ** 2,
        (-0.3276) ** 2,
        (-2.0638) ** 2,
        (-0.9552 - 0.1059) ** 2,
        ( 0.2117 - 0.4314) ** 2,
        ( 0.2597 - -0.0876) ** 2,
        (-0.6986 - -0.0558) ** 2,
        (-0.3355) ** 2,
        (-1.4931) ** 2,
        (-0.5350 - 0.0420) ** 2
    ])

    assert torch.all(torch.isclose(actual, expected, atol=1e-4))
    assert actual.mean() > 0

# def test_multivariate_gaussian_mi(fixed_train_data, fixed_test_data):
#     multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Y)')
#     # mi = multivariate_gauss_statistic(train_x, train_y)
#     assert True
