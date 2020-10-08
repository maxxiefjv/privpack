
import torch
import math
import pytest

from privpack.core.criterion import DiscreteMutualInformation ,BinaryHammingDistance, BinaryMutualInformation, MeanSquaredError, GaussianMutualInformation
from privpack.core.criterion import PGANCriterion

@pytest.fixture
def mock_x_likelihoods():
    return torch.Tensor([
        [0.7, 0.6],
        [0.8, 0.1],
        [0.5, 0.9]
    ])

@pytest.fixture
def mock_public_values():
    return torch.Tensor([
        1,
        0,
        1
    ])

def test_binary_mi_and_hamming(mock_release_probabilities, mock_x_likelihoods, mock_public_values):
    (lambd, delta_constraint) = (1, 0)

    expected_out = torch.Tensor([
        0.4 * math.log2(0.7) + 0.6 * math.log2(0.6) + 0.4 ** 2,
        0.7 * math.log2(0.8) + 0.3 * math.log2(0.1) + 0.3 ** 2,
        0.9 * math.log2(0.5) + 0.1 * math.log2(0.9) + 0.9 ** 2
    ])

    binary_mi_loss_out = BinaryMutualInformation()(mock_release_probabilities, mock_x_likelihoods)
    binary_hamming_loss_out = BinaryHammingDistance(lambd, delta_constraint)(mock_release_probabilities, mock_public_values)
    actual_out = binary_mi_loss_out + binary_hamming_loss_out

    assert torch.all(torch.isclose(actual_out, expected_out))


def test_discrete_mutual_information(mock_release_probabilities, mock_x_likelihoods):
    mock_release_probabilities_all = torch.cat((1 - mock_release_probabilities, mock_release_probabilities), dim=1)
    expected_out = torch.Tensor([
        0.4 * math.log2(0.7) + 0.6 * math.log2(0.6),
        0.7 * math.log2(0.8) + 0.3 * math.log2(0.1),
        0.9 * math.log2(0.5) + 0.1 * math.log2(0.9)
    ])

    actual_out = DiscreteMutualInformation()(mock_release_probabilities_all, mock_x_likelihoods)

    assert torch.all(torch.isclose(actual_out, expected_out))


def test_binary_mutual_information(mock_release_probabilities, mock_x_likelihoods):

    expected_out = torch.Tensor([
        0.4 * math.log2(0.7) + 0.6 * math.log2(0.6),
        0.7 * math.log2(0.8) + 0.3 * math.log2(0.1),
        0.9 * math.log2(0.5) + 0.1 * math.log2(0.9)
    ])

    actual_out = BinaryMutualInformation()(mock_release_probabilities, mock_x_likelihoods)
    assert torch.all(torch.isclose(actual_out, expected_out))

def test_negative_binary_mutual_information(mock_release_probabilities, mock_x_likelihoods):
    mock_release_all_probabilities = torch.cat((1 - mock_release_probabilities, mock_release_probabilities), dim=1)
    expected_out = torch.mul(mock_release_all_probabilities, -torch.log2(mock_x_likelihoods)).sum(dim=1)

    actual_out = - BinaryMutualInformation()(mock_release_probabilities, mock_x_likelihoods)
    assert torch.all(torch.isclose(actual_out, expected_out))
privacy_criterions) = (1, 0)

    expected_out = torch.Tensor([
        0.4 ** 2,
        0.3 ** 2,
        0.9 ** 2
    ])

    actual_out = BinaryHammingDistance(lambd, delta_constraint)(mock_release_probabilities, mock_public_values)
    assert torch.all(torch.isclose(actual_out, expected_out))

def test_gauss_mean_squared_error_loss():
    (lambd, delta_constraint) = (1, 0)
    k = 5

    mock_released_samples = torch.Tensor([
        [0.25],
        [0.75],
        [0],
        [1]
    ]).repeat(k, 1, 1)

    mock_expected_samples = torch.Tensor([
        [0.3],
        [0.5],
        [0.9],
        [0.3],
    ])

    actual_out = MeanSquaredError(lambd, delta_constraint)(mock_released_samples, mock_expected_samples)

    expected_out = torch.Tensor([
        ((0.25 - 0.3) ** 2) ** 2,
        ((0.75 - 0.5) ** 2) ** 2,
        ((0 - 0.9) ** 2) ** 2,
        ((1 - 0.3) ** 2) ** 2,
    ])

    print(expected_out, actual_out)
    assert torch.all(torch.isclose(actual_out, expected_out))


def test_gan_criterion(mock_release_probabilities, mock_x_likelihoods, mock_public_values):
    (lambd, delta_constraint) = (1, 0)

    expected_out = torch.Tensor([
        0.4 * math.log2(0.7) + 0.6 * math.log2(0.6) + 0.4 ** 2,
        0.7 * math.log2(0.8) + 0.3 * math.log2(0.1) + 0.3 ** 2,
        0.9 * math.log2(0.5) + 0.1 * math.log2(0.9) + 0.9 ** 2
    ])

    binary_gan_criterion = PGANCriterion()
    binary_gan_criterion.add_privacy_criterion(BinaryMutualInformation())
    binary_gan_criterion.add_privacy_criterion(BinaryHammingDistance(lambd, delta_constraint))

    actual_out = binary_gan_criterion.privacy_loss(mock_release_probabilities, mock_x_likelihoods, mock_public_values)

    assert torch.all(torch.isclose(actual_out, expected_out))
