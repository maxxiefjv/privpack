
import torch
import math
import pytest

from privpack.losses import PrivacyLoss, UtilityLoss

@pytest.fixture
def mock_release_probabilities():
    return torch.Tensor([
        [0.6],
        [0.3],
        [0.1]
    ])

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

    binary_mi_loss_out = PrivacyLoss().binary_mi_loss(mock_release_probabilities, mock_x_likelihoods)
    binary_hamming_loss_out = UtilityLoss(lambd, delta_constraint).expected_binary_hamming_distance(mock_release_probabilities, mock_public_values)
    actual_out = binary_mi_loss_out + binary_hamming_loss_out

    assert torch.all(torch.isclose(actual_out, expected_out))


def test_discrete_mutual_information(mock_release_probabilities, mock_x_likelihoods):
    mock_release_probabilities_all = torch.cat((1 - mock_release_probabilities, mock_release_probabilities), dim=1)
    expected_out = torch.Tensor([
        0.4 * math.log2(0.7) + 0.6 * math.log2(0.6),
        0.7 * math.log2(0.8) + 0.3 * math.log2(0.1),
        0.9 * math.log2(0.5) + 0.1 * math.log2(0.9)
    ])

    actual_out = PrivacyLoss().discrete_mi_loss(mock_release_probabilities_all, mock_x_likelihoods)

    assert torch.all(torch.isclose(actual_out, expected_out))


def test_binary_mutual_information(mock_release_probabilities, mock_x_likelihoods):

    expected_out = torch.Tensor([
        0.4 * math.log2(0.7) + 0.6 * math.log2(0.6),
        0.7 * math.log2(0.8) + 0.3 * math.log2(0.1),
        0.9 * math.log2(0.5) + 0.1 * math.log2(0.9)
    ])

    actual_out = PrivacyLoss().binary_mi_loss(mock_release_probabilities, mock_x_likelihoods)
    assert torch.all(torch.isclose(actual_out, expected_out))

def test_negative_binary_mutual_information(mock_release_probabilities, mock_x_likelihoods):
    mock_release_all_probabilities = torch.cat((1 - mock_release_probabilities, mock_release_probabilities), dim=1)
    expected_out = torch.mul(mock_release_all_probabilities, -torch.log2(mock_x_likelihoods)).sum(dim=1)

    actual_out = - PrivacyLoss().binary_mi_loss(mock_release_probabilities, mock_x_likelihoods)
    assert torch.all(torch.isclose(actual_out, expected_out))

def test_discrete_hamming_distance(mock_release_probabilities, mock_public_values):
    (lambd, delta_constraint) = (1, 0)

    expected_out = torch.Tensor([
        0.4 ** 2,
        0.3 ** 2,
        0.9 ** 2
    ])

    actual_out = UtilityLoss(lambd, delta_constraint).expected_binary_hamming_distance(mock_release_probabilities, mock_public_values)
    assert torch.all(torch.isclose(actual_out, expected_out))

def test_gauss_mean_squared_error_loss():
    (lambd, delta_constraint) = (1, 0)

    mock_released_samples = torch.Tensor([
        [0.25],
        [0.75],
        [0],
        [1]
    ]).repeat(5, 1, 1)

    mock_expected_samples = torch.Tensor([
        [0.3],
        [0.5],
        [0.9],
        [0.3],
    ])

    actual_out = UtilityLoss(lambd, delta_constraint).expected_mean_squared_error(mock_released_samples, mock_expected_samples)

    expected_out = torch.Tensor([
        ((0.25 - 0.3) ** 2) ** 2,
        ((0.75 - 0.5) ** 2) ** 2,
        ((0 - 0.9) ** 2) ** 2,
        ((1 - 0.3) ** 2) ** 2,
    ])

    print(expected_out, actual_out)
    assert torch.all(torch.isclose(actual_out, expected_out))
