from privpack import BinaryGenerativeAdversarialNetwork as BinaryGAN
from privpack.losses import PrivacyLoss, UtilityLoss
from privpack import datagen

import torch
import pytest

@pytest.fixture
def lambda_and_delta():
    return (1, 0)


@pytest.fixture
def epochs():
    return 1


@pytest.fixture
def batch_size():
    return 1


@pytest.fixture
def uncorrelated_train_and_test_data():
    (norm_dist, acc_dist) = datagen.get_completely_uncorrelated_dist()
    synthetic_data = datagen.generate_binary_data(10, acc_dist)

    train_data = torch.Tensor(synthetic_data[:5])
    test_data = torch.Tensor(synthetic_data[5:])
    return (train_data, test_data)

@pytest.fixture
def mock_release_probabilities():
    return torch.Tensor([
        [0.6],
        [0.3],
        [0.1]
    ])


def create_privatizer_criterion(lambd, delta_constraint):
    def privatizer_criterion(release_probabilities, likelihood_x, actual_public):
        assert release_probabilities.requires_grad
        assert not likelihood_x.requires_grad

        privacy_loss = PrivacyLoss().binary_mi_loss(release_probabilities, likelihood_x)
        utility_loss = UtilityLoss(lambd, delta_constraint).expected_binary_hamming_distance(release_probabilities, actual_public)

        return privacy_loss + utility_loss

    return privatizer_criterion

def adversary_criterion(release_probabilities, likelihood_x, actual_public):
    assert not release_probabilities.requires_grad
    assert likelihood_x.requires_grad
    return -1 * PrivacyLoss().binary_mi_loss(release_probabilities, likelihood_x)

def test_binary_mi_gan_x_likelihoods(lambda_and_delta, batch_size, uncorrelated_train_and_test_data, mock_release_probabilities):
    (lambd, delta_constraint) = lambda_and_delta
    (train_data, test_data) = uncorrelated_train_and_test_data
    privatizer_criterion = create_privatizer_criterion(lambd, delta_constraint)

    binary_gan = BinaryGAN(torch.device('cpu'), privatizer_criterion, adversary_criterion)
    mock_x_likelihoods = binary_gan._get_likelihoods(train_data[:batch_size, 0])
    assert mock_x_likelihoods.size() == torch.Size([batch_size, 2])


def test_binary_mi_gan(epochs, lambda_and_delta, uncorrelated_train_and_test_data):
    (lambd, delta_constraint) = lambda_and_delta
    (train_data, test_data) = uncorrelated_train_and_test_data
    privatizer_criterion = create_privatizer_criterion(lambd, delta_constraint)

    binary_gan = BinaryGAN(torch.device('cpu'), privatizer_criterion, adversary_criterion)
    binary_gan.train(train_data, test_data, epochs)
