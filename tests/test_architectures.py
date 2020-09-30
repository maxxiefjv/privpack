from privpack import BinaryGenerativeAdversarialNetwork as BinaryGAN
from privpack import GaussianGenerativeAdversarialNetwork as GaussGAN
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

def test_binary_mi_gan_x_likelihoods(lambda_and_delta, batch_size, uncorrelated_train_and_test_data, mock_release_probabilities):
    (lambd, delta_constraint) = lambda_and_delta
    (train_data, test_data) = uncorrelated_train_and_test_data

    binary_gan = BinaryGAN(torch.device('cpu'), None, None)
    mock_x_likelihoods = binary_gan._get_likelihoods(train_data[:batch_size, 0])
    assert mock_x_likelihoods.size() == torch.Size([batch_size, 2])


def test_binary_mi_gan(epochs, lambda_and_delta, uncorrelated_train_and_test_data):
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

    (lambd, delta_constraint) = lambda_and_delta
    (train_data, test_data) = uncorrelated_train_and_test_data
    privatizer_criterion = create_privatizer_criterion(lambd, delta_constraint)

    binary_gan = BinaryGAN(torch.device('cpu'), privatizer_criterion, adversary_criterion)
    binary_gan.train(train_data, test_data, epochs)

def test_gaussian_get_x_likelihoods():
    mock_releases = torch.Tensor([
        [0.25],
        [0.75],
        [0],
        [1]
    ])

    mock_x_batch = torch.Tensor([
        [0.5],
        [0.5],
        [1],
        [1]
    ])

    (privacy_size, public_size, release_size, noise_size) = (1, 1, 1, 1)

    gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, None, None, no_hidden_units_per_layer=5, noise_size=1)
    x_likelihoods = gauss_gan._get_log_likelihoods(mock_releases, mock_x_batch)
    assert x_likelihoods.size() == torch.Size([mock_x_batch.size(0), 1])

def test_gaussian_get_expected_x_likelihoods():
    mock_released_samples = torch.Tensor([
        [0.25],
        [0.75],
        [0],
        [1]
    ]).repeat(5, 1, 1)

    mock_x_batch = torch.Tensor([
        [0.5],
        [0.5],
        [1],
        [1]
    ])

    (privacy_size, public_size, release_size, noise_size) = (1, 1, 1, 1)

    gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, None, None, no_hidden_units_per_layer=5, noise_size=1)
    x_likelihoods = gauss_gan._get_expected_log_likelihoods(mock_released_samples, mock_x_batch)
    assert x_likelihoods.size() == torch.Size([mock_x_batch.size(0), 1])

def test_gaussian_privatizer_criterion():

    def create_gauss_privatizer_criterion(lambd, delta_constraint):
        def privatizer_criterion(releases, log_likelihood_x, expected):
            assert releases.requires_grad
            assert not log_likelihood_x.requires_grad

            privacy_loss = PrivacyLoss().gaussian_mutual_information_loss(log_likelihood_x)
            utility_loss = UtilityLoss(lambd, delta_constraint).expected_mean_squared_error(releases, expected)

            return privacy_loss + utility_loss

        return privatizer_criterion

    def adversary_criterion(releases, log_likelihood_x, actual_public):
        assert not releases.requires_grad
        assert log_likelihood_x.requires_grad
        return -1 * PrivacyLoss().gaussian_mutual_information_loss(log_likelihood_x)

    (privacy_size, public_size, release_size, noise_size) = (1, 1, 1, 1)
    (lambd, delta_constraint) = (1, 0)
    epochs = 1

    privatizer_criterion = create_gauss_privatizer_criterion(lambd, delta_constraint)

    gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, privatizer_criterion, adversary_criterion, no_hidden_units_per_layer=5, noise_size=1)
    (mu, cov) = datagen.get_completely_uncorrelated_distribution_params(privacy_size, public_size)

    (gm_train, gm_test) = datagen.generate_gauss_mixture_data(mu, cov)
    gauss_gan.train(gm_train, gm_test, epochs, k=2)
