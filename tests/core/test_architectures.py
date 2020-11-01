from privpack import BinaryPrivacyPreservingAdversarialNetwork as BinaryGAN
from privpack import GaussianPrivacyPreservingAdversarialNetwork as GaussGAN

from privpack.core.criterion import NegativeBinaryMutualInformation, BinaryMutualInformation, BinaryHammingDistance, GaussianMutualInformation, MeanSquaredError
from privpack.core.criterion import PGANCriterion

from privpack.utils import DataGenerator
from privpack.utils.metrics import MultivariateGaussianMutualInformation


import torch
import torch.nn as nn
import pytest
import numpy as np

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
    (norm_dist, acc_dist) = DataGenerator.get_completely_uncorrelated_dist()
    synthetic_data = DataGenerator.generate_binary_data(10, acc_dist)

    train_data = torch.Tensor(synthetic_data[:5])
    test_data = torch.Tensor(synthetic_data[5:])
    return (train_data, test_data)


def test_compute_binary_released_set():
    mock_uniform_rdata = torch.Tensor([
        0.4,
        0.7,
        0.3,
        0.5,
        0.5,
        0,
        1
    ])

    released_data = torch.Tensor([
        0.3,
        0.6,
        0.7,
        0,
        1,
        0.5,
        0.5
    ])

    actual = torch.ceil(released_data - mock_uniform_rdata).to(torch.int)
    expected = torch.Tensor([
        0,
        0,
        1,
        0,
        1,
        1,
        0
    ]).to(torch.int)
    assert torch.all(actual == expected)

def test_binary_mi_gan_x_likelihoods(lambda_and_delta, batch_size, uncorrelated_train_and_test_data, mock_release_probabilities):
    (lambd, delta_constraint) = lambda_and_delta
    (train_data, test_data) = uncorrelated_train_and_test_data

    binary_gan = BinaryGAN(torch.device('cpu'), PGANCriterion())
    mock_x_likelihoods = binary_gan._get_likelihoods(train_data[:batch_size, 0])
    assert mock_x_likelihoods.size() == torch.Size([batch_size, 2])


def test_binary_mi_gan(epochs, lambda_and_delta, uncorrelated_train_and_test_data):
    (lambd, delta_constraint) = lambda_and_delta
    (train_data, test_data) = uncorrelated_train_and_test_data

    binary_gan_criterion = PGANCriterion()

    binary_gan_criterion.add_privacy_criterion(BinaryMutualInformation())
    binary_gan_criterion.add_privacy_criterion(BinaryHammingDistance(lambd, delta_constraint))

    binary_gan_criterion.add_adversary_criterion(NegativeBinaryMutualInformation())

    binary_gan = BinaryGAN(torch.device('cpu'), binary_gan_criterion)
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

    gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, PGANCriterion(), no_hidden_units_per_layer=5, noise_size=1)
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

    gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, PGANCriterion(),
                         no_hidden_units_per_layer=5, noise_size=1)
    x_likelihoods = gauss_gan._get_expected_log_likelihoods(mock_released_samples, mock_x_batch)
    assert x_likelihoods.size() == torch.Size([mock_x_batch.size(0), 1])

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def test_gaussian_release_output_schur_complement(fixed_train_data):
    (privacy_size, public_size, release_size, noise_size) = (5, 5, 5, 5)
    gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, PGANCriterion(),
                         no_hidden_units_per_layer=5, noise_size=1)

    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Z)')

    released_data = gauss_gan.privatizer(torch.Tensor(fixed_train_data)).detach()
    XZ_cov = torch.Tensor(multivariate_gauss_statistic._get_positive_definite_covariance(released_data.numpy(),
                                                                                         fixed_train_data[:, :privacy_size]))

    print("Covariance Matrix is positive semi definite: {}".format(is_pos_def(XZ_cov)))
    schur_complement = multivariate_gauss_statistic._compute_schur_complement(XZ_cov, 5)

    assert schur_complement.size() == torch.Size([5, 5])
    assert torch.det(schur_complement) > 0

def test_gaussian_release_output_schur_100_times(fixed_train_data):
    (privacy_size, public_size, release_size, noise_size) = (5, 5, 5, 5)
    multivariate_gauss_statistic = MultivariateGaussianMutualInformation('I(X;Z)')
    gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, PGANCriterion(),
                         no_hidden_units_per_layer=5, noise_size=1)

    for i in range(100):
        gauss_gan.reset()
        released_data = gauss_gan.privatizer(torch.Tensor(fixed_train_data)).detach()

        XZ_cov = torch.Tensor(multivariate_gauss_statistic._get_positive_definite_covariance(released_data.numpy(),
                                                                                             fixed_train_data[:, :privacy_size]))

        print(XZ_cov)
        print("Covariance Matrix is positive semi definite: {}".format(is_pos_def(XZ_cov)))
        schur_complement = multivariate_gauss_statistic._compute_schur_complement(XZ_cov, 5)
        print("Schur complement is positive semi definite: {}".format(is_pos_def(schur_complement)))

        assert schur_complement.size() == torch.Size([5, 5])
        assert torch.det(schur_complement) > 0


def test_binary_class(lambda_and_delta, batch_size, uncorrelated_train_and_test_data, mock_release_probabilities):
    class My_Privatizer(nn.Module):
        """
        Adversary network consisting of a single linear transformation followed by the non-linear
        Sigmoid activation
        """
        def __init__(self, gan_parent):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(1, 1, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    (lambd, delta_constraint) = lambda_and_delta
    (train_data, test_data) = uncorrelated_train_and_test_data

    binary_gan = BinaryGAN(torch.device('cpu'), PGANCriterion())
    str_representation = str(binary_gan)
    binary_gan.set_privatizer_class(My_Privatizer)
    
    assert not str_representation == str(binary_gan)

# TEST TIME IS WAY TO LONG. Caused by k>1?
# def test_gaussian_privatizer_criterion():

#     def create_gauss_privatizer_criterion(lambd, delta_constraint):
#         def privatizer_criterion(releases, log_likelihood_x, expected):
#             assert releases.requires_grad
#             assert log_likelihood_x.requires_grad

#             privacy_loss = PrivacyLoss().gaussian_mutual_information_loss(releases, log_likelihood_x)
#             utility_loss = UtilityLoss(lambd, delta_constraint).expected_mean_squared_error(releases, expected)

#             return privacy_loss + utility_loss

#         return privatizer_criterion

#     def adversary_criterion(releases, log_likelihood_x, actual_public):
#         assert not releases.requires_grad
#         assert log_likelihood_x.requires_grad
#         return -1 * PrivacyLoss().gaussian_mutual_information_loss(releases, log_likelihood_x)

#     (privacy_size, public_size, release_size, noise_size) = (1, 1, 1, 1)
#     (lambd, delta_constraint) = (1, 0)
#     epochs = 1

#     privatizer_criterion = create_gauss_privatizer_criterion(lambd, delta_constraint)

#     gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, privatizer_criterion, adversary_criterion, no_hidden_units_per_layer=5, noise_size=1)
#     (mu, cov) = DataGenerator.get_completely_uncorrelated_distribution_params(privacy_size, public_size)

#     (gm_train, gm_test) = DataGenerator.generate_gauss_mixture_data(mu, cov, seed=0)
#     gauss_gan.train(gm_train, gm_test, epochs, k=1)
