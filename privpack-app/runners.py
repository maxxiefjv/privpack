
from privpack import GaussianPrivacyPreservingAdversarialNetwork as GaussGAN
from privpack import BinaryPrivacyPreservingAdversarialNetwork as BinaryGAN

from privpack.utils import ComputeDistortion
from privpack.utils import compute_released_data_metrics, elementwise_mse, PartialMultivariateGaussianMutualInformation
from privpack.utils import compute_released_data_metrics, hamming_distance, PartialBivariateBinaryMutualInformation

from privpack.utils import DataGenerator

from privpack.core.criterion import PGANCriterion
from privpack.core.criterion import NegativeGaussianMutualInformation, GaussianMutualInformation, MeanSquaredError
from privpack.core.criterion import BinaryMutualInformation, BinaryHammingDistance, NegativeBinaryMutualInformation

import torch
import json


def get_gaussian_data(privacy_size, public_size, print_metrics=False):
    (mu, cov) = DataGenerator.get_completely_uncorrelated_distribution_params(privacy_size, public_size)
    return DataGenerator.generate_gauss_mixture_data(mu, cov, seed=0)

def get_binary_data(privacy_size, public_size, print_metrics=False):
    (norm_dist, acc_dist) = DataGenerator.get_completely_uncorrelated_dist()
    synthetic_data = DataGenerator.generate_binary_data(10000, acc_dist)

    no_train_samples = 8000
    train_data = torch.Tensor(synthetic_data[:no_train_samples])
    test_data = torch.Tensor(synthetic_data[no_train_samples:])
    return (train_data, test_data)

class PGANRunner():
    def __init__(self, gan_network, metrics, lambd: int, delta: float):
        self.gan_network = gan_network
        self.metrics = metrics

        print("Created runner with parameters lambda: {}, delta: {}".format(lambd, delta))


    def run(self, train_data: torch.Tensor, test_data: torch.Tensor, epochs: int, batch_size: int, k: int) -> None:
        self.gan_network.reset()

        if k:
            self.gan_network.train(train_data, test_data, epochs, batch_size=batch_size, k=k)
        else :
            self.gan_network.train(train_data, test_data, epochs, batch_size=batch_size)
        
        with torch.no_grad():
            released_train_data = gan_network.privatize(train_data)
            released_test_data = gan_network.privatize(test_data)

        metric_results_train = compute_released_data_metrics(released_train_data, train_data, self.metrics)
        metric_results_test = compute_released_data_metrics(released_test_data, test_data, self.metrics)

        metric_results = {
            'train': metric_results_train,
            'test': metric_results_test,
        }

        return metric_results


class GaussianNetworkRunner(PGANRunner):

    def __init__(self, privacy_size: int, public_size: int, hidden_layers_width: int, release_size: int, lambd: int, delta: float):
        super().__init__(
            gan_network = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, 
                                PGANCriterion().add_privacy_criterion(GaussianMutualInformation()).add_privacy_criterion(MeanSquaredError(lambd, delta)).add_adversary_criterion(NegativeGaussianMutualInformation()),
                                no_hidden_units_per_layer=hidden_layers_width, noise_size=5),
            metrics = [
                PartialMultivariateGaussianMutualInformation('E[MI_XZ]', range(0, privacy_size)),
                PartialMultivariateGaussianMutualInformation('E[MI_YZ]', range(privacy_size, privacy_size + public_size)),
                ComputeDistortion('E[mse(z,y)]', range(privacy_size, privacy_size + public_size)).set_distortion_function(elementwise_mse)
            ],
            lambd= lambd, delta= delta
        )

    def run(self, train_data: torch.Tensor, test_data: torch.Tensor, epochs: int, batch_size: int, k: int) -> None:
        super().run(train_data, test_data, epochs, batch_size, k)

class BinaryNetworkRunner(PGANRunner):

    def __init__(self, lambd: int, delta: float):
        super().__init__(
            gan_network =  BinaryGAN(torch.device('cpu'), 
                                     PGANCriterion().add_privacy_criterion(BinaryMutualInformation()).add_privacy_criterion(BinaryHammingDistance(lambd, delta)).add_adversary_criterion(NegativeBinaryMutualInformation())
                                    ),
            metrics = [
                PartialBivariateBinaryMutualInformation('E[MI_ZX]', 0),
                PartialBivariateBinaryMutualInformation('E[MI_ZY]', 1),
                ComputeDistortion('E[hamm(x,y)]', 1).set_distortion_function(lambda x, y: hamming_distance(x, y).to(torch.float64))
            ],
            lambd= lambd, delta= delta
        )

    def run(self, train_data: torch.Tensor, test_data: torch.Tensor, epochs: int, batch_size: int) -> None:
        super().run(train_data, test_data, epochs, batch_size, None)
        