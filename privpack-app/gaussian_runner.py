
from privpack import GaussianPrivacyPreservingAdversarialNetwork as GaussGAN

from privpack.utils import compute_released_data_metrics, elementwise_mse, PartialMultivariateGaussianMutualInformation, ComputeDistortion
from privpack.utils import DataGenerator

from privpack.core.losses import PrivacyLoss, UtilityLoss

import torch
import json

def get_gaussian_data(privacy_size, public_size, print_metrics=False):
    (mu, cov) = DataGenerator.get_completely_uncorrelated_distribution_params(privacy_size, public_size)
    return DataGenerator.generate_gauss_mixture_data(mu, cov, seed=0)

class GaussianNetworkRunner():

    def run(train_data: torch.Tensor, test_data: torch.Tensor, privacy_size: int, 
            public_size: int, hidden_layers_width: int, release_size: int, 
            epochs: int, batch_size: int, lambd: int, delta: float, k: int) -> None:
            
        def create_gauss_privatizer_criterion(lambd, delta_constraint):
            def privatizer_criterion(releases, log_likelihood_x, expected):
                assert releases.requires_grad
                assert log_likelihood_x.requires_grad

                privacy_loss = PrivacyLoss().gaussian_mutual_information_loss(releases, log_likelihood_x)
                utility_loss = UtilityLoss(lambd, delta_constraint).expected_mean_squared_error(releases, expected)

                return privacy_loss + utility_loss

            return privatizer_criterion

        def adversary_criterion(releases, log_likelihood_x, actual_public):
            assert not releases.requires_grad
            assert log_likelihood_x.requires_grad
            return -1 * PrivacyLoss().gaussian_mutual_information_loss(releases, log_likelihood_x)

        metrics = [
            PartialMultivariateGaussianMutualInformation('E[MI_XZ]', range(0, privacy_size)),
            PartialMultivariateGaussianMutualInformation('E[MI_YZ]', range(privacy_size, privacy_size + public_size)),
            ComputeDistortion('E[mse(z,y)]', range(privacy_size, privacy_size + public_size)).set_distortion_function(elementwise_mse)
        ]
        
        results = {}
        for l in lambd:
            for d in delta:
                for no_samples in k:
                    print("Training with parameters lambda: {}, delta: {}".format(l, d))

                    privatizer_criterion = create_gauss_privatizer_criterion(l, d)

                    gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, privatizer_criterion, adversary_criterion,
                                         no_hidden_units_per_layer=hidden_layers_width, noise_size=5)

                    gauss_gan.train(train_data, test_data, epochs, batch_size=batch_size, k=1)
                    
                    with torch.no_grad():
                        released_train_data = gauss_gan.privatize(train_data)
                        released_test_data = gauss_gan.privatize(test_data)

                    compute_released_data_metrics(released_train_data, train_data, metrics)
                    compute_released_data_metrics(released_test_data, test_data, metrics)

                    metric_results = {
                        'train': metric_results_train,
                        'test': metric_results_test,
                    }

                    results.setdefault(l, {}).setdefault(d, {}).setdefault(no_samples, metric_results)
        
        print(json.dumps(results, sort_keys=True, indent=4))