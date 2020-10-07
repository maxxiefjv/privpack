# from privpack.model_selection.validation import CrossValidation

from privpack import BinaryPrivacyPreservingAdversarialNetwork as BinaryGAN

from privpack.utils import compute_released_data_metrics, hamming_distance, PartialBivariateBinaryMutualInformation, ComputeDistortion
from privpack.utils import DataGenerator

from privpack.core.losses import PrivacyLoss, UtilityLoss

import torch
import json

def get_binary_data(privacy_size, public_size, print_metrics=False):
    (norm_dist, acc_dist) = DataGenerator.get_completely_uncorrelated_dist()
    synthetic_data = DataGenerator.generate_binary_data(10000, acc_dist)

    no_train_samples = 8000
    train_data = torch.Tensor(synthetic_data[:no_train_samples])
    test_data = torch.Tensor(synthetic_data[no_train_samples:])
    return (train_data, test_data)

class BinaryNetworkRunner():

    def run(train_data: torch.Tensor, test_data: torch.Tensor, 
            release_size: int, epochs: int, batch_size: int, lambd: int, delta: float) -> None:

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

        metrics = [
            PartialBivariateBinaryMutualInformation('E[MI_ZX]', 0),
            PartialBivariateBinaryMutualInformation('E[MI_ZY]', 1),
            ComputeDistortion('E[hamm(x,y)]', 1).set_distortion_function(lambda x, y: hamming_distance(x, y).to(torch.float64))
        ]
        
        results = {}
        for l in lambd:
            for d in delta:
                print("Training with parameters lambda: {}, delta: {}".format(l, d))

                privatizer_criterion = create_privatizer_criterion(l, d)

                binary_gan = BinaryGAN(torch.device('cpu'), privatizer_criterion, adversary_criterion)
                binary_gan.train(train_data, test_data, epochs, batch_size=batch_size)
                
                with torch.no_grad():
                    released_train_data = binary_gan.privatize(train_data)
                    released_test_data = binary_gan.privatize(test_data)

                compute_released_data_metrics(released_train_data, train_data, metrics)
                compute_released_data_metrics(released_test_data, test_data, metrics)

                metric_results = {
                    'train': metric_results_train,
                    'test': metric_results_test,
                }

                results.setdefault(l, {}).setdefault(d, metric_results)
        
        print(json.dumps(results, sort_keys=True, indent=4))