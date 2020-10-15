from privpack.core.criterion import PGANCriterion
from privpack.core.criterion import BinaryHammingDistance, BinaryMutualInformation, NegativeBinaryMutualInformation
from privpack.core.criterion import MeanSquaredError, GaussianMutualInformation, NegativeGaussianMutualInformation

from privpack.model_selection.experiment import Experiment, Expectations
from privpack.utils.metrics import PartialBivariateBinaryMutualInformation, PartialMultivariateGaussianMutualInformation, ComputeDistortion

from privpack.utils import hamming_distance, elementwise_mse

class ExperimentRunner():
    def __init__(self):
        pass


class BinaryExperiment(ExperimentRunner):

    def __init__(self):
        pass

    def run(self, data, epochs, batch_size, lambd, delta, verbose=False):
        results = {}
        n_splits = 3

        for d in delta:
            for l in lambd:
                network_criterion = PGANCriterion()
                network_criterion.add_privacy_criterion(BinaryHammingDistance(lambd=l, delta_constraint=d)).add_privacy_criterion(BinaryMutualInformation())
                network_criterion.add_adversary_criterion(NegativeBinaryMutualInformation())

                network = BinaryGAN(torch.device('cpu'), network_criterion)

                expectations = Expectations()
                expectations.add_expectation(PartialBivariateBinaryMutualInformation('I(X;Z)', 0), 0, lambda x,y: np.isclose(x, y))
                expectations.add_expectation(PartialBivariateBinaryMutualInformation('I(Y;Z)', 1), 1, lambda x,y: np.isclose(x, y))
                expectations.add_expectation(ComputeDistortion('E[hamm(x,y)]', 1).set_distortion_function(lambda x, y: hamming_distance(x, y).to(torch.float64)), 0.5, lambda x,y: x < y)

                experiment = Experiment(network, expectations)
                runs_results = experiment.run(data, n_splits=n_splits, epochs=epochs, batch_size=batch_size, verbose=True)
                results.setdefault(d, {}).setdefault(l, runs_results['averages'])

        return results


class GaussianExperiment(ExperimentRunner):

    def __init__(self):
        pass

    def run(self, data, epochs, batch_size, lambd, delta, k, verbose=False):
        results = {}
        n_splits = 3

        for d in delta:
            for l in lambd:
                for no_samples in k:
                    
                    network_criterion = PGANCriterion()
                    network_criterion.add_privacy_criterion(MeanSquaredError(lambd=l, delta_constraint=d)).add_privacy_criterion(GaussianMutualInformation())
                    network_criterion.add_adversary_criterion(NegativeGaussianMutualInformation())

                    network = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, network_criterion, 
                                    no_hidden_units_per_layer=hidden_layers_width, noise_size=5)

                    expectations = Expectations()
                    expectations.add_expectation(PartialMultivariateGaussianMutualInformation('E[I(X;Z)]', range(0, privacy_size)), 0, lambda x,y: np.isclose(x, y))
                    expectations.add_expectation(PartialMultivariateGaussianMutualInformation('E[I(Y;Z)]', range(privacy_size, privacy_size + public_size)), 1, lambda x,y: np.isclose(x, y))
                    expectations.add_expectation(ComputeDistortion('E[mse(z,y)]', range(privacy_size, privacy_size + public_size)).set_distortion_function(elementwise_mse), 0.5, lambda x,y: x < y)

                    experiment = Experiment(network, expectations)
                    runs_results = experiment.run(train_data, n_splits=n_splits, epochs=epochs, batch_size=batch_size, verbose=verbose, k=k)
                    results.setdefault(d, {}).setdefault(l, runs_results['averages'])

        return results