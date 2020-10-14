from privpack import BinaryPrivacyPreservingAdversarialNetwork as BinaryGAN
from privpack.core.criterion import PGANCriterion
from privpack.core.criterion import BinaryHammingDistance, BinaryMutualInformation, NegativeBinaryMutualInformation

from privpack.model_selection.experiment import Experiment, Expectations
from privpack.utils.metrics import PartialBivariateBinaryMutualInformation, ComputeDistortion
from privpack.utils import hamming_distance
from privpack.utils import DataGenerator

import numpy as np
import torch 

def test_experiment(uncorrelated_train_and_test_data):
    (train_data, test_data) = uncorrelated_train_and_test_data

    (lambd, delta_constraint) = (500, 0.5)
    (epochs, batch_size) = (1, 200)
    n_splits = 2
    
    network_criterion = PGANCriterion()
    network_criterion.add_privacy_criterion(BinaryHammingDistance(lambd, delta_constraint)).add_privacy_criterion(BinaryMutualInformation())
    network_criterion.add_adversary_criterion(NegativeBinaryMutualInformation())

    network = BinaryGAN(torch.device('cpu'), network_criterion)

    expectations = Expectations()
    expectations.add_expectation(PartialBivariateBinaryMutualInformation('I(X;Z)', 0), 0, lambda x,y: np.isclose(x,y))
    expectations.add_expectation(PartialBivariateBinaryMutualInformation('I(Y;Z)', 1), 1, lambda x,y: np.isclose(x,y))
    expectations.add_expectation(ComputeDistortion('E[hamm(x,y)]', 1).set_distortion_function(lambda x, y: hamming_distance(x, y).to(torch.float64)), 0.5, lambda x,y: x < y)

    experiment = Experiment(network, expectations)
    runs_results = experiment.run(train_data, n_splits=n_splits, epochs=epochs, batch_size=batch_size)
    
    assert list(runs_results.keys()) == list(range(n_splits)) + ['averages']