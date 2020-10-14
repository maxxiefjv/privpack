from runners import GaussianNetworkRunner, get_gaussian_data
from runners import BinaryNetworkRunner, get_binary_data

from privpack import BinaryPrivacyPreservingAdversarialNetwork as BinaryGAN
from privpack import GaussianPrivacyPreservingAdversarialNetwork as GaussGAN

from privpack.core.criterion import PGANCriterion
from privpack.core.criterion import BinaryHammingDistance, BinaryMutualInformation, NegativeBinaryMutualInformation
from privpack.core.criterion import MeanSquaredError, GaussianMutualInformation, NegativeGaussianMutualInformation

from privpack.model_selection.experiment import Experiment, Expectations
from privpack.utils.metrics import PartialBivariateBinaryMutualInformation, PartialMultivariateGaussianMutualInformation, ComputeDistortion

from privpack.utils import hamming_distance, elementwise_mse

import argparse
import torch
import numpy as np
import json

def run_gaussian(args):
    (privacy_size, public_size, hidden_layers_width, release_size) = (5, 5, 20, 5)
    (epochs, batch_size, lambd, delta, k) = (args.epochs, args.batchsize, args.lambd, args.delta, args.sample)
    (train_data, test_data) = get_gaussian_data(privacy_size, public_size, print_metrics=True)
    
    results = {}
    if len(lambd) == 1 and len(delta) == 1 and len(k) == 1:
        runner = GaussianNetworkRunner(privacy_size, public_size, hidden_layers_width, release_size, lambd[0], delta[0])
        metric_results = runner.run(train_data, test_data, epochs, batch_size, k[0])
        results.setdefault(l, {}).setdefault(d, metric_results)
        
        print(json.dumps(results, sort_keys=True, indent=4))
        return

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
                runs_results = experiment.run(train_data, n_splits=n_splits, epochs=epochs, batch_size=batch_size)
                results.setdefault(d, {}).setdefault(l, runs_results['averages'])

    print(json.dumps(results, sort_keys=True, indent=4))


def run_binary(args):
    (privacy_size, public_size, release_size) = (1, 1, 1)
    (epochs, batch_size, lambd, delta) = (args.epochs, args.batchsize, args.lambd, args.delta)
    (train_data, test_data) = get_binary_data(privacy_size, public_size, print_metrics=True)
    results = {}

    if len(lambd) == 1 and len(delta) == 1:
        runner = BinaryNetworkRunner(lambd[0], delta[0])
        metric_results = runner.run(train_data, test_data, epochs, batch_size)
        results.setdefault(l, {}).setdefault(d, metric_results)
        
        print(json.dumps(results, sort_keys=True, indent=4))
        return
    
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
            runs_results = experiment.run(train_data, n_splits=n_splits, epochs=epochs, batch_size=batch_size)
            results.setdefault(d, {}).setdefault(l, runs_results['averages'])

    print(json.dumps(results, sort_keys=True, indent=4))
    

network_arg_switcher = {
    'binary': run_binary,
    'gaussian': run_gaussian
}

ap = argparse.ArgumentParser(description="""
This is an (example) implementation of the privpack library. Please consult the privpack documenation for the exact use of the 
arguments and options defined below.
""")

ap.add_argument('network', help="Define which implementation of the GAN defined in the privpack library to run.", 
                            metavar="{binary, gaussian}",
                            choices=network_arg_switcher.keys())

ap.add_argument('-l', '--lambd', help="Define the lambda to use in the loss function. Train a network instance per value specified.",
                                  type=int,
                                  nargs='*',
                                  default=[500])

ap.add_argument('-d', '--delta', help="Define the delta to use in the loss function. Train a network instance per value specified.",
                                 type=float,
                                 nargs='*',
                                 default=[0.5])

ap.add_argument('-k', '--sample', help="Only valid for gaussian network.Define the number of samples to be drawn from the privatizer network during training. Train a network instance per value specified.",
                                  type=int,
                                  nargs='*',
                                  metavar="NO. SAMPLES",
                                  default=[1])
                                  
ap.add_argument('-b', '--batchsize', help="Define the number of samples used per minibatch iteration.",
                                     type=int,
                                     default=200)

ap.add_argument('-e', '--epochs', help="Define the number of epochs to run the training process.",
                                  type=int,
                                  default=500)


def main():
    args = ap.parse_args()
    network_arg_switcher[args.network](args)

if __name__ == "__main__":
    main()
else:
    raise EnvironmentError('This is an (example) implementation of the privpack library and should not be used as library.')

