from runners import GaussianNetworkRunner, BinaryNetworkRunner
from experiments import BinaryExperiment, GaussianExperiment
from priv_bmi import BMIExperiment
from utils import get_binary_data, get_gaussian_data
from utils import save_results

from privpack import BinaryPrivacyPreservingAdversarialNetwork as BinaryGAN
from privpack import GaussianPrivacyPreservingAdversarialNetwork as GaussGAN



import argparse
import torch
import numpy as np
import json

def run_bmi(args):
    BMIExperiment().run(args)

def run_gaussian_1(args):
    (privacy_size, public_size, hidden_layers_width, release_size) = (1, 1, 5, 1)
    run_gaussian(privacy_size, public_size, hidden_layers_width, release_size, args)

def run_gaussian_5(args):
    (privacy_size, public_size, hidden_layers_width, release_size) = (5, 5, 20, 5)
    run_gaussian(privacy_size, public_size, hidden_layers_width, release_size, args)

def run_gaussian(privacy_size, public_size, hidden_layers_width, release_size, args):
    (epochs, batch_size, lambd, delta, k) = (args.epochs, args.batchsize, args.lambd, args.delta, args.sample)
    (train_data, test_data) = get_gaussian_data(privacy_size, public_size, args.train_input, print_metrics=True)

    if not k:
        k = [1]
    
    results = {}
    if len(lambd) == 1 and len(delta) == 1 and len(k) == 1:
        runner = GaussianNetworkRunner(privacy_size, public_size, hidden_layers_width, release_size, lambd[0], delta[0], lr=1e-2)
        results = runner.run(train_data, test_data, epochs, batch_size, k[0], verbose=True)
    else:
        runner = GaussianExperiment(privacy_size, public_size, hidden_layers_width, release_size)
        results = runner.run(train_data, epochs, batch_size, lambd, delta, k, verbose=True)

    print(json.dumps(results, sort_keys=True, indent=4))
    save_results(results, args)

def run_binary(args):
    (privacy_size, public_size, release_size) = (1, 1, 1)
    (epochs, batch_size, lambd, delta) = (args.epochs, args.batchsize, args.lambd, args.delta)
    (train_data, test_data) = get_binary_data(privacy_size, public_size, args.train_input, print_metrics=True)
    
    results = {}
    if len(lambd) == 1 and len(delta) == 1:
        runner = BinaryNetworkRunner(lambd[0], delta[0])
        results = runner.run(train_data, test_data, epochs, batch_size, verbose=True)
    else:
        runner = BinaryExperiment()
        results = runner.run(train_data, epochs, batch_size, lambd, delta, verbose=True)
    
    print(json.dumps(results, sort_keys=True, indent=4))
    save_results(results, args)


network_arg_switcher = {
    'binary': run_binary,
    'gaussian1': run_gaussian_1,
    'gaussian5': run_gaussian_5,
    'bmi': run_bmi
}

ap = argparse.ArgumentParser(description="""
This is an (example) implementation of the privpack library. Please consult the privpack documenation for the exact use of the 
arguments and options defined below.
""")

ap.add_argument('network', help="Define which implementation of the GAN defined in the privpack library to run.", 
                            metavar=list(network_arg_switcher.keys()),
                            choices=network_arg_switcher.keys())

ap.add_argument('-l', '--lambd', help="Define the lambda to use in the loss function. Train a network instance per value specified.",
                                  type=int,
                                  nargs='*',
                                  default=[500])

ap.add_argument('-d', '--delta', help="Define the delta to use in the loss function. Train a network instance per value specified.",
                                 type=float,
                                 nargs='*',
                                 default=np.linspace(1,0,11))

ap.add_argument('-k', '--sample', help="Only valid for gaussian network.Define the number of samples to be drawn from the privatizer network during training. Train a network instance per value specified.",
                                  type=int,
                                  nargs='*',
                                  metavar="NO. SAMPLES",
                                  default=None)
                                  
ap.add_argument('-b', '--batchsize', help="Define the number of samples used per minibatch iteration.",
                                     type=int,
                                     default=200)

ap.add_argument('-e', '--epochs', help="Define the number of epochs to run the training process.",
                                  type=int,
                                  default=500)

ap.add_argument('-o', '--output', help="Store the results in a specified file to json format. Default output is no output.",
                                  type=str,
                                  default=None)

ap.add_argument('-i', '--train-input', help="Specify the input to use in the training procedure.",
                                  type=str,
                                  default='uncorrelated')

ap.add_argument('--output-as-suffix', help='Use the output argument as suffix to the default generated outputname',
                                        default=False, dest='suffix',
                                        action='store_true')

def main():
    args = ap.parse_args()
    network_arg_switcher[args.network](args)

if __name__ == "__main__":
    main()
else:
    raise EnvironmentError('This is an (example) implementation of the privpack library and should not be used as library.')

