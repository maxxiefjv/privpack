from gaussian_runner import GaussianNetworkRunner, get_gaussian_data
from binary_runner import BinaryNetworkRunner, get_binary_data

import argparse

def run_gaussian(args):
    (privacy_size, public_size, hidden_layers_width, release_size) = (5, 5, 20, 5)
    (epochs, batch_size, lambd, delta, k) = (args.epochs, args.batchsize, args.lambd, args.delta, args.sample)
    (train_data, test_data) = get_gaussian_data(privacy_size, public_size, print_metrics=True)
    
    GaussianNetworkRunner.run(train_data, test_data, privacy_size, public_size, hidden_layers_width, release_size, 
                              epochs, batch_size, lambd, delta, k)

def run_binary(args):
    (privacy_size, public_size, release_size) = (1, 1, 1)
    (epochs, batch_size, lambd, delta) = (args.epochs, args.batchsize, args.lambd, args.delta)
    (train_data, test_data) = get_binary_data(privacy_size, public_size, print_metrics=True)
    
    BinaryNetworkRunner.run(train_data, test_data, release_size, epochs, batch_size, lambd, delta)

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
                                 type=int,
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

