from privpack.utils import DataGenerator
from privpack.utils.metrics import BivariateBinaryMutualInformation, MultivariateGaussianMutualInformation

import os.path
import json

import torch
from torch.distributions import MultivariateNormal

import numpy as np
import scipy

gaussian_data = {
    'uncorrelated': DataGenerator.get_completely_uncorrelated_distribution_params,
    'ppan': DataGenerator.get_ppan_distribution_params
}

binary_data = {
    'uncorrelated': DataGenerator.get_completely_uncorrelated_dist,
    'correlated': DataGenerator.get_completely_correlated_dist,
    'random': DataGenerator.random_binary_dist
}

def get_gaussian_data(privacy_size, public_size, train_input_name, print_metrics=False):
    if not train_input_name in gaussian_data:
        raise RuntimeError('Train data input {} does not exist. For gaussian data we provide: {}'.format(train_input_name, gaussian_data.keys()))

    (mu, cov) = gaussian_data[train_input_name](privacy_size, public_size)
    (train_data, test_data) = DataGenerator.generate_gauss_mixture_data(mu, cov, seed=0)

    if print_metrics:
        print_gaussian_data_statistics(mu, cov, privacy_size, public_size, train_data, test_data)

    return (train_data, test_data)

def get_binary_data(privacy_size, public_size, train_input_name, print_metrics=False):
    if not train_input_name in binary_data:
        raise RuntimeError('Train data input {} does not exist. For binary data we provide: {}'.format(train_input_name, binary_data.keys()))
    
    (norm_dist, acc_dist) = binary_data[train_input_name]()
    synthetic_data = DataGenerator.generate_binary_data(10000, acc_dist)

    no_train_samples = 8000
    train_data = torch.Tensor(synthetic_data[:no_train_samples])
    test_data = torch.Tensor(synthetic_data[no_train_samples:])
    
    if print_metrics:
        print_binary_data_statistics(norm_dist, train_data, test_data)

    return (train_data, test_data)

def print_gaussian_data_statistics(mu, cov, x_dim, y_dim, train_data, test_data):
    act_mi = MultivariateGaussianMutualInformation('').compute_mutual_information(cov, x_dim)
    est_cov = np.cov(train_data.T)
    est_mi = MultivariateGaussianMutualInformation('').compute_mutual_information(est_cov, x_dim)

    print("Gaussian Mixture - Actual Mutual Information: {}, \
                            \nEstimated Mutual Information from generated training data: {}".format(act_mi, est_mi))

    multivariate_normal = MultivariateNormal(mu[y_dim:], cov[y_dim:, y_dim:])
    print("Entropy of Y: {}\n".format(multivariate_normal.entropy()))


def print_binary_data_statistics(norm_dist, train_data, test_data):

    num_y1_samples_train = sum([1 for x in train_data if x[1] == 1])
    num_y0_samples_train = len(train_data) - num_y1_samples_train

    print("The binary training data consists of " + str(num_y1_samples_train) +
            " y=1 samples" + " ({}%)".format(100 * (num_y1_samples_train / len(train_data))))
    print("The binary training data consists of " + str(num_y0_samples_train) +
            " y=0 samples" + " ({}%)".format(100 * (num_y0_samples_train / len(train_data))))

    num_y1_samples_test = sum([1 for x in test_data if x[1] == 1])
    num_y0_samples_test = len(test_data) - num_y1_samples_test
    print("The binary test data consists of " + str(num_y1_samples_test) +
            " y=1 samples" + " ({}%)".format(100 * (num_y1_samples_test / len(test_data))))
    print("The binary test data consists of " + str(num_y0_samples_test) +
            " y=0 samples" + " ({}%)".format(100 * (num_y0_samples_test / len(test_data))))

    est_dist = BivariateBinaryMutualInformation('').estimate_binary_distribution(train_data)
    P_x = [sum(est_dist[0, :]), sum(est_dist[1, :])]
    P_y = [sum(est_dist[:, 0]), sum(est_dist[:, 1])]
    act_P_y = [sum(norm_dist[:, 0]), sum(norm_dist[:, 1])]

    enotrpy_y = sum(scipy.special.entr(P_y)) / np.log(2)
    act_enotrpy_y = sum(scipy.special.entr(act_P_y)) / np.log(2)

    print("Estimated entropy P_y = {}".format(enotrpy_y))
    print("Actual entropy P_y = {}".format(act_enotrpy_y))

    est_mi = BivariateBinaryMutualInformation('').compute_mutual_information(est_dist)
    act_mi = BivariateBinaryMutualInformation('').compute_mutual_information(norm_dist)
    print("Binary data - Actual Mutual Information: {}, Estimated Mutual Information from generated training data: {}".format(act_mi, est_mi))

def suffix_outputname(outputname, suffix_name, suffix):
    outputname += '_{}'.format(suffix_name)
    if type(suffix) == list:
        return outputname + '_[' + ','.join(map(str, suffix)) + ']'

    return outputname + '_' + str(suffix)

def generate_output_filename(args):
    outputname = args.network
    outputname = suffix_outputname(outputname, 'lambda', args.lambd)
    outputname = suffix_outputname(outputname, 'delta', args.delta)
    outputname = suffix_outputname(outputname, 'no-samples',  args.sample) if args.sample else outputname
    outputname = suffix_outputname(outputname, 'batchsize', args.batchsize)
    outputname = suffix_outputname(outputname, 'epochs', args.epochs)
    outputname = suffix_outputname(outputname, 'data-type', args.train_input)

    return outputname

def validate_outputname(path, idx=1):
    if os.path.exists(path + '.json'):
        if (idx > 1):
            path = path.split('(')[0][:-1]
        return validate_outputname(path + ' ({})'.format(idx), idx + 1)

    return path

def save_results(results, args):
    if not os.path.isdir('results'):
        os.mkdir('results')

    if args.suffix:
        path = generate_output_filename(args)
        path = 'results/' + path
        path += '_' + args.output if args.output else ''
    else:
        path = args.output if args.output else generate_output_filename(args)
        path = 'results/' + path

    path = validate_outputname(path)

    json.dump( results, open( path + '.json', 'w' ), indent=4 )
