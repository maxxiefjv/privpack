"""
Generative adversarial networks to release Binary and Gaussian data,
this module defines the following classes:

- `GenerativeAdversarialNetwork`

Exception classes:

Functions:


How to use this module
======================
(See the individual classes, methods and attributes for more details)

1. Import .... TODO

2. Define a instance .... TODO

"""

import time

import copy
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from privpack import compute_released_data_statistics, compute_binary_released_set
from privpack import binary_bivariate_distortion_zy, binary_bivariate_mutual_information_zx, binary_bivariate_mutual_information_zy
from privpack import get_likelihood_xi_given_z

class GenerativeAdversarialNetwork(abc.ABC):

    """
    A Generative Adversarial Network defined using the PyTorch library.

    This abstract class expects one to implement an update adversary method as well
    as an update privatizer method. The naming is according to the goal of this library; release
    privatized data optimized in accordance to a privacy-utility trade-off.
    """

    def __init__(self, device, network_statistics, lr=1e-4):
        """
        Initialize a `GenerativeAdversarialNetwork` object.

        Parameters:

        - `device`: the device to be used: CPU or CUDA.
        - `Validator`: The Validator class used to evaluate the current state of the network.
        """
        self.lr = lr
        self.device = device
        self.network_statistic_functions = network_statistics

    def set_device(self, device):
        """
        Change the device used by this network.

        Parameters:

        - `device`: the device to be used: CPU or CUDA
        """
        self.device = device

    def get_device(self):
        """Get the device this network is currently using."""
        return self.device

    def get_privatizer(self):
        """Get a deep copy of the privatizer network that belongs to this network."""
        return copy.deepcopy(self.privatizer)

    def get_adversary(self):
        """Get a deep copy of the adversary network that belongs to this network."""
        return copy.deepcopy(self.adversary)

    @abc.abstractmethod
    def _update_adversary(self, entry, x_batch, y_batch):
        """
        Abstract method called during training of the network to update the adversary.
        """
        pass

    @abc.abstractmethod
    def _update_privatizer(self, entry, x_batch, y_batch):
        """
        Abstract method called during training of the network to update the privatizer.
        """
        pass

    def _get_network_statistics(self, train_data, test_data):
        if len(self.network_statistic_functions) == 0:
            return {}

        released_samples_train_data = compute_binary_released_set(self.get_privatizer(), train_data)
        released_samples_test_data = compute_binary_released_set(self.get_privatizer(), test_data)

        network_statistics_train = compute_released_data_statistics(released_samples_train_data, train_data, self.network_statistic_functions)
        network_statistics_test = compute_released_data_statistics(released_samples_test_data, test_data, self.network_statistic_functions)

        network_statistics = {
            'train': network_statistics_train,
            'test': network_statistics_test,
        }

        return network_statistics

    def _print_network_update(self, train_data, test_data, epoch, elapsed, adversary_loss, privatizer_loss):
        """
            private function created to print the core statistics of the network in its current state.
        """

        network_statistics = self._get_network_statistics(train_data, test_data)
        print('epoch: {}, time: {:.3f}s, Adversary loss: {:.3f}, Privatizer loss: {:.3f}'.format(
            epoch, elapsed, adversary_loss, privatizer_loss))

        if bool(network_statistics):
            print('{}'.format(network_statistics))

    def train(self, train_data, test_data, epochs,
              batch_size=1,
              privatizer_train_every_n=1,
              adversary_train_every_n=1,
              data_sampler=None, k=1):

        """
        Train the Generative Adversarial Network using the implemented privatizer and adversary network.
        The privatizer network and adversary network are both trained using the supplied `train_data`. However,
        the privatizer network is trained every nth batch-iteration supported by the `privatizer_train_every_n` parameter.
        Identically, the adversary network is trained only every nth batch-iteration supported by the `adversary_train_every_n`.
        Where both should be divisible by 5 due to the current logging system.1

        Parameters:

        - `train_data`: the training data used for training the generative adversarial network.
        - `test_data`: the testing data used for printing validation results on the generative adversarial network.
        - `batch_size`: The batch size used when training with the supplied `train_data`.
        - `privatizer_train_every_n`: Parameter defining when to update the privatizer network; Default=1.
        - `adversary_train_every_n`: Parameter defining when to update the adversary network; Default=1.
        - `data_sampler`: Function used for generating samples by the privatizer network.
        - `k`: The number of samples which should be generated by the supplied data_sampler function.

        """

        # For logging reasons only....
        if privatizer_train_every_n != 1 and privatizer_train_every_n % 5 != 0:
            raise Exception('Privatizer is constrained to be trained every or every fifth loop, but is trained every {} loop'.format(
                privatizer_train_every_n))

        if adversary_train_every_n != 1 and adversary_train_every_n % 5 != 0:
            raise Exception('Adversary is constrained to be trained every or every fifth loop, but is trained every {} loop'.format(
                privatizer_train_every_n))

        print('Using device:', self.device)
        print()

        start = time.time()

        adversary_loss = 0
        privatizer_loss = 0

        loader = DataLoader(train_data, batch_size=batch_size)
        for epoch in range(epochs):

            # Additional Info when using cuda
            if self.device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(
                    torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                print('Cached:   ', round(
                    torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
                print()

            self.adversary.train()
            self.privatizer.train()

            for i, sample in enumerate(loader):
                x_batch = sample[:, :self.privacy_size].to(self.device)
                y_batch = sample[:, self.privacy_size:].to(self.device)
                sample = sample.to(self.device)

                self.adversary_optimizer.zero_grad()
                self.privatizer_optimizer.zero_grad()

                if data_sampler is not None:
                    self.sampled_data = data_sampler(sample, k)

                if i % adversary_train_every_n == 0:
                    adversary_loss = self._update_adversary(sample, x_batch, y_batch)

                # Update privatizer
                if i % privatizer_train_every_n == 0:
                    privatizer_loss = self._update_privatizer(sample, x_batch, y_batch)

                # Elapsed time
                elapsed = time.time() - start  # Keep track of how much time has elapsed

                if i % 500 == 0:
                    self._print_network_update(train_data, test_data, epoch, elapsed, adversary_loss.item(), privatizer_loss.item())

class BinaryGenerativeAdversarialNetwork(GenerativeAdversarialNetwork):

    class _Privatizer(nn.Module):
        def __init__(self, ppan):
            super(BinaryGenerativeAdversarialNetwork._Privatizer, self).__init__()
            self.ppan = ppan
            self.model = nn.Sequential(
                nn.Linear(ppan.n_noise + ppan.inp_size, 1, bias=False),
                nn.Sigmoid()
            )

        def get_one_hot_encoded_input(self, w):
            w = w.view(-1, 2)
            x = w[:, 0]
            y = w[:, 1]
            one_hot_x = nn.functional.one_hot(x.to(torch.int64), 2)
            one_hot_y = nn.functional.one_hot(y.to(torch.int64), 2)

            concat_w = torch.cat((one_hot_x, one_hot_y), 1).float()
            return concat_w

        def forward(self, w):
            one_hot_encoded = self.get_one_hot_encoded_input(w)
            one_hot_encoded.to(self.ppan.device)
            return self.model(self.get_one_hot_encoded_input(w))

    class _Adversary(nn.Module):
        def __init__(self, ppan):
            super(BinaryGenerativeAdversarialNetwork._Adversary, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(1, 1, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    def __init__(self, device, privatizer_criterion, adversary_criterion, lr=1e-4):
        super().__init__(device, [
            binary_bivariate_mutual_information_zx,
            binary_bivariate_mutual_information_zy,
            binary_bivariate_distortion_zy
        ], lr=lr)

        self.n_noise = 0  # Size of the noise vector
        self.inp_size = 4
        self.privacy_size = 1

        self.adversary = self._Adversary(self).to(device)
        self.privatizer = self._Privatizer(self).to(device)

        self.adversary_optimizer = optim.Adam(
            self.adversary.parameters(), lr=self.lr)
        self.privatizer_optimizer = optim.Adam(
            self.privatizer.parameters(), lr=self.lr)

        self._privatizer_criterion = privatizer_criterion
        self._adversary_criterion = adversary_criterion

    def __str__(self) -> str:
        return "Binary Privacy-Preserving Adversarial Network"

    def reset(self) -> None:
        self.adversary = self._Adversary(self).to(self.device)
        self.privatizer = self._Privatizer(self).to(self.device)

        self.adversary_optimizer = optim.Adam(
            self.adversary.parameters(), lr=self.lr)
        self.privatizer_optimizer = optim.Adam(
            self.privatizer.parameters(), lr=self.lr)

    def _get_likelihoods(self, x_batch):
        x_likelihoods_given_zeros = get_likelihood_xi_given_z(self.adversary(torch.zeros(x_batch.size(0), 1)), x_batch)
        x_likelihoods_given_ones = get_likelihood_xi_given_z(self.adversary(torch.ones(x_batch.size(0), 1)), x_batch)
        x_likelihoods = torch.cat((x_likelihoods_given_zeros, x_likelihoods_given_ones), dim=1)
        return x_likelihoods

    def _update_adversary(self, entry, x_batch, y_batch):
        released = self.privatizer(entry).detach()
        x_likelihoods = self._get_likelihoods(x_batch)

        # Get sample mean
        adversary_loss = self._adversary_criterion(released, x_likelihoods, y_batch).mean()
        adversary_loss.backward()

        self.adversary_optimizer.step()

        return adversary_loss

    def _update_privatizer(self, entry, x_batch, y_batch):
        released = self.privatizer(entry)
        x_likelihoods = self._get_likelihoods(x_batch).detach()

        # Get sample mean
        privatizer_loss = self._privatizer_criterion(released, x_likelihoods, y_batch).mean()
        privatizer_loss.backward()

        self.privatizer_optimizer.step()

        return privatizer_loss

    def train(self, train_data, test_data, epochs, batch_size=1):
        return super().train(train_data, test_data, epochs, k=None,
                             batch_size=batch_size, data_sampler=None,
                             privatizer_train_every_n=5, adversary_train_every_n=1)
