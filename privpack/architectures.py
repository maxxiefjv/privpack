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
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

from privpack import compute_released_data_statistics
from privpack import binary_bivariate_distortion_zy, binary_bivariate_mutual_information_zx, binary_bivariate_mutual_information_zy
from privpack import compute_mutual_information_gaussian_zy, compute_mutual_information_gaussian_zx, compute_mse_distortion_zy
from privpack import get_likelihood_xi_given_z
from privpack import sampled_data_from_network

class GenerativeAdversarialNetwork(abc.ABC):

    """
    A Generative Adversarial Network defined using the PyTorch library.

    This abstract class expects one to implement an update adversary method as well
    as an update privatizer method. The naming is according to the goal of this library; release
    privatized data optimized in accordance to a privacy-utility trade-off.
    """

    def __init__(self, device, privacy_size, public_size, network_statistics, lr=1e-4):
        """
        Initialize a `GenerativeAdversarialNetwork` object.

        Parameters:

        - `device`: the device to be used: CPU or CUDA.
        - `Validator`: The Validator class used to evaluate the current state of the network.
        """
        self.lr = lr
        self.device = device
        self.privacy_size = privacy_size
        self.public_size = public_size
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

    @abc.abstractmethod
    def _compute_released_set(self, data):
        pass

    def _get_network_statistics(self, train_data, test_data):
        if len(self.network_statistic_functions) == 0:
            return {}

        with torch.no_grad():
            released_samples_train_data = self._compute_released_set(train_data)
            released_samples_test_data = self._compute_released_set(test_data)

        network_statistics_train = compute_released_data_statistics(released_samples_train_data, train_data, self.network_statistic_functions, self.privacy_size)
        network_statistics_test = compute_released_data_statistics(released_samples_test_data, test_data, self.network_statistic_functions, self.privacy_size)

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
            print(json.dumps(network_statistics, sort_keys=True, indent=4))

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
                    self.sampled_data = data_sampler(self.privatizer, sample, k)

                if i % adversary_train_every_n == 0:
                    adversary_loss = self._update_adversary(sample, x_batch, y_batch)

                # Update privatizer
                if i % privatizer_train_every_n == 0:
                    privatizer_loss = self._update_privatizer(sample, x_batch, y_batch)

                # Elapsed time
                elapsed = time.time() - start  # Keep track of how much time has elapsed

                if i % 1000 == 0:
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

    def __init__(self, device, privatizer_criterion, adversary_criterion, lr=1e-4,
                 privatizer_train_every_n=5, adversary_train_every_n=1):
        super().__init__(device, 1, 1, [
            binary_bivariate_mutual_information_zx,
            binary_bivariate_mutual_information_zy,
            binary_bivariate_distortion_zy
        ], lr=lr)

        self.n_noise = 0  # Size of the noise vector
        self.inp_size = 4

        self.privatizer_train_every_n = privatizer_train_every_n
        self.adversary_train_every_n = adversary_train_every_n

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

    def _compute_released_set(self, data):
        released_data = self.privatizer(data)
        random_uniform_tensor = torch.rand(released_data.size())
        return torch.ceil(released_data - random_uniform_tensor).to(torch.int).view(-1, 1)

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
                             privatizer_train_every_n=self.privatizer_train_every_n, adversary_train_every_n=self.adversary_train_every_n)

class GaussianGenerativeAdversarialNetwork(GenerativeAdversarialNetwork):

    class _Privatizer(nn.Module):
        def __init__(self, ppan):
            super(GaussianGenerativeAdversarialNetwork._Privatizer, self).__init__()
            self.ppan = ppan

            self.model = nn.Sequential(

                nn.Linear(ppan.n_noise + ppan.privacy_size + ppan.public_size,
                          ppan.no_hidden_units_per_layer, bias=False),
                nn.BatchNorm1d(num_features=ppan.no_hidden_units_per_layer),
                nn.ReLU(),
                #
                nn.Linear(ppan.no_hidden_units_per_layer,
                          ppan.no_hidden_units_per_layer, bias=False),
                nn.BatchNorm1d(num_features=ppan.no_hidden_units_per_layer),
                nn.ReLU(),
                #
                nn.Linear(ppan.no_hidden_units_per_layer,
                          ppan.release_size, bias=False)
            )

        def forward(self, x):
            noise = torch.rand(x.size(0), self.ppan.n_noise)
            inp = torch.cat((x, noise), dim=1)
            return self.model(inp)

    class _Adversary(nn.Module):
        def __init__(self, ppan):
            super(GaussianGenerativeAdversarialNetwork._Adversary, self).__init__()
            self.inp_size = ppan.release_size
            self.out_size = self.inp_size * 2

            self.model = nn.Sequential(
                nn.Linear(self.inp_size, ppan.no_hidden_units_per_layer, bias=False),
                nn.BatchNorm1d(num_features=ppan.no_hidden_units_per_layer),
                nn.ReLU(),
                #
                nn.Linear(ppan.no_hidden_units_per_layer,
                          ppan.no_hidden_units_per_layer, bias=False),
                nn.BatchNorm1d(num_features=ppan.no_hidden_units_per_layer),
                nn.ReLU(),
                #
                # Produce Mu and Sigma.
                nn.Linear(ppan.no_hidden_units_per_layer,
                          self.out_size, bias=False)
            )

        def get_output_size(self):
            return self.out_size

        def forward(self, x):
            return self.model(x)

    def __init__(self, device, privacy_size, public_size, release_size,
                 privatizer_criterion, adversary_criterion, noise_size=5,
                 no_hidden_units_per_layer=20,
                 lr=1e-3, privatizer_train_every_n=5, adversary_train_every_n=1):

        super().__init__(device, privacy_size, public_size, [
            compute_mutual_information_gaussian_zx,
            compute_mutual_information_gaussian_zy,
            compute_mse_distortion_zy
        ], lr)
        self.no_hidden_units_per_layer = no_hidden_units_per_layer
        self.n_noise = noise_size
        self.release_size = release_size

        self.privatizer_train_every_n = privatizer_train_every_n
        self.adversary_train_every_n = adversary_train_every_n

        self.device = device

        self.adversary = self._Adversary(self).to(device)
        self.privatizer = self._Privatizer(self).to(device)

        self.adversary_optimizer = optim.Adam(
            self.adversary.parameters(), lr=self.lr)
        self.privatizer_optimizer = optim.Adam(
            self.privatizer.parameters(), lr=self.lr)

        self._privatizer_criterion = privatizer_criterion
        self._adversary_criterion = adversary_criterion

        self.mus = torch.Tensor([])
        self.covs = torch.Tensor([])

    def __str__(self):
        return '''Gaussian Mixture Model:
        \n
        Learning Rate: {}
        Train Privatizer Every nth: {}
        Train Adversary Every nth: {}

        -----------------------------\n
        {}\n
        {}\n
        '''.format(self.lr, self.privatizer_train_every_n, self.adversary_train_every_n,
                   self.privatizer, self.adversary)

    def reset(self):
        self.adversary = self._Adversary(self).to(self.device)
        self.privatizer = self._Privatizer(self).to(self.device)

        self.adversary_optimizer = optim.Adam(
            self.adversary.parameters(), lr=self.lr)
        self.privatizer_optimizer = optim.Adam(
            self.privatizer.parameters(), lr=self.lr)

        self.mus = torch.Tensor([])
        self.covs = torch.Tensor([])

    def _compute_released_set(self, data):
        released_set = self.privatizer(data)
        return released_set

    def _get_adversary_distribution(self, adversary_out):
        adv_mu = adversary_out[:int(self.adversary.get_output_size() / 2)]
        adv_cov = torch.square(adversary_out[int(self.adversary.get_output_size() / 2):])

        try:
            # print(adv_mu, adv_cov)
            normal = MultivariateNormal(adv_mu, torch.diag(adv_cov))
        except Exception:
            print(adv_mu, adv_cov)
            adv_cov = torch.square(torch.randn_like(adv_cov)) + adv_cov
            adv_mu = torch.randn_like(adv_mu) + adv_mu
            normal = MultivariateNormal(adv_mu, torch.diag(adv_cov))

        self.mus = torch.cat((self.mus, adv_mu.detach()), dim=0) if len(self.mus) == 0 else adv_mu.detach()
        self.covs = torch.cat((self.covs, adv_cov.detach()), dim=0) if len(self.covs) == 0 else adv_cov.detach()

        return normal

    def _get_log_prob(self, adversary_out, xi):
        adv_multivariate_output = self._get_adversary_distribution(adversary_out)
        return adv_multivariate_output.log_prob(xi.float())

    def _get_log_likelihoods(self, released, x_batch):
        adversary_out = self.adversary(released)
        neg_log_likelihoods_Xi_given_z = torch.Tensor([]).to(self.device)
        for (out_entry, xi) in zip(adversary_out, x_batch):
            neg_log_likelihood_Xi_given_z = -self._get_log_prob(out_entry, xi).view(1, -1)
            neg_log_likelihoods_Xi_given_z = torch.cat((neg_log_likelihoods_Xi_given_z, neg_log_likelihood_Xi_given_z), dim=0)

        return neg_log_likelihoods_Xi_given_z

    def _get_expected_log_likelihoods(self, released_samples, x_batch):
        expected_loglikelihood = 0
        for release in released_samples:
            neg_log_likelihood_Xi_given_z = self._get_log_likelihoods(release, x_batch)
            expected_loglikelihood += neg_log_likelihood_Xi_given_z / len(released_samples)

        return expected_loglikelihood

    def _update_adversary(self, entry, x_batch, y_batch):
        released_samples = self.sampled_data.detach()
        x_log_likelihoods = self._get_expected_log_likelihoods(released_samples, x_batch)

        # Sample mean loss
        adversary_loss = self._adversary_criterion(released_samples, x_log_likelihoods, y_batch).mean()
        adversary_loss.backward()

        self.adversary_optimizer.step()

        return adversary_loss

    def _update_privatizer(self, entry, x_batch, y_batch):
        released_samples = self.sampled_data
        x_log_likelihoods = self._get_expected_log_likelihoods(released_samples, x_batch)

        # Sample mean loss
        privatizer_loss = self._privatizer_criterion(released_samples, x_log_likelihoods, y_batch).mean()
        privatizer_loss.backward()

        self.privatizer_optimizer.step()

        return privatizer_loss

    def train(self, train_data, test_data, epochs, k=1, batch_size=2):
        return super().train(train_data, test_data, epochs, k=k,
                             batch_size=batch_size, data_sampler=sampled_data_from_network,
                             privatizer_train_every_n=self.privatizer_train_every_n,
                             adversary_train_every_n=self.adversary_train_every_n)
