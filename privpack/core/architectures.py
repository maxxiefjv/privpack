"""
Generative adversarial networks to release Binary and Gaussian data,
this module defines the following classes:

- `GenerativeAdversarialNetwork`
- `BinaryPrivacyPreservingAdversarialNetwork`
- `GaussianPrivacyPreservingAdversarialNetwork`

"""

import time

import copy
import abc
import json

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

from privpack.utils import compute_released_data_metrics
from privpack.utils import (
    ComputeDistortion, PartialBivariateBinaryMutualInformation, PartialMultivariateGaussianMutualInformation,
    hamming_distance, elementwise_mse
)
from privpack.utils import get_likelihood_xi_given_z
from privpack.utils import sample_from_network

from privpack.core.criterion import PGANCriterion

class GenerativeAdversarialNetwork(abc.ABC):

    """
    A Generative Adversarial Network defined using the PyTorch library.

    This abstract class expects one to implement an update adversary method as well
    as an update privatizer method. The naming is according to the goal of this library; release
    privatized data optimized in accordance to a privacy-utility trade-off.
    """

    def __init__(self, device, privacy_size, public_size, gan_criterion: PGANCriterion, metrics, lr=1e-3):
        """
        Initialize a `GenerativeAdversarialNetwork` object.

        Parameters:

        - `device`: the device to be used: CPU or CUDA.
        - `privacy_size`: The number of dimensions considered private parts.
        - `public_size`: The number of dimensions considered public parts.
        - `metrics`: A list of metric used to compute the performance of the network.
        """
        self.device = device
        self.privacy_size = privacy_size
        self.public_size = public_size
        self.metrics = metrics
        self.lr = lr

        self.gan_criterion = gan_criterion
        self._privatizer_criterion = gan_criterion.privacy_loss
        self._adversary_criterion = gan_criterion.adversary_loss

    def __call__(self, data):
        return self.privatize(data)

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

    @abc.abstractclassmethod
    def reset(self):
        """
        Reset the parameters of both networks. 
        """
        pass

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
    def privatize(self, data):
        """
        Privatize the provided data using the privatizer in the network.

        Parameters:

        - `data`: data to be privatized by the network.

        return the privatized version of the provided data.
        """
        pass

    def _get_metric_results(self, train_data, test_data):
        if len(self.metrics) == 0:
            return {}

        with torch.no_grad():
            released_samples_train_data = self.privatize(train_data)
            released_samples_test_data = self.privatize(test_data)

        metric_results_train = compute_released_data_metrics(released_samples_train_data, train_data, self.metrics)
        metric_results_test = compute_released_data_metrics(released_samples_test_data, test_data, self.metrics)

        metric_results = {
            'train': metric_results_train,
            'test': metric_results_test,
        }

        return metric_results

    def _print_network_update(self, train_data, test_data, epoch, elapsed, adversary_loss, privatizer_loss):
        """
            private function created to print the core metrics of the network in its current state.
        """

        metric_results = self._get_metric_results(train_data, test_data)
        print('epoch: {}, time: {:.3f}s, Adversary loss: {:.3f}, Privatizer loss: {:.3f}'.format(
            epoch, elapsed, adversary_loss, privatizer_loss))

        if bool(metric_results):
            print(json.dumps(metric_results, sort_keys=True, indent=4))

    def train(self, train_data, test_data, epochs,
              batch_size=1,
              privatizer_train_every_n=1,
              adversary_train_every_n=1,
              data_sampler=None, k=1, verbose=False):

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
        - `lr`: Learning Rate indicating the step-size of adjusting the network parameters.
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

                if verbose and i % 1000 == 0:
                    self._print_network_update(train_data, test_data, epoch, elapsed, adversary_loss.item(), privatizer_loss.item())

class BinaryPrivacyPreservingAdversarialNetwork(GenerativeAdversarialNetwork):
    """
    A Binary implementation of the Generative Adversarial Network defined using the PyTorch library.

    This class implements the Generative Adversarial Base framework and thereby defines this class
    to produce a single privacy preserved binary output given two binary inputs. This is done using the
    defined privatizer network. The adversary network estimates the probability of the original private value.
    How the networks are to learn the best outputs is learned using user provided criterions. It is expected
    to be according to an optimal Privacy-Utility Trade-off.
    """

    class _Privatizer(nn.Module):
        """
        Privatizer network consisting of a single linear transformation followed by the non-linear
        Sigmoid activation.
        """
        def __init__(self, ppan):
            super().__init__()
            self.ppan = ppan
            self.model = nn.Sequential(
                nn.Linear(ppan.n_noise + ppan.inp_size, 1, bias=False),
                nn.Sigmoid()
            )

        def get_one_hot_encoded_input(self, w):
            """
                Transform ourdata 2D w to a one-hot encoded alternative:

                W ->(x==0,x==1,y==0,y==1)
                (0,0) -> (1, 0, 1, 0)
                (1,0) -> (0, 1, 1, 0)
                (0,1) -> (1, 0, 0, 1)
                (1,1) -> (0, 1, 0, 1)
            """
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
        """
        Adversary network consisting of a single linear transformation followed by the non-linear
        Sigmoid activation
        """
        def __init__(self, ppan):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(1, 1, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    def __init__(self, device, binary_gan_criterion: PGANCriterion, lr=1e-2):
        """
        The behavior of the `BinaryPrivacyPreservingAdversarialNetwork` is mostly defined by the privatizer
        and adversary criterion provided on init.

        Parameters:

        - `privatizer_criterion`: Identifies how to compute the loss of the privatizer netwowrk.
        - `adversary_criterion`: Identifies how to compute the loss of the adversary netwowrk.
        """
        super().__init__(device, privacy_size=1, public_size=1, gan_criterion=binary_gan_criterion, metrics=[
            PartialBivariateBinaryMutualInformation('E[I(X;Z)]', 0),
            PartialBivariateBinaryMutualInformation('E[I(Y;Z)]', 1),
            ComputeDistortion('E[hamm(x,y)]', 1).set_distortion_function(lambda x, y: hamming_distance(x, y).to(torch.float64))
        ], lr=lr)

        self.n_noise = 0  # Size of the noise vector
        self.inp_size = 4

        self.adversary = self._Adversary(self).to(device)
        self.privatizer = self._Privatizer(self).to(device)

        self.adversary.apply(self._weights_init)
        self.privatizer.apply(self._weights_init)

        self.adversary_optimizer = optim.Adam(
            self.adversary.parameters(), lr=self.lr)
        self.privatizer_optimizer = optim.Adam(
            self.privatizer.parameters(), lr=self.lr)

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 1)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 1)

    def __str__(self) -> str:
        """
        Informal string representation of this class.
        """
        return "Binary Privacy-Preserving Adversarial Network"

    def reset(self) -> None:
        """
        Provides the possibility to undo all the learned parameters.
        """
        self.adversary = self._Adversary(self).to(self.device)
        self.privatizer = self._Privatizer(self).to(self.device)

        self.adversary_optimizer = optim.Adam(
            self.adversary.parameters(), lr=self.lr)
        self.privatizer_optimizer = optim.Adam(
            self.privatizer.parameters(), lr=self.lr)

    def save(self):
        """
        Save the networks learned parameters.
        """
        raise NotImplementedError("Save function not yet implemented")

    def load(self):
        """
        Load the parameters of the privatizer and adversary network.
        """
        raise NotImplementedError("Load function not yet implemented")

    def privatize(self, data):
        released_data = self.privatizer(data)
        random_uniform_tensor = torch.rand(released_data.size())
        return torch.ceil(released_data - random_uniform_tensor).to(torch.int).view(-1, 1)

    def _get_likelihoods(self, x_batch):
        """
        Get the likelihoods of the provided x_batch for both values of the possible releases: {0,1}.

        Parameters:

        - `x_batch`: Compute the likelihood of this data estimated by the adversary

        return the likelihoods for the provided data given both possible releases.
        """
        x_likelihoods_given_zeros = get_likelihood_xi_given_z(self.adversary(torch.zeros(x_batch.size(0), 1)), x_batch)
        x_likelihoods_given_ones = get_likelihood_xi_given_z(self.adversary(torch.ones(x_batch.size(0), 1)), x_batch)
        x_likelihoods = torch.cat((x_likelihoods_given_zeros, x_likelihoods_given_ones), dim=1)
        return x_likelihoods

    def _update_adversary(self, entry, x_batch, y_batch):
        """
        This function is called every, depending on the train parameters, nth iteration. It is
        used to update the adversary network, and should do so using: the full entry, private and public parts.

        Parameters:

        - `entry`: concatenated version of (x_batch,y_batch).
        - 'x_batch`: private parts of entry.
        - `y_batch`: public parts of entry.

        return the adversary loss computed by `adversary_criterion`
        """
        released = self.privatizer(entry).detach()
        x_likelihoods = self._get_likelihoods(x_batch)

        # Get sample mean
        adversary_loss = self._adversary_criterion(released, x_likelihoods, y_batch).mean()
        adversary_loss.backward()

        self.adversary_optimizer.step()

        return adversary_loss

    def _update_privatizer(self, entry, x_batch, y_batch):
        """
        This function is called every, depending on the train parameters, nth iteration. It is
        used to update the privatizer network, and should do so using: the full entry, private and public parts.

        Parameters:

        - `entry`: concatenated version of (x_batch,y_batch).
        - 'x_batch`: private parts of entry.
        - `y_batch`: public parts of entry.

        return the adversary loss computed by `privatizer_criterion`
        """
        released = self.privatizer(entry)
        x_likelihoods = self._get_likelihoods(x_batch).detach()

        # Get sample mean
        privatizer_loss = self._privatizer_criterion(released, x_likelihoods, y_batch).mean()
        privatizer_loss.backward()

        self.privatizer_optimizer.step()

        return privatizer_loss

    def train(self, train_data, test_data, epochs, batch_size=1, privatizer_train_every_n=5, adversary_train_every_n=1, verbose=False):
        return super().train(train_data, test_data, epochs, k=None,
                             batch_size=batch_size, data_sampler=None,
                             privatizer_train_every_n=privatizer_train_every_n,
                             adversary_train_every_n=adversary_train_every_n, verbose=verbose)

class GaussianPrivacyPreservingAdversarialNetwork(GenerativeAdversarialNetwork):
    """
    A Gaussian implementation of the Generative Adversarial Network defined using the PyTorch library.

    This class implements the Generative Adversarial Base framework and thereby defines this class
    to produce privatized gaussian outputs given assumed to be gaussian inputs. This is done using the
    defined privatizer network. The adversary network estimates the probability of the original private value.
    How the networks are to learn the best outputs is learned using user provided criterions. It is expected
    to be according to an optimal Privacy-Utility Trade-off.
    """
    class _Privatizer(nn.Module):
        def __init__(self, ppan):
            super(GaussianPrivacyPreservingAdversarialNetwork._Privatizer, self).__init__()
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
            super(GaussianPrivacyPreservingAdversarialNetwork._Adversary, self).__init__()
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
                 gauss_gan_criterion, lr=1e-3, noise_size=5,
                 no_hidden_units_per_layer=20):
        """
        The behavior of the `GaussianPrivacyPreservingAdversarialNetwork` is mostly defined by the privatizer
        and adversary criterion provided on init. This defines how to learn the best Privacy-Utility Trade-off
        parameters for the privatizer and adversary network.

        Data used in this network should be private data first then public data. e.g. Consider W to consist of
        private (X) and public (Y) data then: W <=> (X,Y).

        Parameters:

        - `privacy_size`: The number of private dimensions in the data used with this network.
        - `public_size`: The number of public dimensions in the data used with this network.
        - `release_size`: The number of dimensions the privatizer network should produce.
        - `privatizer_criterion`: Identifies how to compute the loss of the privatizer network.
        - `adversary_criterion`: Identifies how to compute the loss of the adversary network.
        - `noise_size`: The number of noise dimension to add to the input of the privatizer network. Needed
        for the universal approximator mechanisms to work.
        - `no_hidden_units_per_layer`: Every layer in the gaussian network will have the number of nodes defined
        by this parameter.
        """
        super().__init__(device, privacy_size, public_size, gan_criterion=gauss_gan_criterion, metrics=[
            PartialMultivariateGaussianMutualInformation('E[I(X;Z)]', range(0, privacy_size)),
            PartialMultivariateGaussianMutualInformation('E[I(Y;Z)]', range(privacy_size, privacy_size + public_size)),
            ComputeDistortion('E[mse(z,y)]', range(privacy_size, privacy_size + public_size)).set_distortion_function(elementwise_mse)
        ], lr=lr)

        self.no_hidden_units_per_layer = no_hidden_units_per_layer
        self.n_noise = noise_size
        self.release_size = release_size

        self.device = device

        self.adversary = self._Adversary(self).to(device)
        self.privatizer = self._Privatizer(self).to(device)

        self.adversary.apply(self._weights_init)
        self.privatizer.apply(self._weights_init)

        self.adversary_optimizer = optim.Adam(
            self.adversary.parameters(), lr=self.lr)
        self.privatizer_optimizer = optim.Adam(
            self.privatizer.parameters(), lr=self.lr)

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

        self.adversary.apply(self._weights_init)
        self.privatizer.apply(self._weights_init)

        self.adversary_optimizer = optim.Adam(
            self.adversary.parameters(), lr=self.lr)
        self.privatizer_optimizer = optim.Adam(
            self.privatizer.parameters(), lr=self.lr)

        self.mus = torch.Tensor([])
        self.covs = torch.Tensor([])

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 1)
        elif classname.find('Linear') != -1:
            # print(m.weight)
            torch.nn.init.normal_(m.weight, 0.0, 1)

    def privatize(self, data):
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
            self.adversary = self._Adversary(self).to(self.device)
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

    def train(self, train_data, test_data, epochs, k=1, batch_size=2, privatizer_train_every_n=5, adversary_train_every_n=1, verbose=False):
        return super().train(train_data, test_data, epochs, k=k,
                             batch_size=batch_size, data_sampler=sample_from_network,
                             privatizer_train_every_n=privatizer_train_every_n,
                             adversary_train_every_n=adversary_train_every_n, verbose=verbose)
