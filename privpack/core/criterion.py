"""
The loss-function defined in this module are used to learn the optimal
direction for a privatizer network.

This module defines the following classes:

- `PrivacyLoss`
- `UtilityLoss`

"""

import torch
import abc

from privpack.utils import hamming_distance, elementwise_mse

__all__ = [
    'PrivacyCriterion',
    'DiscreteMutualInformation',
    'BinaryMutualInformation',
    'NegativeBinaryMutualInformation',
    'GaussianMutualInformation',
    'UtilityCriterion',
    'BinaryHammingDistance',
    'MeanSquaredError',
    'PGANCriterion'
]

class Criterion(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractclassmethod
    def __call__(self, actual, expected):
        pass
    
    @abc.abstractclassmethod
    def __str__(self):
        pass

    def _expected_loss(self, release_probabilities, losses):
        """
        Parameters:

        - `release_probabilities`: probability of obtaining corresponding loss
        - `losses`: losses computed using one of the subclasses' functions

        return the expected loss for each entry.
        """
        return torch.mul(release_probabilities, losses).sum(dim=1)

class PGANCriterion():
    """
    Create a criterion class 
    """

    def __init__(self):
        self.privacy_criterions = []
        self.adversary_criterions = []

    def __str__(self):
        str_repr_privacy_crits = []
        str_repr_adversary_crits = []

        for criterion in self.privacy_criterions:
            str_repr_privacy_crits += str(criterion) + '\n'

        for criterion in self.adversary_criterions:
            str_repr_adversary_crits += '\t' + str(criterion)

        return f"Privacy Criterion: \n {str_repr_privacy_crits} \n Adversary Criterion: \n{str_repr_adversary_crits} "

    def to_json_dict(self):
        return {
            'PrivacyCriterion' : [str(x) for x in self.privacy_criterions],
            'AdversaryCriterion' : [str(x) for x in self.adversary_criterions],
        }

    def add_privacy_criterion(self, criterion: Criterion):
        self.privacy_criterions.append(criterion)
        return self

    def add_adversary_criterion(self, criterion: Criterion):
        self.adversary_criterions.append(criterion)
        return self

    def _compute_loss(self, releases, actual_private_values, actual_public_values, criterions):
        total_loss = 0

        for criterion in criterions:
            if issubclass(type(criterion), PrivacyCriterion):
                total_loss += criterion(releases, actual_private_values)
            elif issubclass(type(criterion), UtilityCriterion):
                total_loss += criterion(releases, actual_public_values)
            else:
                raise NotImplementedError("Unhandled Criterion type.")

        return total_loss

    def privacy_loss(self, releases, actual_private_values, actual_public_values):
        return self._compute_loss(releases, actual_private_values, actual_public_values, self.privacy_criterions)

    def adversary_loss(self, releases, actual_private_values, actual_public_values):
        return self._compute_loss(releases, actual_private_values, actual_public_values, self.adversary_criterions)

class PrivacyCriterion(Criterion):
    """
    Privacy Criterion is a component including loss functions correlated to Information Theoretic Losses.

    Using these loss function optimum can be achieved for:

    - Mutual Information
    - Maximal Leakage
    - Alpha-Tunable Information Leakage
    """

    def __init__(self):
        pass

    def __str__(self):
        return str(type(self).__name__)

    @abc.abstractclassmethod
    def __call__(self, releases, likelihood_x) -> torch.Tensor:
        pass

class DiscreteMutualInformation(PrivacyCriterion):

    def __call__(self, releases, likelihood_x):
        return self.discrete_mi_loss(releases, likelihood_x)

    def discrete_mi_loss(self, release_all_probabilities, likelihood_x):
        """
        Compute loss variant of the mutual information between X and released Z
        provided the probability per possible release, and that release's their
        related computed likelihoods of x. This is similar to the log-loss

        - `probability_releases`: tensor of probabilities per release option: z=0 and z=1.
        - `likelihood_x`: computed likelihood of x given z.

        return the mutual information for discrete values.
        """
        return super()._expected_loss(release_all_probabilities, torch.log2(likelihood_x))

class BinaryMutualInformation(DiscreteMutualInformation):

    def __call__(self, releases, likelihood_x):
        return self.binary_mi_loss(releases, likelihood_x)

    def binary_mi_loss(self, release_probabilities, likelihood_x):
        """
        Function limited to computing the log-loss for binary cases, using discrete_mutual_information_loss.
        """
        release_all_probabilities = torch.cat((1 - release_probabilities, release_probabilities), dim=1)
        return super().discrete_mi_loss(release_all_probabilities, likelihood_x)

class NegativeBinaryMutualInformation(BinaryMutualInformation):
        def __call__(self, releases, actual_private_values):
            return -1 * super().__call__(releases, actual_private_values)

class GaussianMutualInformation(PrivacyCriterion):

    def __call__(self, releases, likelihood_x):
        return self.gaussian_mutual_information_loss(releases, likelihood_x)

    def gaussian_mutual_information_loss(self, releases, log_likelihoods):
        k = releases.size(0)
        probabilities = torch.Tensor([1 / k]).repeat(log_likelihoods.size(0)).view(log_likelihoods.size())
        return super()._expected_loss(probabilities, log_likelihoods)

class NegativeGaussianMutualInformation(GaussianMutualInformation):
        def __call__(self, releases, actual_private_values):
            return -1 * super().__call__(releases, actual_private_values)

class UtilityCriterion(Criterion):
    """
    Utitlity loss concerns itself with loss function related to utility/distortion (or disutility). Each
    utility loss is computed according to the formula:

    :math:`\\lambda max(0, E[d(X,Y)] - \\delta)^2`

    At the moment this class includes:

    - Hamming distance
    """

    def __init__(self, lambd, delta_constraint):
        self.lambd = lambd
        self.delta_constraint = delta_constraint

    def __str__(self):
        return type(self).__name__ + '(lambda={}, delta={})'.format(self.lambd, self.delta_constraint)

    @abc.abstractclassmethod
    def __call__(self, releases, likelihood_x):
        pass

    def _expected_loss(self, release_all_probabilities, losses):
        expected_distortion = super()._expected_loss(release_all_probabilities, losses)
        return self.lambd * torch.max(torch.zeros_like(expected_distortion),
                                      (expected_distortion - self.delta_constraint)) ** 2

class BinaryHammingDistance(UtilityCriterion):

    def __call__(self, releases, likelihood_x):
        return self.expected_binary_hamming_distance(releases, likelihood_x)
    
    def expected_binary_hamming_distance(self, release_probabilities, expected):
        """
        Compute the hamming distance for both possible binary values.

        Parameters:

        - `probability_releases`: tensor of probabilities per release option: z=0 and z=1.
        - `expected`: expected/most utility release values.
        """
        release_all_probabilities = torch.cat((1 - release_probabilities, release_probabilities), dim=1)

        hamming_distance_zeros = hamming_distance(torch.zeros_like(expected), expected).view(-1, 1)
        hamming_distance_ones = hamming_distance(torch.ones_like(expected), expected).view(-1, 1)

        hamming_distances = torch.cat((hamming_distance_zeros, hamming_distance_ones), dim=1)

        return super()._expected_loss(release_all_probabilities, hamming_distances)

class MeanSquaredError(UtilityCriterion):
    
    def __call__(self, releases, likelihood_x):
        return self.expected_mean_squared_error(releases, likelihood_x)

    def expected_mean_squared_error(self, releases, expected):
        """
        Compute the mean squared error (MSE) between the releases and most utility values.

        Parameters:

        - `releases`: tensor of all releases done by some privatizer network.
        - `expected`: expected/most utility release values.
        """
        if len(releases.size()) < 3:
            raise RuntimeError("Tensor must have at least three dimensions: [releases, no_samples, no_features], but has shape: {}".format(releases.size()))

        summed_mse = torch.zeros(releases.size(1))
        for release in releases:
            mse = elementwise_mse(release, expected)
            summed_mse += mse.view(summed_mse.size())

        summed_mse = summed_mse.unsqueeze(1)
        k = releases.size(0)
        probabilities = torch.Tensor([1 / k]).repeat(summed_mse.size(0)).view(summed_mse.size())

        return super()._expected_loss(probabilities, summed_mse)
