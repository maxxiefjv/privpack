import torch

from privpack import hamming_distance

class Loss():
    def __init__(self):
        pass

class PrivacyLoss(Loss):

    def __init__(self):
        pass

    def discrete_mi_loss(self, release_all_probabilities, likelihood_x):
        """
        Compute loss variant of the mutual information between X and released Z 
        provided the probability per possible release, and that release's their 
        related computed likelihoods of x. This is similar to the log-loss

        - `probability_releases`: tensor of probabilities per release option: z=0 and z=1.
        - `likelihood_x`: computed likelihood of x given z.

        return the mutual information for discrete values.
        """
        return torch.mul(release_all_probabilities, torch.log2(likelihood_x)).sum(dim=1)

    def binary_mi_loss(self, release_probabilities, likelihood_x):
        """
        Function limited to computing the log-loss for binary cases, using discrete_mutual_information_loss.
        """
        release_all_probabilities = torch.cat( (1 - release_probabilities, release_probabilities), dim=1)
        return self.discrete_mi_loss(release_all_probabilities, likelihood_x)

    def gaussian_mutual_information(self):
        pass


class UtilityLoss(Loss):

    def __init__(self, lambd, delta_constraint):
        self.lambd = lambd
        self.delta_constraint = delta_constraint

    def _expected_loss(self, release_all_probabilities, losses):
        expected_distortion = torch.mul(losses , release_all_probabilities).sum(dim=1)
        # print(expected_distortion)
        return self.lambd * torch.max(torch.zeros_like(expected_distortion),
                                      (expected_distortion - self.delta_constraint)) ** 2

    def expected_binary_hamming_distance(self, release_probabilities, expected):
        release_all_probabilities = torch.cat( (1 - release_probabilities, release_probabilities), dim=1)

        hamming_distance_zeros = hamming_distance(torch.zeros_like(expected), expected).view(-1, 1)
        hamming_distance_ones = hamming_distance(torch.ones_like(expected), expected).view(-1, 1)

        hamming_distances = torch.cat( (hamming_distance_zeros, hamming_distance_ones), dim=1)

        return self._expected_loss(release_all_probabilities, hamming_distances)

        