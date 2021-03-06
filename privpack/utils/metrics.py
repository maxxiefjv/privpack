"""
Statistic classes are used to compute full data-set metrics.
This module supports the following metric classes:

- `PartialBivariateBinaryMutualInformation`
- `PartialMultivariateGaussianMutualInformation`
- `ComputeDistortion`


How to use this module
======================
(See the individual classes, methods and attributes for more details)

1. Import .... TODO

2. Define a instance .... TODO

"""
from typing import Callable, List

import math
import abc
import torch
import numpy as np

class Metric(abc.ABC):
    """
    Base class for creating full data-set Metric.
    """
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def __call__(self, released_data, data):
        pass

class BivariateBinaryMutualInformation(Metric):

    def compute_mutual_information(self, dist: torch.Tensor) -> float:
        P_x = [sum(dist[0, :]), sum(dist[1, :])]
        P_y = [sum(dist[:, 0]), sum(dist[:, 1])]
        MI_binXY = 0

        for y in [0, 1]:
            for x in [0, 1]:

                if P_x[x] == 0 or P_y[y] == 0 or dist[x, y] == 0:
                    continue
                MI_binXY += dist[x, y].item() * math.log2((dist[x, y].item() / (P_x[x] * P_y[y])))

        return MI_binXY

    def estimate_binary_distribution(self, data: torch.Tensor) -> torch.Tensor:
        Px1_y1 = sum([1 for entry in data if entry[0] == 1 and entry[1] == 1]) / len(data)
        Px1_y0 = sum([1 for entry in data if entry[0] == 1 and entry[1] == 0]) / len(data)
        Px0_y1 = sum([1 for entry in data if entry[0] == 0 and entry[1] == 1]) / len(data)
        Px0_y0 = sum([1 for entry in data if entry[0] == 0 and entry[1] == 0]) / len(data)

        return torch.Tensor([
            [Px0_y0, Px0_y1],
            [Px1_y0, Px1_y1]
        ])

    def mi(self, released_data: torch.Tensor, data: torch.Tensor) -> float:
        full_data_tensor = torch.cat((released_data.view(-1, 1), data.view(-1, 1)), dim=1)
        full_data_distribution = self.estimate_binary_distribution(full_data_tensor)
        return self.compute_mutual_information(full_data_distribution)

    def __call__(self, released_data: torch.Tensor, data: torch.Tensor) -> float:
        return self.mi(released_data, data)

class PartialBivariateBinaryMutualInformation(BivariateBinaryMutualInformation):

    def __init__(self, name: str, dimension: int):
        super().__init__(name)
        self.dimension = dimension

    def __call__(self, released_data: torch.Tensor, data: torch.Tensor) -> float:
        return super().__call__(released_data, data[:, self.dimension])

class MultivariateGaussianMutualInformation(Metric):

    def compute_mutual_information(self, full_cov_table, released_data_size) -> float:
        # TODO: What...torch.square? schur_complement should be invertible.
        # full_cov_table = torch.square(torch.Tensor(full_cov_table))
        full_cov_table = torch.Tensor(full_cov_table)
        schur_complement = self._compute_schur_complement(full_cov_table, released_data_size)
        x_cov = full_cov_table[:released_data_size, :released_data_size]

        estimated_mutual_information = .5 * torch.log((torch.det(x_cov) / torch.det(schur_complement)))
        return estimated_mutual_information.item()

    def _prepare_schur_complement(self, cov_table, released_data_size):
        """
        Get elements from cov_table such that:

        | A B |
        | C D |

        """
        A = cov_table[:released_data_size, :released_data_size]
        B = cov_table[:released_data_size, released_data_size:]
        C = cov_table[released_data_size:, :released_data_size]
        D = cov_table[released_data_size:, released_data_size:]

        return (A, B, C, D)

    def _compute_schur_complement(self, cov_table, released_data_size):
        """
        Get elements from cov_table such that:

        | A B |
        | C D |

        Consequently return the schur complement of the A block: D - C * pinv(A) * B
        """
        (A, B, C, D) = self._prepare_schur_complement(cov_table, released_data_size)
        assert torch.equal(C.T, B)
        return (A - torch.matmul(torch.matmul(B,torch.inverse(D)), B.T))

    def _get_positive_definite_covariance(self, numpy_release, numpy_data):
        """
        Get the covariance matrix of the matrix [numpy_release, numpy_data]. Add small uniform noise
        to guarantee a positive definite covariance matrix.
        """
        release_no_cols = numpy_release.shape[1]
        data_no_cols = numpy_data.shape[1]
        # print(numpy_release)
        return np.cov(numpy_data.T, numpy_release.T) + np.diag(np.full((release_no_cols + data_no_cols), 1e-3))

    def mi(self, released_data: torch.Tensor, data: torch.Tensor) -> float:
        """
        Collect the results of the helper functions, and return the mutual information.
        """
        numpy_release = released_data.cpu().numpy()
        numpy_data = data.cpu().numpy()

        full_cov = self._get_positive_definite_covariance(numpy_release, numpy_data)
        return self.compute_mutual_information(full_cov, released_data.size(1))

    def __call__(self, released_data, data):
        return self.mi(released_data, data)

class PartialMultivariateGaussianMutualInformation(MultivariateGaussianMutualInformation):

    def __init__(self, name: str, dimensions: List[int]):
        super().__init__(name)
        self.dimensions = dimensions

    def __call__(self, released_data: torch.Tensor, data: torch.Tensor) -> float:
        return super().__call__(released_data, data[:, self.dimensions])

class ComputeDistortion(Metric):
    def __init__(self, name: str, dimensions: List[int]):
        super().__init__(name)
        self.dimensions = dimensions

    def set_distortion_function(self, distortion_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.distortion = distortion_func
        return self

    def compute_distortion(self, released_data: torch.Tensor, data: torch.Tensor) -> float:
        inp_data = data[:, self.dimensions]
        if not inp_data.size() == released_data.size():
            raise RuntimeError('Tensors must have same size: got {} and {}'.format(released_data.size(), inp_data.size()))

        return self.distortion(released_data, inp_data).mean().item()

    def __call__(self, released_data: torch.Tensor, data: torch.Tensor) -> float:
        return self.compute_distortion(released_data, data)
