"""
Cross Test File Fixture sharing
"""
from privpack.utils import DataGenerator

import numpy as np
import pytest
import torch

@pytest.fixture(scope="module")
def mock_release_probabilities():
    return torch.Tensor([
        [0.6],
        [0.3],
        [0.1]
    ])


@pytest.fixture
def fixed_train_data():
    """
    The generated data consists of 10 columns, generated using the DataGenerator.get_ppan_distribution_params function.
    """
    return np.loadtxt('tests/data/static_train_data.csv', delimiter=',')


@pytest.fixture
def fixed_test_data():
    """
    The generated data consists of 10 columns, generated using the DataGenerator.get_ppan_distribution_params function, with seed=0.
    """
    return np.loadtxt('tests/data/static_test_data.csv', delimiter=',')

@pytest.fixture
def get_random_train_data():
    def _generate_random_train_data():
        (mu, cov) = DataGenerator.get_ppan_distribution_params(5, 5)
        return DataGenerator.generate_gauss_mixture_data(mu, cov)[0]

    return _generate_random_train_data
