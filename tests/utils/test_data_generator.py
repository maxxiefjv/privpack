
import numpy as np
import torch

from privpack.utils import DataGenerator

def test_ppan_gauss_param_scalar_experiment():
    x_dim = y_dim = 1
    (mu, cov) = DataGenerator.get_ppan_distribution_params(x_dim, y_dim)
    
    assert torch.all(mu == torch.Tensor([0, 0]))
    assert torch.all(cov == torch.Tensor([
        [1, 0.85],
        [0.85, 1]
    ]))

def test_ratios_binary_data():

    for i in range(100):
        (norm_dist, acc_dist) = DataGenerator.get_completely_uncorrelated_dist()
        synthetic_data = DataGenerator.generate_binary_data(10000, acc_dist)

        train_data = torch.Tensor(synthetic_data[:8000])
        test_data = torch.Tensor(synthetic_data[8000:])
        
        (unique_train, unique_train_count) = np.unique(train_data, axis=0, return_counts=True)
        (unique_test, unique_test_count) = np.unique(test_data, axis=0, return_counts=True)
        unique_total_count = unique_train_count + unique_test_count
        
        train_sample_ratios = unique_train_count / unique_total_count
        test_sample_ratios = unique_test_count / unique_total_count

        assert np.all(np.isclose(train_sample_ratios, 0.8, atol=5e-2))
        assert np.all(np.isclose(test_sample_ratios, 0.2, atol=5e-2))

def test_uniqueness_gaussian_data():

    for i in range(100):
        (mu, cov) = DataGenerator.get_completely_uncorrelated_distribution_params(5,5)
        (train_data, test_data) = DataGenerator.generate_gauss_mixture_data(mu, cov)

        (unique_train, unique_train_count) = np.unique(train_data, axis=0, return_counts=True)
        (unique_test, unique_test_count) = np.unique(test_data, axis=0, return_counts=True)
        
        assert not (np.any(unique_train_count > 1))
        assert not (np.any(unique_test_count > 1))