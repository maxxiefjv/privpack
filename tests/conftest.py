"""
Cross Test File Fixture sharing
"""
import pytest
import torch

@pytest.fixture
def mock_release_probabilities():
    return torch.Tensor([
        [0.6],
        [0.3],
        [0.1]
    ])