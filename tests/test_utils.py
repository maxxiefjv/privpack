import torch

from privpack import compute_released_data_statistics, binary_bivariate_mutual_information_statistic
from privpack import hamming_distance
from privpack import compute_binary_released_set

def test_compute_binary_released_set():
    mock_uniform_rdata = torch.Tensor([
        0.4,
        0.7,
        0.3,
        0.5,
        0.5,
        0,
        1
    ])

    released_data = torch.Tensor([
        0.3,
        0.6,
        0.7,
        0,
        1,
        0.5,
        0.5
    ])

    actual = torch.ceil(released_data - mock_uniform_rdata).to(torch.int)
    expected = torch.Tensor([
        0,
        0,
        1,
        0,
        1,
        1,
        0
    ]).to(torch.int)
    assert torch.all(actual == expected)


def test_compute_released_data_statistics():
    def mutual_information_zx(released_data, data):
        return binary_bivariate_mutual_information_statistic(released_data, data[:, 0]).item()

    def mutual_information_zy(released_data, data):
        return binary_bivariate_mutual_information_statistic(released_data, data[:, 1]).item()

    def distortion_zy(released_data, data):
        return hamming_distance(released_data, data[:, 1]).to(torch.float64).mean().item()

    mock_released_data = torch.Tensor([
        0,
        0,
        1,
        1
    ])

    mock_data = torch.Tensor([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    expected = {
        'mutual_information_zx': 0.0,
        'mutual_information_zy': 1.0,
        'distortion_zy': 0.0
    }

    actual = compute_released_data_statistics(mock_released_data, mock_data, [
        mutual_information_zx,
        mutual_information_zy,
        distortion_zy
    ])

    assert expected == actual
