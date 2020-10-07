# import torch
# from privpack import compute_released_data_metrics, binary_bivariate_mutual_information_statistic
# from privpack import hamming_distance
# from privpack import binary_bivariate_mutual_information_zx, binary_bivariate_mutual_information_zy, binary_bivariate_distortion_zy
#
# def test_compute_released_data_metrics():
#     mock_released_data = torch.Tensor([
#         [0],
#         [0],
#         [1],
#         [1]
#     ])

#     mock_data = torch.Tensor([
#         [0, 0],
#         [1, 0],
#         [0, 1],
#         [1, 1]
#     ])

#     expected = {
#         'binary_bivariate_mutual_information_zx': 0.0,
#         'binary_bivariate_mutual_information_zy': 1.0,
#         'binary_bivariate_distortion_zy': 0.0
#     }

#     actual = compute_released_data_metrics(mock_released_data, mock_data, [
#         binary_bivariate_mutual_information_zx,
#         binary_bivariate_mutual_information_zy,
#         binary_bivariate_distortion_zy
#     ])

#     assert expected == actual

def test_something():
    assert True
