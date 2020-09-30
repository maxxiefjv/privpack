import math
import torch

def compute_released_data_statistics(released_data, data, statistics):
    statistics_report = {}
    for statistic in statistics:
        data_stats = statistic(released_data, data)
        statistics_report[statistic.__name__] = data_stats

    return statistics_report

#### Bivariate Binary related Utility functions

def get_likelihood_xi_given_z(adversary_out, Xi):
        res = adversary_out  # P(Xi = 1 | z)
        # If Xi = 0 -> res = 1 - res; else (if Xi = 1) -> res = res
        res = (1 - Xi) * (1 - res) + Xi * res

        # if Xi == 0:
        #     res = 1 - adversary_out ## 1 - P(Xi = 1 | z) = P(Xi = 0 | z)

        return res  # P(x | z)

def compute_binary_released_set(privatizer, data):
    released_data = privatizer(data).detach()
    random_uniform_tensor = torch.rand(released_data.size())
    return torch.ceil(released_data - random_uniform_tensor).to(torch.int).view(-1, 1)

def binary_bivariate_mutual_information_statistic(released_data, data):
    full_data_tensor = torch.cat( (released_data.view(-1, 1), data.view(-1, 1)), dim=1)
    full_data_distribution = estimate_binary_distribution(full_data_tensor)
    return compute_mutual_information_binary(full_data_distribution)

def binary_bivariate_mutual_information_zx(released_data, data):
    return binary_bivariate_mutual_information_statistic(released_data, data[:,0])

def binary_bivariate_mutual_information_zy(released_data, data):
    return binary_bivariate_mutual_information_statistic(released_data, data[:,1])

def binary_bivariate_distortion_zy(released_data, data):
    return hamming_distance(released_data, data[:, 1].view(-1,1)).to(torch.float64).mean().item()

def hamming_distance(actual, expected):
    return (actual != expected).to(torch.int)

def compute_mutual_information_binary(dist):
    P_x = [sum(dist[0, :]), sum(dist[1, :])]
    P_y = [sum(dist[:, 0]), sum(dist[:, 1])]
    MI_binXY = 0

    for y in [0, 1]:
        for x in [0, 1]:

            if P_x[x] == 0 or P_y[y] == 0 or dist[x, y] == 0:
                continue
            MI_binXY += dist[x, y] * math.log2((dist[x, y] / (P_x[x] * P_y[y])))

    return MI_binXY

def estimate_binary_distribution(data):
    Px1_y1 = sum([1 for entry in data if entry[0] ==
                  1 and entry[1] == 1]) / len(data)
    Px1_y0 = sum([1 for entry in data if entry[0] ==
                  1 and entry[1] == 0]) / len(data)
    Px0_y1 = sum([1 for entry in data if entry[0] ==
                  0 and entry[1] == 1]) / len(data)
    Px0_y0 = sum([1 for entry in data if entry[0] ==
                  0 and entry[1] == 0]) / len(data)

    return torch.Tensor([
        [Px0_y0, Px0_y1],
        [Px1_y0, Px1_y1]
    ])

#### (Multivariate) Gaussian related Utility functions
def compute_mutual_information_gaussian(full_cov_table, x_size):
    full_cov_table = torch.Tensor(full_cov_table)
    schur_complement = _compute_schur_complement(full_cov_table, x_size)
    x_cov = full_cov_table[:x_size, :x_size]
    if (torch.det(x_cov) == 0):
        print('determinent x_cov is zero.')
    if (torch.det(schur_complement) == 0):
        print('determinent schur_complement is zero.')

    return .5 * torch.log((torch.det(x_cov) / torch.det(schur_complement)))

def _compute_schur_complement(cov_table, x_size):
    A = cov_table[:x_size, :x_size]
    B = cov_table[:x_size, x_size:]
    C = cov_table[x_size:, :x_size]
    D = cov_table[x_size:, x_size:]

    return A - B * torch.pinverse(D) * C
