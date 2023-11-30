import torch
import clustering_package.util_files.distance_util as distance_util


def compute_probability(d_closest, m, tau=1.0, mu=2.00):
    """Compute the probability that documents/representatives is associated
    to the English, English-French or French

    A(v, 0) = exp(tau *  (1/mu - de/df))
    A(v, 1) = exp(tau *  min(de/df - 1/mu, df/de - 1/mu))
    A(v, 2) = exp(tau *  (1/mu - df/de))

    P(v,i) = A(v,i)/ (sum_{i=0}^2 A(v,i)), 0 ≤ i ≤ 2

    :param tau:
    :param d_closest:
    :param m:
    :param mu:
    :return:
    """
    de, df = d_closest
    A = torch.stack((torch.exp(tau * (1 / mu - de / df)),
                     torch.exp(tau * torch.min(de / df - 1 / mu, df / de - 1 / mu)),
                     torch.exp(tau * (1 / mu - df / de))), -1) * m
    sum_A = torch.sum(A, 1)
    P = A / sum_A.unsqueeze(1)
    return P


def compute_closest_dist(v, space, space_mask, dist_fun=distance_util.dist_sum_of_squares, l_nearest_1=1, l_nearest_2=1):
    dist = dist_fun(v, space)
    dist_1 = dist * torch.squeeze(space_mask)
    dist_1[dist_1 == 0.0] = 1e+32
    d1 = torch.kthvalue(dist_1, l_nearest_1, 1, True)[0].squeeze(1)

    dist_2 = dist * torch.squeeze(1 - space_mask)
    dist_2[dist_2 == 0.0] = 1e+32
    d2 = torch.kthvalue(dist_2, l_nearest_2, 1, True)[0].squeeze(1)

    return d1, d2


def compute_closest_dist_with_mask(v, space, space_mask, v_mask, dist_fun=distance_util.dist_dim_sum_of_squares,
                                   l_nearest_1=1, l_nearest_2=1):
    dist, _ = dist_fun(space, v, v_mask)
    dist_1 = dist.T * torch.squeeze(space_mask)
    dist_1[dist_1 == 0.0] = 1e+32
    d1 = torch.kthvalue(dist_1, l_nearest_1, 1, True)[0].squeeze(1)

    dist_2 = dist.T * torch.squeeze(1 - space_mask)
    dist_2[dist_2 == 0.0] = 1e+32
    d2 = torch.kthvalue(dist_2, l_nearest_2, 1, True)[0].squeeze(1)

    return d1, d2
