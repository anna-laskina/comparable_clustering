import math
import torch


def elimination_update(dist_dim, eta, alpha, rep_mask):
    n_clusters = dist_dim.shape[0]
    features_dim = dist_dim.shape[2]
    number_of_dimensions = math.ceil(features_dim * eta)

    dist = torch.sum(dist_dim * rep_mask.unsqueeze(1).expand(-1, dist_dim.shape[1], -1), 2).T
    exp = torch.exp(- alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, n_clusters)))
    softmax = exp / torch.sum(exp, 1).unsqueeze(1)

    ind0 = torch.arange(0, n_clusters, dtype=torch.int64)
    ind1 = torch.max(torch.sum(
        dist_dim *
        rep_mask.unsqueeze(1).expand(-1, dist_dim.shape[1], -1) *
        softmax.T.unsqueeze(2).expand(-1, -1, dist_dim.shape[2]),
        1), 1)[1]
    rep_mask[ind0, ind1] = 0

    if_continue = False if sum(rep_mask[0]) < number_of_dimensions else True
    return rep_mask, if_continue


def sampling_update(dist_dim, eta, alpha, rep_mask=None):
    """

    :param dist_dim:  [k,n,p]
    :param eta:
    :param alpha:
    :param rep_mask:
    :return:
    """
    n_clusters = dist_dim.shape[0]
    features_dim = dist_dim.shape[2]
    number_of_dimensions = math.ceil(features_dim * eta)
    # dist = torch.sum(dist_dim * rep_mask.unsqueeze(1).expand(-1, dist_dim.shape[1], -1), 2).T
    dist = torch.sum(dist_dim, 2).T
    exp = torch.exp(- alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, n_clusters)))
    softmax = exp / torch.sum(exp, 1).unsqueeze(1)
    ind0 = torch.arange(0, n_clusters, dtype=torch.int64).unsqueeze(1).expand(-1, number_of_dimensions)
    ind1 = torch.sort(torch.sum(
        dist_dim * softmax.T.unsqueeze(2).expand(-1, -1, dist_dim.shape[2]), 1), 1)[1][:, :number_of_dimensions]
    rep_mask = torch.zeros(n_clusters, features_dim)
    rep_mask[ind0, ind1] = 1
    return rep_mask, True


def sampling_mask_update(dist_dim, eta, alpha, rep_mask=None):
    """

    :param dist_dim:  [k,n,p]
    :param eta:
    :param alpha:
    :param rep_mask:
    :return:
    """
    n_clusters = dist_dim.shape[0]
    features_dim = dist_dim.shape[2]
    number_of_dimensions = math.ceil(features_dim * eta)
    dist = torch.sum(dist_dim * rep_mask.unsqueeze(1).expand(-1, dist_dim.shape[1], -1), 2).T
    # dist = torch.sum(dist_dim, 2).T
    exp = torch.exp(- alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, n_clusters)))
    softmax = exp / torch.sum(exp, 1).unsqueeze(1)
    ind0 = torch.arange(0, n_clusters, dtype=torch.int64).unsqueeze(1).expand(-1, number_of_dimensions)
    ind1 = torch.sort(torch.sum(
        dist_dim * softmax.T.unsqueeze(2).expand(-1, -1, dist_dim.shape[2]), 1), 1)[1][:, :number_of_dimensions]
    rep_mask = torch.zeros(n_clusters, features_dim)
    rep_mask[ind0, ind1] = 1
    return rep_mask, True


def pick_update(dist_dim, eta, alpha, rep_mask=None):
    n_clusters = dist_dim.shape[0]
    features_dim = dist_dim.shape[2]
    number_of_dimensions = math.ceil(features_dim * eta)

    ind0 = torch.arange(0, n_clusters, dtype=torch.int64).unsqueeze(1).expand(-1, number_of_dimensions)
    ind1 = torch.sort(torch.sum(dist_dim, 1), 1)[1][:, : number_of_dimensions]
    rep_mask = torch.zeros(n_clusters, features_dim)
    rep_mask[ind0, ind1] = 1
    return rep_mask, True


def pick_exp_update(dist_dim, eta, alpha, rep_mask=None):
    n_clusters = dist_dim.shape[0]
    features_dim = dist_dim.shape[2]
    number_of_dimensions = math.ceil(features_dim * eta)
    dist = torch.sum(dist_dim * rep_mask.unsqueeze(1).expand(-1, dist_dim.shape[1], -1), 2).T
    exp = torch.exp(-alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, n_clusters)))
    softmax = exp / torch.sum(exp, 1).unsqueeze(1)
    ind0 = torch.arange(0, n_clusters, dtype=torch.int64).unsqueeze(1).expand(-1, number_of_dimensions)
    ind1 = torch.sort(torch.sum(dist_dim * softmax.T.unsqueeze(2).expand(-1, -1, dist_dim.shape[2]), 1), 1)[1][:, : number_of_dimensions]
    rep_mask = torch.zeros(n_clusters, features_dim)
    rep_mask[ind0, ind1] = 1
    return rep_mask, True


def reduce_update(dist_dim, eta, alpha, rep_mask=None):
    n_clusters = dist_dim.shape[0]
    features_dim = dist_dim.shape[2]
    number_of_dimensions = math.ceil(features_dim * eta)

    ind0 = torch.arange(0, n_clusters, dtype=torch.int64)
    ind1 = torch.max(torch.sum(dist_dim * rep_mask.unsqueeze(1).expand(-1, dist_dim.shape[1], -1), 1), 1)[1]
    rep_mask[ind0, ind1] = 0

    if_continue = False if sum(rep_mask[0]) < number_of_dimensions else True
    return rep_mask, if_continue


def no_update(dist_dim, eta, alpha, rep_mask=None):
    return rep_mask, True


def set_mask_update(update_type):
    update_dict = {
        'elimination': elimination_update,
        'sampling': sampling_update,
        'sampling_mask': sampling_mask_update,
        'pick': pick_update,
        'pick_exp': pick_exp_update,
        'reduce': reduce_update,
        'non': no_update,
    }
    if update_type not in update_dict.keys():
        print(f'Unknown type of mask update!')
    return update_dict.get(update_type, None)


if __name__ == '__main__':
    print('ok')
