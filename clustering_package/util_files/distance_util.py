import torch
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from torchmetrics.functional import pairwise_cosine_similarity


def dist_sum_of_squares(x, y):
    return torch.cdist(x, y) ** 2


def dist_euclidean(x, y):
    return torch.cdist(x, y)


def dist_cosine(x, y):
    return 1 - pairwise_cosine_similarity(x, y)


def dist_sum_of_squares_np(x, y):
    return euclidean_distances(x, y) ** 2


def dist_euclidean_np(x, y):
    return euclidean_distances(x, y)


def dist_cosine_np(x, y):
    return cosine_distances(x, y)


def dist_dim_sum_of_squares(x, y, y_mask=None):
    """

    :param x: [n * p]
    :param y: [k * p]
    :param y_mask: [k * p] in {0,1}
    :return:
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    if y_mask is None:
        y_mask = torch.ones(y.shape)
    if not torch.is_tensor(y_mask):
        y_mask = torch.tensor(y_mask, dtype=torch.float32)
    x_expand = x.unsqueeze(1).expand(-1, y.shape[0], -1)
    y_expand = y.unsqueeze(0).expand(x.shape[0], -1, -1)
    mask_expand = y_mask.unsqueeze(0).expand(x.shape[0], -1, -1)
    dist_dim = (x_expand - y_expand) ** 2
    dist_mask = torch.sum(dist_dim * mask_expand, 2)
    return dist_mask, dist_dim


def dist_dim_euclidean(x, y, y_mask=None):
    """

    :param x: [n * p]
    :param y: [k * p]
    :param y_mask: [k * p] in {0,1}
    :return:
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    if y_mask is None:
        y_mask = torch.ones(y.shape)
    if not torch.is_tensor(y_mask):
        y_mask = torch.tensor(y_mask, dtype=torch.float32)
    x_expand = x.unsqueeze(1).expand(-1, y.shape[0], -1)
    y_expand = y.unsqueeze(0).expand(x.shape[0], -1, -1)
    mask_expand = y_mask.unsqueeze(0).expand(x.shape[0], -1, -1)
    dist_dim = (x_expand - y_expand) ** 2
    dist_mask = torch.sum(dist_dim * mask_expand, 2) ** 0.5
    return dist_mask, dist_dim


def loss_sum_of_squares(x, y):
    return torch.mean(torch.sum(torch.square(x - y), 1))


def set_dist_function(dist_name, lib='torch'):
    DIST_LIST = {
        'euclidean': dist_euclidean,
        'sum_of_squares': dist_sum_of_squares,
        'cosine': dist_cosine,
        'euclidean_dim': dist_dim_euclidean,
        'sum_of_squares_dim': dist_dim_sum_of_squares,
        'euclidean_np': dist_euclidean_np,
        'sum_of_squares_np': dist_sum_of_squares_np,
        'cosine_np': dist_cosine_np,
        'loss_sum_of_squares': loss_sum_of_squares
    }
    if lib == 'np':
        dist_name += '_np'
    if dist_name not in DIST_LIST.keys():
        print('Unknown name of the distance.')
        return None
    return DIST_LIST[dist_name]


if __name__ == '__main__':
    print('ok')
