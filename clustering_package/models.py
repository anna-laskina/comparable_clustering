import torch
import numpy as np
import torch.nn as nn

from clustering_package.util_files import utils, distance_util, mask_util, comparable_util


class KmeansBase(nn.Module):
    def __init__(self, n_clusters=5, random_state=None, dist_metric='euclidean', features_dim=None,
                 strategy='random', centroid_info=None):
        super().__init__()
        self.seed = random_state
        if self.seed is not None:
            utils.fix_seed(self.seed)
        self.n_clusters = n_clusters
        self.compute_dist = distance_util.set_dist_function(dist_name=dist_metric, lib='torch')
        self.compute_dist_np = distance_util.set_dist_function(dist_name=dist_metric, lib='np')
        if features_dim is None and centroid_info is not None:
            features_dim = centroid_info.shape[0]
        self.features_dim = features_dim
        self.cluster_rep = nn.Parameter(
            self.centroids_initialize(strategy=strategy, info=centroid_info))

    def centroids_initialize(self, strategy='random', info=None):
        if strategy == 'init':
            centroids = info
        elif strategy == 'rand++':
            centroids = [info[np.random.randint(info.shape[0]), :]]
            centroid_dist = []
            for _ in range(self.n_clusters - 1):
                centroid_dist.append(self.compute_dist_np(info, [centroids[-1]])[:, 0])
                centroids.append(info[np.argmax(np.min(centroid_dist, 0)), :])
        elif strategy == 'k-means++':
            from sklearn.cluster import KMeans
            _model = KMeans(n_clusters=self.n_clusters, init='k-means++')
            _model.fit(info)
            centroids = _model.cluster_centers_
        elif strategy == 'random':
            [high, low] = [1, -1] if info is None else [np.max(info), np.min(info)]
            centroids = (high - low) * torch.rand((self.n_clusters, self.features_dim)) + low
        else:
            print('Unknown initialize strategy for centroids. Centroids will be chosen randomly.')
            centroids = self.centroids_initialize(strategy='random', info=info)
        if not torch.is_tensor(centroids):
            centroids = torch.tensor(np.array(centroids), dtype=torch.float32)
        return centroids


class BatchKmeans(KmeansBase):
    def __init__(self, n_clusters=5, random_state=None, dist_metric='euclidean', features_dim=None, strategy='random',
                 centroid_info=None):
        super().__init__(n_clusters=n_clusters, random_state=random_state, dist_metric=dist_metric,
                         features_dim=features_dim, strategy=strategy, centroid_info=centroid_info)

    def compute_loss(self, data):
        dist = self.compute_dist(data, self.cluster_rep)
        assignments = dist.min(1, keepdim=True).indices
        loss_kmeans = dist.gather(1, assignments).squeeze(1).mean()
        # loss_kmeans = dist.gather(1, assignments).squeeze(1).sum()
        return loss_kmeans, dist


class AlphaKmeans(KmeansBase):
    def __init__(self, n_clusters=5, random_state=None, dist_metric='euclidean', features_dim=None, strategy='random',
                 centroid_info=None):
        super().__init__(n_clusters=n_clusters, random_state=random_state, dist_metric=dist_metric,
                         features_dim=features_dim, strategy=strategy, centroid_info=centroid_info)

    def compute_loss(self, data, alpha):
        dist = self.compute_dist(data, self.cluster_rep)
        exp = torch.exp(-alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, self.n_clusters)))
        softmax = exp / torch.sum(exp, 1).unsqueeze(1)
        loss_alpha_kmeans = torch.mean(torch.sum(softmax * dist, 1))
        return loss_alpha_kmeans, dist


class MaskAlphaKmeans(KmeansBase):
    def __init__(self, n_clusters=5, random_state=None, dist_metric='euclidean', features_dim=None, strategy='random',
                 centroid_info=None, eta=0.9, mask_update_type='pick_exp'):
        super().__init__(n_clusters=n_clusters, random_state=random_state, dist_metric=dist_metric,
                         features_dim=features_dim, strategy=strategy, centroid_info=centroid_info)

        self.eta = eta
        self.rep_mask = torch.ones(self.cluster_rep.shape)

        if '+' in mask_update_type:
            update_type = mask_update_type[: mask_update_type.rfind('+')]
            switch_type = mask_update_type[mask_update_type.rfind('+') + 1:]
        else:
            update_type, switch_type = mask_update_type, 'sampling'

        self.update_mask_fun = mask_util.set_mask_update(update_type=update_type)
        self.switch_mask_fun = mask_util.set_mask_update(update_type=switch_type)
        self.compute_dim_dist = distance_util.set_dist_function(dist_name=dist_metric + '_dim', lib='torch')

    def compute_loss(self, data, alpha):
        dist, dist_dim = self.compute_dim_dist(data, self.cluster_rep, self.rep_mask)
        exp = torch.exp(-alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, self.n_clusters)))
        softmax = exp / torch.sum(exp, 1).unsqueeze(1)
        loss_alpha_kmeans = torch.mean(torch.sum(softmax * dist, 1))
        return loss_alpha_kmeans, dist, dist_dim

    def update_rep_mask(self, data, alpha=1):
        _, dist_dim = self.compute_dim_dist(data, self.cluster_rep, self.rep_mask)
        rep_mask, if_continue = self.update_mask_fun(dist_dim=dist_dim, eta=self.eta, alpha=alpha,
                                                     rep_mask=self.rep_mask)
        self.rep_mask = rep_mask
        if not if_continue:
            self.update_mask_fun = self.switch_mask_fun


class ComparableAlphaKmeans(KmeansBase):
    def __init__(self, n_clusters=5, random_state=None, dist_metric='euclidean', features_dim=None, strategy='random',
                 centroid_info=None, mu=2.0):
        super().__init__(n_clusters=n_clusters, random_state=random_state, dist_metric=dist_metric,
                         features_dim=features_dim, strategy=strategy, centroid_info=centroid_info)

        self.mu = mu

    def update_rep_prob(self, space, space_mask, tau):
        return comparable_util.compute_probability(
            d_closest=comparable_util.compute_closest_dist(v=self.cluster_rep.data, space=space, space_mask=space_mask,
                                                           dist_fun=self.compute_dist),
            m=torch.ones((self.n_clusters, 3)), tau=tau, mu=self.mu)

    def update_data_prob(self, space, space_mask, tau):
        return comparable_util.compute_probability(
            d_closest=comparable_util.compute_closest_dist(v=space, space=space, space_mask=space_mask,
                                                           dist_fun=self.compute_dist),
            m=torch.squeeze(torch.stack((space_mask, torch.ones((len(space_mask), 1)), 1 - space_mask), - 1)), tau=tau,
            mu=self.mu)

    def compute_loss(self, data, rep_prob, data_prob, alpha):
        dist = self.compute_dist(data, self.cluster_rep)
        exp = torch.exp(-alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, self.n_clusters)))
        exp_compar = exp.unsqueeze(2).expand(-1, -1, 3) * rep_prob
        sun_exp_compar = torch.sum(exp_compar, 1)
        sun_exp_compar = sun_exp_compar + torch.ones(sun_exp_compar.size()) * 1e-30
        exp_compar_norm = (exp_compar / sun_exp_compar).unsqueeze(1).expand(-1, self.n_clusters, -1)
        f_x = torch.sum(dist.unsqueeze(2).expand(-1, -1, 3) * exp_compar_norm, 1)
        loss_com_kmeans = torch.mean(torch.sum(data_prob * f_x, 1))
        return loss_com_kmeans, dist


class MaskComparableAlphaKmeans(KmeansBase):
    def __init__(self, n_clusters=5, random_state=None, dist_metric='euclidean', features_dim=None,
                 strategy='random', centroid_info=None, mu=2.0, eta=0.9, mask_update_type='pick_exp'):
        super().__init__(n_clusters=n_clusters, random_state=random_state, dist_metric=dist_metric,
                         features_dim=features_dim, strategy=strategy, centroid_info=centroid_info)

        self.eta = eta
        self.mu = mu

        if '+' in mask_update_type:
            update_type = mask_update_type[: mask_update_type.rfind('+')]
            switch_type = mask_update_type[mask_update_type.rfind('+') + 1:]
        else:
            update_type, switch_type = mask_update_type, 'sampling'

        self.update_mask_fun = mask_util.set_mask_update(update_type=update_type)
        self.switch_mask_fun = mask_util.set_mask_update(update_type=switch_type)
        self.compute_dim_dist = distance_util.set_dist_function(dist_name=dist_metric + '_dim', lib='torch')

        self.rep_mask = torch.ones(self.cluster_rep.shape)

    def update_rep_prob(self, space, space_mask, tau):
        return comparable_util.compute_probability(
            d_closest=comparable_util.compute_closest_dist_with_mask(v=self.cluster_rep.data, space=space,
                                                                     space_mask=space_mask, v_mask=self.rep_mask,
                                                                     dist_fun=self.compute_dim_dist),
            m=torch.ones((self.n_clusters, 3)), tau=tau, mu=self.mu)

    def update_data_prob(self, space, space_mask, tau):
        return comparable_util.compute_probability(
            d_closest=comparable_util.compute_closest_dist(v=space, space=space, space_mask=space_mask,
                                                           dist_fun=self.compute_dist),
            m=torch.squeeze(torch.stack((space_mask, torch.ones((len(space_mask), 1)), 1 - space_mask), - 1)), tau=tau,
            mu=self.mu)

    def compute_loss(self, data, rep_prob, data_prob, alpha):
        dist, dist_dim = self.compute_dim_dist(data, self.cluster_rep, self.rep_mask)
        exp = torch.exp(-alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, self.n_clusters)))
        exp_compar = exp.unsqueeze(2).expand(-1, -1, 3) * rep_prob
        sun_exp_compar = torch.sum(exp_compar, 1)
        sun_exp_compar = sun_exp_compar + torch.ones(sun_exp_compar.size()) * 1e-30
        exp_compar_norm = (exp_compar / sun_exp_compar).unsqueeze(1).expand(-1, self.n_clusters, -1)
        f_x = torch.sum(dist.unsqueeze(2).expand(-1, -1, 3) * exp_compar_norm, 1)
        loss_com_kmeans = torch.mean(torch.sum(data_prob * f_x, 1))
        return loss_com_kmeans, dist, dist_dim

    def update_rep_mask(self, data, alpha=1):
        _, dist_dim = self.compute_dim_dist(data, self.cluster_rep, self.rep_mask)
        rep_mask, if_continue = self.update_mask_fun(dist_dim=dist_dim, eta=self.eta, alpha=alpha,
                                                     rep_mask=self.rep_mask)
        self.rep_mask = rep_mask
        if not if_continue:
            self.update_mask_fun = self.switch_mask_fun


class AutoEncoder(nn.Module):
    def __init__(self, input_size, embedding_dim, n_clusters, one_encoder=False, dist_metric='sum_of_squares',
                 random_state=None, hidden_1_size=500, hidden_2_size=500, hidden_3_size=2000):
        super().__init__()
        self.seed = random_state
        if self.seed is not None:
            utils.fix_seed(self.seed)
        self.input_dim = input_size
        self.n_clusters = n_clusters
        self.one_encoder = one_encoder
        self.embedding_size = self.def_embedding_dim(embedding_dim)
        self.compute_loss_fun = distance_util.set_dist_function('loss_' + dist_metric)

        self.encoder_1 = nn.Sequential(nn.Linear(input_size, hidden_1_size), nn.ReLU(),
                                       nn.Linear(hidden_1_size, hidden_2_size), nn.ReLU(),
                                       nn.Linear(hidden_2_size, hidden_3_size), nn.ReLU(),
                                       nn.Linear(hidden_3_size, self.embedding_size))
        self.encoder_2 = nn.Sequential(nn.Linear(input_size, hidden_1_size), nn.ReLU(),
                                       nn.Linear(hidden_1_size, hidden_2_size), nn.ReLU(),
                                       nn.Linear(hidden_2_size, hidden_3_size), nn.ReLU(),
                                       nn.Linear(hidden_3_size, self.embedding_size))
        self.decoder = nn.Sequential(nn.Linear(self.embedding_size, hidden_3_size), nn.ReLU(),
                                     nn.Linear(hidden_3_size, hidden_2_size), nn.ReLU(),
                                     nn.Linear(hidden_2_size, hidden_1_size), nn.ReLU(),
                                     nn.Linear(hidden_1_size, input_size))

    def def_embedding_dim(self, type_emb_size):
        if type(type_emb_size) is int:
            embedding_dim_ = type_emb_size
        elif type_emb_size[0] == 'x' and type_emb_size[1:].isnumeric():
            embedding_dim_ = self.n_clusters * int(type_emb_size[1:])
        else:
            embedding_dim_ = self.n_clusters
        return embedding_dim_

    def forward(self, x, mask):
        """
        :param x: batch_size x in_dim
        :param mask: batch_size x 1
        :return:
        """
        if self.one_encoder:
            h = self.encoder_1(x)
        else:
            h_1 = self.encoder_1(x)
            h_1 = h_1 * mask
            h_2 = self.encoder_2(x)
            h_2 = h_2 * (1 - mask)
            h = h_1 + h_2
        y = self.decoder(h)
        return y, h

    def compute_loss_ae(self, x, mask):
        y, h = self.forward(x, mask)
        loss_ae = self.compute_loss_fun(x, y)
        return loss_ae, h


class DeepKmeansBase(KmeansBase):
    def __init__(self, input_size, random_state=None, n_clusters=5, lambda_=1, dist_metric='sum_of_squares',
                 strategy='random', centroid_info=None,
                 one_encoder=False, hidden_1_size=500, hidden_2_size=500, hidden_3_size=2000, embedding_dim='x10'):

        self.embedding_size = self.def_embedding_dim(embedding_dim, n_clusters)
        super().__init__(n_clusters=n_clusters, random_state=random_state, dist_metric=dist_metric,
                         features_dim=self.embedding_size, strategy=strategy, centroid_info=centroid_info)

        self.lambda_ = lambda_
        self.one_encoder = one_encoder

        self.compute_ae_loss_fun = distance_util.set_dist_function('loss' + dist_metric)

        self.encoder_1 = nn.Sequential(nn.Linear(input_size, hidden_1_size), nn.ReLU(),
                                       nn.Linear(hidden_1_size, hidden_2_size), nn.ReLU(),
                                       nn.Linear(hidden_2_size, hidden_3_size), nn.ReLU(),
                                       nn.Linear(hidden_3_size, self.embedding_size))
        self.encoder_2 = nn.Sequential(nn.Linear(input_size, hidden_1_size), nn.ReLU(),
                                       nn.Linear(hidden_1_size, hidden_2_size), nn.ReLU(),
                                       nn.Linear(hidden_2_size, hidden_3_size), nn.ReLU(),
                                       nn.Linear(hidden_3_size, self.embedding_size))
        self.decoder = nn.Sequential(nn.Linear(self.embedding_size, hidden_3_size), nn.ReLU(),
                                     nn.Linear(hidden_3_size, hidden_2_size), nn.ReLU(),
                                     nn.Linear(hidden_2_size, hidden_1_size), nn.ReLU(),
                                     nn.Linear(hidden_1_size, input_size))

    def def_embedding_dim(self, type_emb_size, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        if type(type_emb_size) is int:
            embedding_dim_ = type_emb_size
        elif type_emb_size[0] == 'x' and type_emb_size[1:].isnumeric():
            embedding_dim_ = n_clusters * int(type_emb_size[1:])
        else:
            embedding_dim_ = n_clusters
        return embedding_dim_

    def forward(self, x, mask):
        """
        :param x: batch_size x in_dim
        :param mask: batch_size x 1
        :return:
        """
        if self.one_encoder:
            h = self.encoder_1(x)
        else:
            h_1 = self.encoder_1(x)
            h_1 = h_1 * mask
            h_2 = self.encoder_2(x)
            h_2 = h_2 * (1 - mask)
            h = h_1 + h_2
        y = self.decoder(h)
        return y, h

    def load_part_of_state_dict(self, state_dict):
        model_dict = self.state_dict()
        update_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(update_state_dict)
        self.load_state_dict(model_dict)


class DeepAlphaKmeans(DeepKmeansBase):
    def __init__(self, input_size, random_state=None, n_clusters=5, lambda_=1, dist_metric='sum_of_squares',
                 strategy='random', centroid_info=None,
                 one_encoder=False, hidden_1_size=500, hidden_2_size=500, hidden_3_size=2000, embedding_dim='x10',
                 ):
        super().__init__(input_size=input_size, lambda_=lambda_, n_clusters=n_clusters, random_state=random_state,
                         dist_metric=dist_metric,  strategy=strategy, centroid_info=centroid_info,
                         one_encoder=one_encoder, hidden_1_size=hidden_1_size, hidden_2_size=hidden_2_size,
                         hidden_3_size=hidden_3_size, embedding_dim=embedding_dim)

    def compute_loss(self, data, data_mask, alpha):
        y, h = self.forward(data, data_mask)

        dist = self.compute_dist(h, self.cluster_rep)
        exp = torch.exp(-alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, self.n_clusters)))
        softmax = exp / torch.sum(exp, 1).unsqueeze(1)

        loss_kmeans = torch.mean(torch.sum(softmax * dist, 1))
        loss_ae = self.compute_ae_loss_fun(data, y)
        loss = loss_ae + self.lambda_ * loss_kmeans

        return loss_ae, loss_kmeans, loss, dist, h


class DeepMaskAlphaKmeans(DeepKmeansBase):
    def __init__(self, input_size, eta=0.95, mask_update_type=None,  random_state=None, n_clusters=5, lambda_=1,
                 dist_metric='sum_of_squares', strategy='random', centroid_info=None,
                 one_encoder=False, hidden_1_size=500, hidden_2_size=500, hidden_3_size=2000, embedding_dim='x10',
                 ):
        super().__init__(input_size=input_size, lambda_=lambda_, n_clusters=n_clusters, random_state=random_state,
                         dist_metric=dist_metric,  strategy=strategy, centroid_info=centroid_info,
                         one_encoder=one_encoder, hidden_1_size=hidden_1_size, hidden_2_size=hidden_2_size,
                         hidden_3_size=hidden_3_size, embedding_dim=embedding_dim)

        # Mask init
        self.eta = eta
        self.rep_mask = torch.ones(self.cluster_rep.shape)

        if '+' in mask_update_type:
            update_type = mask_update_type[: mask_update_type.rfind('+')]
            switch_type = mask_update_type[mask_update_type.rfind('+') + 1:]
        else:
            update_type, switch_type = mask_update_type, 'sampling'

        self.update_mask_fun = mask_util.set_mask_update(update_type=update_type)
        self.switch_mask_fun = mask_util.set_mask_update(update_type=switch_type)
        self.compute_dim_dist = distance_util.set_dist_function(dist_name=dist_metric + '_dim', lib='torch')

    def compute_loss(self, data, data_mask, alpha):
        y, h = self.forward(data, data_mask)

        dist, dist_dim = self.compute_dim_dist(h, self.cluster_rep, self.rep_mask)
        exp = torch.exp(-alpha * (dist - torch.min(dist, 1)[0].unsqueeze(1).expand(-1, self.n_clusters)))
        softmax = exp / torch.sum(exp, 1).unsqueeze(1)

        loss_kmeans = torch.mean(torch.sum(softmax * dist, 1))
        loss_ae = self.compute_ae_loss_fun(data, y)
        loss = loss_ae + self.lambda_ * loss_kmeans

        return loss_ae, loss_kmeans, loss, dist, h, dist_dim

    def update_rep_mask(self, data, alpha=1):
        _, dist_dim = self.compute_dim_dist(data, self.cluster_rep, self.rep_mask)
        rep_mask, if_continue = self.update_mask_fun(dist_dim=dist_dim, eta=self.eta, alpha=alpha,
                                                     rep_mask=self.rep_mask)
        self.rep_mask = rep_mask
        if not if_continue:
            self.update_mask_fun = self.switch_mask_fun


if __name__ == '__main__':
    print('ok')
