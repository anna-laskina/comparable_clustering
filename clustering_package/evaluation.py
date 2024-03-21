import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import rand_score, adjusted_mutual_info_score, accuracy_score, adjusted_rand_score


def sum_up(matrix):
    # sum_upper_triangular_matrix
    return (np.sum(matrix) - np.trace(matrix)) / 2


def hullermeier_fri(P, Q):
    equivalence_relation_P = 1 - distance_matrix(P, P, p=1) / P.shape[1]
    equivalence_relation_Q = 1 - distance_matrix(Q, Q, p=1) / Q.shape[1]
    return 1 -  \
           (2 / (P.shape[0] * (P.shape[0] - 1))) * (sum_up(np.abs(equivalence_relation_P - equivalence_relation_Q)))


def our_fri(P, Q):
    connectivity_matrix_P = cosine_similarity(P, P)
    connectivity_matrix_Q = cosine_similarity(Q, Q)

    # q = connectivity_matrix_P != 0.0
    n_of_pairs = int(P.shape[0] * (P.shape[0] - 1) * 0.5)
    matrix_delta = (connectivity_matrix_P - connectivity_matrix_Q) ** 2
    matrix_A = np.where(connectivity_matrix_P != 0.0, matrix_delta, 0.0)
    matrix_B = matrix_delta - matrix_A
    val_a, val_b = np.sum(matrix_A) / 2,  np.sum(matrix_B) / 2
    alpha_1 = (np.count_nonzero(connectivity_matrix_P) - np.count_nonzero(np.diagonal(connectivity_matrix_P))) / 2
    alpha_0 = n_of_pairs - alpha_1
    # alpha_0 = (np.count_nonzero(connectivity_matrix_P != 0.0)-np.count_nonzero(np.diagonal(connectivity_matrix_P)))/2
    # alpha_1 = n_of_pairs - alpha_0
    lambda_0 = alpha_1 / alpha_0 if alpha_0 * alpha_1 != 0 else 1
    lambda_1 = alpha_0 / alpha_1 if alpha_1 * alpha_0 != 0 else 1
    return 1 - (1 / n_of_pairs) * (lambda_1 * val_a + lambda_0 * val_b)


METRIC_ABBREV = {'rand': rand_score, 'acc': accuracy_score, 'ari': adjusted_rand_score,
                 'ami': adjusted_mutual_info_score, 'fri_h': hullermeier_fri, 'fri_o': our_fri,
                 }


def evaluate(y_true, y_pred, metrics):
    return [METRIC_ABBREV[metric](y_true, y_pred) for metric in metrics]


if __name__ == '__main__':
    print('ok')
