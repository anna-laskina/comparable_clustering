SAVE_ROOT_PATH = 'output'
SAVE_PATH_VER = 'output/models/'


EMBEDDING_MODEL_INFO = {
    'st_pmL': {
        'st_model_name':'paraphrase-multilingual-MiniLM-L12-v2',
        'normalize' : True
    },
    'emb_st':{
        'st_model_name': 'all-MiniLM-L6-v2',
        'normalize': True
    },
    'emb_stpm': {
        'st_model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'normalize': True
    },
}

DEF_KM_PARAM = {
    'sklearn': {
        'abv': 'S',
        'lr': None, 'n_init': 1,
        'init_strategy': 'k++', 'dist_function': 'euclidean', 'low': -1, 'high': 1,
    },
    'batch': {
        'init_strategy': 'rand++', 'dist_function': 'sum_of_squares',
        'n_epochs': 5, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': True, 'delta': 0.01, 'tolerance': 100,
        'batch_size': 256, 'shuffle': True,
    },
    'alpha': {
        'abv': 'A',
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'init_strategy': 'k++', 'dist_function': 'euclidean', 'low': -1, 'high': 1,
        'n_epochs': 500, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': True, 'delta': 0.01, 'tolerance': 100,
        'batch_size': 256, 'shuffle': True,
    },
    'lead': {
        'abv': 'L',
        'eta': 0.95, 'mask_update': 'pick_exp',
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'init_strategy': 'k++', 'dist_function': 'sum_of_squares_dim', 'low': -1, 'high': 1,
        'n_epochs': 500, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': True, 'delta': 0.01, 'tolerance': 100,
        'batch_size': 256, 'shuffle': True,
    },
    'comparable': {
        'abv': 'C',
        'mu': 2.0, 'compute_rep': 'epoch',  # 'ones', 'batch',
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'tau_annealing_start': 1.0, 'tau_annealing_max': 1.0, 'tau_base': 2.0,
        'init_strategy': 'k++', 'dist_function': 'euclidean', 'low': -1, 'high': 1,
        'n_epochs': 500, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': True, 'delta': 0.01, 'tolerance': 100,
        'batch_size': 256, 'shuffle': True,
    },
    'mask_comparable': {
        'abv': 'N',
        'eta': 0.95, 'mask_update': 'pick_exp', 'mu': 2.0, 'compute_rep': 'epoch',  # 'ones', 'batch',
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'tau_annealing_start': 1.0, 'tau_annealing_max': 1.0, 'tau_base': 2.0,
        'init_strategy': 'k++', 'dist_function': 'sum_of_squares_dim', 'low': -1, 'high': 1,
        'n_epochs': 500, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': True, 'delta': 0.01, 'tolerance': 100,
        'batch_size': 256, 'shuffle': True,
    },
    'AE': {
        'batch_size': 256, 'shuffle': True,
        'one_encoder': False, 'dist_metric': 'sum_of_squares',
        'hidden_1_size': 500, 'hidden_2_size': 500, 'hidden_3_size': 2000,
        'n_epochs': 100, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': True, 'delta': 0.01, 'tolerance': 100, 'embedding_dim': 'x10', 'source': 'emb_stpm',
    },
    'dak': {
        'abv': 'DA',
        'lambda_': 1,
        'one_encoder': False, 'hidden_1_size': 500, 'hidden_2_size': 500, 'hidden_3_size': 2000, 'embedding_dim': 'x10',
        'pre_train': False, 'ae_ver': None,
        'init_strategy': 'k-means', 'low': -1, 'high': 1, 'dist_function': 'sum_of_squares',
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'n_epochs': 500, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without', 'batch_size': 256, 'shuffle': True,
        'early_stop': True, 'delta': 0.01, 'tolerance': 100,
    },
    'dmak': {
        'abv': 'DMA',
        'lambda_': 1, 'eta': 0.95, 'mask_update_type': 'sampling',
        'one_encoder': False, 'hidden_1_size': 500, 'hidden_2_size': 500, 'hidden_3_size': 2000, 'embedding_dim': 'x10',
        'pre_train': False, 'ae_ver': None,
        'init_strategy': 'k-means', 'low': -1, 'high': 1, 'dist_function': 'sum_of_squares',
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'n_epochs': 500, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without', 'batch_size': 256, 'shuffle': True,
        'early_stop': True, 'delta': 0.01, 'tolerance': 100,
    },
    'dcak': {
        'abv': 'DCA',
        'lambda_': 1, 'mu': 2.0, 'compute_rep': 'epoch',
        'one_encoder': False, 'hidden_1_size': 500, 'hidden_2_size': 500, 'hidden_3_size': 2000, 'embedding_dim': 'x10',
        'pre_train': False, 'ae_ver': None,
        'init_strategy': 'k-means', 'low': -1, 'high': 1, 'dist_function': 'sum_of_squares',
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'tau_annealing_start': 1.0, 'tau_annealing_max': 1.0, 'tau_base': 2.0,
        'n_epochs': 500, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without', 'batch_size': 256, 'shuffle': True,
        'early_stop': True, 'delta': 0.01, 'tolerance': 100,
    },
    'hdbscan': {
        'min_cluster_size': 5, 'min_samples': None,
    },

}

DEF_ARG = {
    'dataset': {
        'language_1': 'en', 'language_2': 'fr', 'get_type': 'wot', 'embedding_type': 'emb_stpm',
        'dataset_path': 'source/wikipedia/', 'embedding_path': 'output/emb/AE/',
    },
    'verbose': {
        'verbose_level': 0, 'print_val': 1, 'save_verbose': True, 'save_val': 1, 'balance': 1,
        'compute_score': False, 'metrics': None, 'target': None, 'target_matrix': None,
        'tqdm_off': True, 'without_print': False
    },
    'assign': {
        'alpha': 1.0, 'gamma': 0.75, 'assignment_type': 'hard', 'considering_cluster_type': True,
        'dist_metric': 'euclidean'
    },
    'eval': {
        'metrics': ['mtmp_B', 'ari', 'ami', 'k'], 'seeds': [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575],
        'noise': True, 'noise_mark': -1,
        'n_best': 3, 'var': True
    }
}