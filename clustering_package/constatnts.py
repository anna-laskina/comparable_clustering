SAVE_ROOT_PATH = 'output'
SAVE_PATH_VER = 'output/models'
EMB_PATH = 'output/emb'
SEED_VALS = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]
ALG_NAMES = {
    'ae': 'AE',
    's': 'Sklearn',
    'bk': 'BatchKmeans',
    'ak': 'AlphaKmeans',
    'mak': 'MaskAlphaKmeans',
    'cak': 'ComparableAlphaKmeans',
    'mcak': 'MaskComparableAlphaKmeans',
    'dak': 'DeepAlphaKmeans',
    'dmak': 'DeepMaskAlphaKmeans',
}


EMBEDDING_MODEL_INFO = {
    'st_pmL': {
        'st_model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'normalize': True
    },
    'emb_st': {
        'st_model_name': 'all-MiniLM-L6-v2',
        'normalize': True
    },
    'emb_stpm': {
        'st_model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'normalize': True
    },
}

MODEL_PARAMS = {
    'ae': {
        'batch_size': 256, 'shuffle': True,
        'one_encoder': False, 'dist_metric': 'sum_of_squares',
        'hidden_1_size': 500, 'hidden_2_size': 500, 'hidden_3_size': 2000,
        'n_epochs': 300, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': False, 'delta': None, 'tolerance': None, 'embedding_dim': 'x1', 'source': 'emb_stpm',
    },
    's': {
        'init_strategy': 'k-means++',
    },
    'bk': {
        'batch_size': 256, 'shuffle': True,
        'init_strategy': 'k-means++', 'dist_function': 'sum_of_squares',
        'n_epochs': 300, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': False, 'delta': None, 'tolerance': None,
    },
    'ak': {
        'batch_size': 256, 'shuffle': True,
        'init_strategy': 'k-means++', 'dist_function': 'sum_of_squares',
        'n_epochs': 300, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': False, 'delta': None, 'tolerance': None,
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
    },
    'mak': {
        'batch_size': 256, 'shuffle': True,
        'init_strategy': 'k-means++', 'dist_function': 'sum_of_squares',
        'n_epochs': 300, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': False, 'delta': None, 'tolerance': None,
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'eta': 0.95, 'mask_update_type': 'elimination+non',
    },
    'cak': {
        'batch_size': 256, 'shuffle': True,
        'init_strategy': 'k-means++', 'dist_function': 'sum_of_squares',
        'n_epochs': 300, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': False, 'delta': None, 'tolerance': None,
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'tau_annealing_start': 1.0, 'tau_annealing_max': 1.0, 'tau_base': 2.0,
        'mu': 2.0, 'compute_rep_frequency': 'epoch',  # 'ones', 'batch',
    },
    'mcak': {
        'batch_size': 256, 'shuffle': True,
        'init_strategy': 'k-means++', 'dist_function': 'sum_of_squares',
        'n_epochs': 300, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': False, 'delta': None, 'tolerance': None,
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'tau_annealing_start': 1.0, 'tau_annealing_max': 1.0, 'tau_base': 2.0,
        'mu': 2.0, 'compute_rep_frequency': 'epoch',  # 'ones', 'batch',
        'eta': 0.95, 'mask_update_type': 'elimination+non',
    },
    'dak': {
        'batch_size': 256, 'shuffle': True,
        'init_strategy': 'k-means++', 'dist_function': 'sum_of_squares',
        'one_encoder': False, 'dist_metric': 'sum_of_squares',
        'hidden_1_size': 500, 'hidden_2_size': 500, 'hidden_3_size': 2000, 'embedding_dim': 'x1',
        'n_epochs': 300, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': False, 'delta': None, 'tolerance': None,
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'pre_train': False, 'ae_ver': None, 'lambda_': 1,
    },
    'dmak': {
        'batch_size': 256, 'shuffle': True,
        'init_strategy': 'k-means++', 'dist_function': 'sum_of_squares',
        'one_encoder': False, 'dist_metric': 'sum_of_squares',
        'hidden_1_size': 500, 'hidden_2_size': 500, 'hidden_3_size': 2000, 'embedding_dim': 'x1',
        'n_epochs': 300, 'lr': 1e-4, 'lr_s': 1.0, 'scheduler_alg': 'without',
        'early_stop': False, 'delta': None, 'tolerance': None,
        'alpha_annealing_start': 1.0, 'alpha_annealing_max': 1.0, 'alpha_base': 2.0,
        'pre_train': False, 'ae_ver': None, 'lambda_': 1,
        'eta': 0.95, 'mask_update_type': 'elimination+non',
    },
}

DEF_ARG = {
    'dataset': {
        'language_1': 'en', 'language_2': 'fr', 'get_type': 'wot', 'embedding_type': 'emb_stpm',
        'dataset_path': 'source/wikipedia/', 'embedding_path': 'output/emb/',
    },
    'verbose': {
        'verbose_level': 0, 'print_val': 1,
    },
}
