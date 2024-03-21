import os.path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from clustering_package import models, train, constatnts
from clustering_package.dataset import DKMDataset
from clustering_package.util_files import run_util, utils


def run_autoencoder(dataset_name, seed, **kwargs):
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    ae_arg = run_util.generate_version_info('ae', **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    ae_ver, ae_arg = run_util.version_verification('ae', alg_info=ae_arg, input_version=kwargs.get('ae_ver', None),
                                                   mode_=kwargs.get('ver_mode', 'create'),
                                                   verbose_level=verbose_arg['verbose_level'])
    dataset = DKMDataset(dataset_name=dataset_name, seed=seed, dataset_path=dataset_arg['dataset_path'],
                         embedding_path=dataset_arg['embedding_path'],
                         embedding_type=dataset_arg['embedding_type'], get_type=dataset_arg['get_type'],
                         emb_kwargs=kwargs)
    dataloader = DataLoader(dataset, batch_size=ae_arg['batch_size'], shuffle=ae_arg['shuffle'])

    model_save_path = os.path.join(save_path, f'models/AE/model/dataset_{dataset.dataset_name}/')
    emb_save_path = os.path.join(save_path, f'emb/AE/dataset_{dataset.dataset_name}/')

    utils.path_check(path=model_save_path, if_create=True)
    utils.path_check(path=emb_save_path, if_create=True)

    model = models.AutoEncoder(input_size=dataset.embedding_size, embedding_dim=ae_arg['embedding_dim'],
                               n_clusters=dataset.n_clusters, one_encoder=ae_arg['one_encoder'],
                               dist_metric=ae_arg['dist_metric'], random_state=seed,
                               hidden_1_size=ae_arg['hidden_1_size'],
                               hidden_2_size=ae_arg['hidden_2_size'], hidden_3_size=ae_arg['hidden_3_size'])

    if os.path.exists(os.path.join(model_save_path, f'model_{dataset.dataset_name}_ae{ae_ver}_s{seed}.pt')):
        utils.verbose_print('The model with this parameters have already been trained', verbose_arg['verbose_level'], 1)
        state_dict, _, _ = utils.load_model(
            os.path.join(model_save_path, f'model_{dataset.dataset_name}_ae{ae_ver}_s{seed}.pt'))
        model_dict = model.state_dict()
        update_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(update_state_dict)
        model.load_state_dict(model_dict)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=ae_arg['lr'])
        scheduler = run_util.set_lr_scheduler(optimizer=optimizer, scheduler_alg=ae_arg['scheduler_alg'],
                                              lr_scheduler=ae_arg['lr_s'], n_epochs=ae_arg['n_epochs'],
                                              learning_rate=ae_arg['lr'])
        earlystopper = run_util.EarlyStopper(tolerance=ae_arg['tolerance'], delta=ae_arg['delta'],
                                             off=not ae_arg['early_stop'])

        model, history = train.train_AutoEncoder(dataloader=dataloader, model=model, optimizer=optimizer,
                                                 scheduler=scheduler, earlystopper=earlystopper,
                                                 n_epochs=ae_arg['n_epochs'],
                                                 verbose_level=verbose_arg['verbose_level'],
                                                 print_val=verbose_arg['print_val'])

        utils.save_model(model, None, None, os.path.join(model_save_path,
                                                         f'model_{dataset.dataset_name}_ae{ae_ver}_s{seed}.pt'))
    with torch.no_grad():
        embeddings = np.zeros((len(dataset.data), model.embedding_size), dtype=float)
        for i, batch in enumerate(dataloader, 0):
            data, mask, indices = batch
            _, batch_embedding = model.compute_loss_ae(data.float(), mask)
            for j in range(len(indices)):
                embeddings[indices[j], :] = batch_embedding.detach().numpy()[j, :]

    utils.save_data(embeddings, os.path.join(emb_save_path, f'embedding_ae{ae_ver}_s{seed}.npy'), data_type='np')


def run_Sklearn(dataset_name, seed, **kwargs):
    alg_abv = 's'
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    alg_arg = run_util.generate_version_info(alg_abv, **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    alg_ver, alg_arg = run_util.version_verification(alg_abv, alg_info=alg_arg,
                                                     input_version=kwargs.get('alg_ver', None),
                                                     mode_=kwargs.get('ver_mode', 'create'),
                                                     verbose_level=verbose_arg['verbose_level'] - 1)

    dataset_ver, dataset_arg = run_util.version_verification('dataset', alg_info=dataset_arg,
                                                             input_version=kwargs.get('dataset_ver', None),
                                                             mode_=kwargs.get('dataset_ver_mode', 'create'),
                                                             verbose_level=verbose_arg['verbose_level'] - 1)
    run_mode = kwargs.get('run_mode', 'train')

    dataset = DKMDataset(dataset_name=dataset_name, seed=seed,
                         dataset_path=dataset_arg['dataset_path'], embedding_path=dataset_arg['embedding_path'],
                         embedding_type=dataset_arg['embedding_type'], get_type=dataset_arg['get_type'],
                         emb_kwargs=kwargs)

    cluster_save_path = os.path.join(save_path,
                                     f'clusters/{constatnts.ALG_NAMES[alg_abv]}/dataset_{dataset.dataset_name}/')
    clusters_file_name = f'{alg_abv}_clusters_d{dataset_ver}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=cluster_save_path, if_create=True)

    rep_save_path = os.path.join(save_path,
                                 f'models/{constatnts.ALG_NAMES[alg_abv]}/rep/dataset_{dataset.dataset_name}/')
    rep_file_name = f'rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=rep_save_path, if_create=True)

    utils.verbose_print(f'Data {dataset_name} dv{dataset_ver} {alg_abv}_v{alg_ver}', verbose_arg['verbose_level'], 0)
    model = KMeans(n_clusters=dataset.n_clusters, init=alg_arg['init_strategy'], random_state=seed)

    existence = os.path.exists(os.path.join(cluster_save_path, clusters_file_name))
    if existence and run_mode == 'train':
        utils.verbose_print('The model with this parameters have already been trained', verbose_arg['verbose_level'], 0)
    else:
        if existence and run_mode == 'retrain':
            utils.verbose_print(
                'The model with this parameters have already been trained, BUT now it is going to retraining.',
                verbose_arg['verbose_level'], 0)
        elif existence and run_mode == 'debug':
            utils.verbose_print('The model with this parameters have already been trained. Debug mode.',
                                verbose_arg['verbose_level'], 0)

        model.fit(dataset.data)

        if run_mode != 'debug':
            utils.save_data(model.labels_, os.path.join(cluster_save_path, clusters_file_name), data_type='np')
            utils.save_data(model.cluster_centers_, os.path.join(rep_save_path, rep_file_name), data_type='np')


def run_KmeansBatch(dataset_name, seed, **kwargs):
    alg_abv = 'bk'
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    alg_arg = run_util.generate_version_info(alg_abv, **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    alg_ver, alg_arg = run_util.version_verification(alg_abv, alg_info=alg_arg,
                                                     input_version=kwargs.get('alg_ver', None),
                                                     mode_=kwargs.get('ver_mode', 'create'),
                                                     verbose_level=verbose_arg['verbose_level'] - 1)

    dataset_ver, dataset_arg = run_util.version_verification('dataset', alg_info=dataset_arg,
                                                             input_version=kwargs.get('dataset_ver', None),
                                                             mode_=kwargs.get('dataset_ver_mode', 'create'),
                                                             verbose_level=verbose_arg['verbose_level'] - 1)
    run_mode = kwargs.get('run_mode', 'train')

    dataset = DKMDataset(dataset_name=dataset_name, seed=seed,
                         dataset_path=dataset_arg['dataset_path'], embedding_path=dataset_arg['embedding_path'],
                         embedding_type=dataset_arg['embedding_type'], get_type=dataset_arg['get_type'],
                         emb_kwargs=kwargs)
    dataloader = DataLoader(dataset, batch_size=alg_arg['batch_size'], shuffle=alg_arg['shuffle'])

    # model_save_path = os.path.join(save_path, f'models/AlphaKmeans/model/dataset_{dataset.dataset_name}/')
    # model_file_name = f'model_{dataset.dataset_name}_v{alg_ver}_s{seed}.pt'
    # utils.path_check(path=model_save_path, if_create=True)

    cluster_save_path = os.path.join(save_path,
                                     f'clusters/{constatnts.ALG_NAMES[alg_abv]}/dataset_{dataset.dataset_name}/')
    clusters_file_name = f'{alg_abv}_clusters_d{dataset_ver}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=cluster_save_path, if_create=True)

    rep_save_path = os.path.join(save_path,
                                 f'models/{constatnts.ALG_NAMES[alg_abv]}/rep/dataset_{dataset.dataset_name}/')
    rep_file_name = f'rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=rep_save_path, if_create=True)

    utils.verbose_print(f'Data {dataset_name} dv{dataset_ver} {alg_abv}_v{alg_ver}', verbose_arg['verbose_level'], 0)
    model = models.BatchKmeans(n_clusters=dataset.n_clusters, random_state=seed, dist_metric=alg_arg['dist_function'],
                               features_dim=dataset.embedding_size, strategy=alg_arg['init_strategy'],
                               centroid_info=dataset.data)

    existence = os.path.exists(os.path.join(cluster_save_path, clusters_file_name))
    if existence and run_mode == 'train':
        utils.verbose_print('The model with this parameters have already been trained', verbose_arg['verbose_level'], 0)
    else:
        if existence and run_mode == 'retrain':
            utils.verbose_print(
                'The model with this parameters have already been trained, BUT now it is going to retraining.',
                verbose_arg['verbose_level'], 0)
        elif existence and run_mode == 'debug':
            utils.verbose_print('The model with this parameters have already been trained. Debug mode.',
                                verbose_arg['verbose_level'], 0)

        optimizer = torch.optim.Adam(model.parameters(), lr=alg_arg['lr'])
        scheduler = run_util.set_lr_scheduler(optimizer=optimizer, scheduler_alg=alg_arg['scheduler_alg'],
                                              lr_scheduler=alg_arg['lr_s'], n_epochs=alg_arg['n_epochs'],
                                              learning_rate=alg_arg['lr'])
        earlystopper = run_util.EarlyStopper(tolerance=alg_arg['tolerance'], delta=alg_arg['delta'],
                                             off=not alg_arg['early_stop'])

        model, history = train.train_BatchKmeans(model=model, dataloader=dataloader, optimizer=optimizer,
                                                 scheduler=scheduler, earlystopper=earlystopper,
                                                 n_epochs=alg_arg['n_epochs'],
                                                 verbose_level=verbose_arg['verbose_level'] - 1,
                                                 print_val=verbose_arg['print_val'])

        with torch.no_grad():
            clusters = model.clusters(dataset.data).numpy()
        if run_mode != 'debug':
            utils.save_data(clusters, os.path.join(cluster_save_path, clusters_file_name), data_type='np')
            utils.save_data(model.cluster_rep.detach().numpy(), os.path.join(rep_save_path, rep_file_name),
                            data_type='np')


def run_AlphaKmeans(dataset_name, seed, **kwargs):
    alg_abv = 'ak'
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    alg_arg = run_util.generate_version_info('ak', **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    alg_ver, alg_arg = run_util.version_verification('ak', alg_info=alg_arg, input_version=kwargs.get('alg_ver', None),
                                                     mode_=kwargs.get('ver_mode', 'create'),
                                                     verbose_level=verbose_arg['verbose_level'] - 1)

    dataset_ver, dataset_arg = run_util.version_verification('dataset', alg_info=dataset_arg,
                                                             input_version=kwargs.get('dataset_ver', None),
                                                             mode_=kwargs.get('dataset_ver_mode', 'create'),
                                                             verbose_level=verbose_arg['verbose_level'] - 1)
    run_mode = kwargs.get('run_mode', 'train')

    dataset = DKMDataset(dataset_name=dataset_name, seed=seed,
                         dataset_path=dataset_arg['dataset_path'], embedding_path=dataset_arg['embedding_path'],
                         embedding_type=dataset_arg['embedding_type'], get_type=dataset_arg['get_type'],
                         emb_kwargs=kwargs)
    dataloader = DataLoader(dataset, batch_size=alg_arg['batch_size'], shuffle=alg_arg['shuffle'])

    # model_save_path = os.path.join(save_path, f'models/AlphaKmeans/model/dataset_{dataset.dataset_name}/')
    # model_file_name = f'model_{dataset.dataset_name}_v{alg_ver}_s{seed}.pt'
    # utils.path_check(path=model_save_path, if_create=True)

    cluster_save_path = os.path.join(save_path,
                                     f'clusters/{constatnts.ALG_NAMES[alg_abv]}/dataset_{dataset.dataset_name}/')
    clusters_file_name = f'{alg_abv}_clusters_d{dataset_ver}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=cluster_save_path, if_create=True)

    rep_save_path = os.path.join(save_path,
                                 f'models/{constatnts.ALG_NAMES[alg_abv]}/rep/dataset_{dataset.dataset_name}/')
    rep_file_name = f'rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=rep_save_path, if_create=True)

    utils.verbose_print(f'Data {dataset_name} dv{dataset_ver} ak_v{alg_ver}', verbose_arg['verbose_level'], 0)
    model = models.AlphaKmeans(n_clusters=dataset.n_clusters, random_state=seed, dist_metric=alg_arg['dist_function'],
                               features_dim=dataset.embedding_size, strategy=alg_arg['init_strategy'],
                               centroid_info=dataset.data)

    existence = os.path.exists(os.path.join(cluster_save_path, clusters_file_name))
    if existence and run_mode == 'train':
        utils.verbose_print('The model with this parameters have already been trained', verbose_arg['verbose_level'], 0)
    else:
        if existence and run_mode == 'retrain':
            utils.verbose_print(
                'The model with this parameters have already been trained, BUT now it is going to retraining.',
                verbose_arg['verbose_level'], 0)
        elif existence and run_mode == 'debug':
            utils.verbose_print('The model with this parameters have already been trained. Debug mode.',
                                verbose_arg['verbose_level'], 0)

        optimizer = torch.optim.Adam(model.parameters(), lr=alg_arg['lr'])
        scheduler = run_util.set_lr_scheduler(optimizer=optimizer, scheduler_alg=alg_arg['scheduler_alg'],
                                              lr_scheduler=alg_arg['lr_s'], n_epochs=alg_arg['n_epochs'],
                                              learning_rate=alg_arg['lr'])
        earlystopper = run_util.EarlyStopper(tolerance=alg_arg['tolerance'], delta=alg_arg['delta'],
                                             off=not alg_arg['early_stop'])
        train_args = {'alphas': run_util.create_annealing_array(
            n_size=alg_arg['n_epochs'], annealing_start_val=alg_arg['alpha_annealing_start'],
            annealing_max_val=alg_arg['alpha_annealing_max'], base=alg_arg['alpha_base'])
        }

        model, history = train.train_AlphaKmeans(model=model, dataloader=dataloader, optimizer=optimizer,
                                                 scheduler=scheduler, earlystopper=earlystopper,
                                                 n_epochs=alg_arg['n_epochs'], train_args=train_args,
                                                 verbose_level=verbose_arg['verbose_level'] - 1,
                                                 print_val=verbose_arg['print_val'])
        # if run_mode != 'debug':
        #    utils.save_model(model, None, None, os.path.join(model_save_path, model_file_name))

        with torch.no_grad():
            clusters = model.clusters(dataset.data, train_args['alphas'][-1]).numpy()
        if run_mode != 'debug':
            utils.save_data(clusters, os.path.join(cluster_save_path, clusters_file_name), data_type='np')
            utils.save_data(model.cluster_rep.detach().numpy(), os.path.join(rep_save_path, rep_file_name),
                            data_type='np')


def run_MaskAlphaKmeans(dataset_name, seed, **kwargs):
    alg_abv = 'mak'
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    alg_arg = run_util.generate_version_info('mak', **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    alg_ver, alg_arg = run_util.version_verification('mak', alg_info=alg_arg, input_version=kwargs.get('alg_ver', None),
                                                     mode_=kwargs.get('ver_mode', 'create'),
                                                     verbose_level=verbose_arg['verbose_level'] - 1)

    dataset_ver, dataset_arg = run_util.version_verification('dataset', alg_info=dataset_arg,
                                                             input_version=kwargs.get('dataset_ver', None),
                                                             mode_=kwargs.get('dataset_ver_mode', 'create'),
                                                             verbose_level=verbose_arg['verbose_level'] - 1)
    run_mode = kwargs.get('run_mode', 'train')

    dataset = DKMDataset(dataset_name=dataset_name, seed=seed,
                         dataset_path=dataset_arg['dataset_path'], embedding_path=dataset_arg['embedding_path'],
                         embedding_type=dataset_arg['embedding_type'], get_type=dataset_arg['get_type'],
                         emb_kwargs=kwargs)
    dataloader = DataLoader(dataset, batch_size=alg_arg['batch_size'], shuffle=alg_arg['shuffle'])

    cluster_save_path = os.path.join(save_path,
                                     f'clusters/{constatnts.ALG_NAMES[alg_abv]}/dataset_{dataset.dataset_name}/')
    clusters_file_name = f'{alg_abv}_clusters_d{dataset_ver}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=cluster_save_path, if_create=True)

    mask_save_path = os.path.join(save_path,
                                  f'models/{constatnts.ALG_NAMES[alg_abv]}/mask/dataset_{dataset.dataset_name}/')
    mask_file_name = f'mask_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=mask_save_path, if_create=True)

    rep_save_path = os.path.join(save_path,
                                 f'models/{constatnts.ALG_NAMES[alg_abv]}/rep/dataset_{dataset.dataset_name}/')
    rep_file_name = f'rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=rep_save_path, if_create=True)

    utils.verbose_print(f'Data {dataset_name} dv{dataset_ver} mak_v{alg_ver}', verbose_arg['verbose_level'], 0)
    model = models.MaskAlphaKmeans(n_clusters=dataset.n_clusters, random_state=seed,
                                   dist_metric=alg_arg['dist_function'], features_dim=dataset.embedding_size,
                                   strategy=alg_arg['init_strategy'], centroid_info=dataset.data, eta=alg_arg['eta'],
                                   mask_update_type=alg_arg['mask_update_type'])

    existence = os.path.exists(os.path.join(cluster_save_path, clusters_file_name))
    if existence and run_mode == 'train':
        utils.verbose_print('The model with this parameters have already been trained', verbose_arg['verbose_level'], 0)
    else:
        if existence and run_mode == 'retrain':
            utils.verbose_print(
                'The model with this parameters have already been trained, BUT now it is going to retraining.',
                verbose_arg['verbose_level'], 0)
        elif existence and run_mode == 'debug':
            utils.verbose_print('The model with this parameters have already been trained. Debug mode.',
                                verbose_arg['verbose_level'], 0)

        optimizer = torch.optim.Adam(model.parameters(), lr=alg_arg['lr'])
        scheduler = run_util.set_lr_scheduler(optimizer=optimizer, scheduler_alg=alg_arg['scheduler_alg'],
                                              lr_scheduler=alg_arg['lr_s'], n_epochs=alg_arg['n_epochs'],
                                              learning_rate=alg_arg['lr'])
        earlystopper = run_util.EarlyStopper(tolerance=alg_arg['tolerance'], delta=alg_arg['delta'],
                                             off=not alg_arg['early_stop'])
        train_args = {'alphas': run_util.create_annealing_array(
            n_size=alg_arg['n_epochs'],
            annealing_start_val=alg_arg['alpha_annealing_start'],
            annealing_max_val=alg_arg['alpha_annealing_max'],
            base=alg_arg['alpha_base'])
        }

        model, history = train.train_MaskAlphaKmeans(model=model, dataloader=dataloader, optimizer=optimizer,
                                                     scheduler=scheduler, earlystopper=earlystopper,
                                                     n_epochs=alg_arg['n_epochs'], train_args=train_args,
                                                     verbose_level=verbose_arg['verbose_level'] - 1,
                                                     print_val=verbose_arg['print_val'])
        # if run_mode != 'debug':
        #    utils.save_model(model, None, None, os.path.join(model_save_path, model_file_name))

        with torch.no_grad():
            clusters = model.clusters(dataset.data, train_args['alphas'][-1]).numpy()
        if run_mode != 'debug':
            utils.save_data(clusters, os.path.join(cluster_save_path, clusters_file_name), data_type='np')
            utils.save_data(model.rep_mask.detach().numpy(), os.path.join(mask_save_path, mask_file_name),
                            data_type='np')
            utils.save_data(model.cluster_rep.detach().numpy(), os.path.join(rep_save_path, rep_file_name),
                            data_type='np')


def run_ComparableAlphaKmeans(dataset_name, seed, **kwargs):
    alg_abv = 'cak'
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    alg_arg = run_util.generate_version_info(alg_abv, **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    alg_ver, alg_arg = run_util.version_verification(alg_abv, alg_info=alg_arg,
                                                     input_version=kwargs.get('alg_ver', None),
                                                     mode_=kwargs.get('ver_mode', 'create'),
                                                     verbose_level=verbose_arg['verbose_level'] - 1)

    dataset_ver, dataset_arg = run_util.version_verification('dataset', alg_info=dataset_arg,
                                                             input_version=kwargs.get('dataset_ver', None),
                                                             mode_=kwargs.get('dataset_ver_mode', 'create'),
                                                             verbose_level=verbose_arg['verbose_level'] - 1)
    run_mode = kwargs.get('run_mode', 'train')

    dataset = DKMDataset(dataset_name=dataset_name, seed=seed,
                         dataset_path=dataset_arg['dataset_path'], embedding_path=dataset_arg['embedding_path'],
                         embedding_type=dataset_arg['embedding_type'], get_type=dataset_arg['get_type'],
                         emb_kwargs=kwargs)
    dataloader = DataLoader(dataset, batch_size=alg_arg['batch_size'], shuffle=alg_arg['shuffle'])

    cluster_save_path = os.path.join(save_path,
                                     f'clusters/{constatnts.ALG_NAMES[alg_abv]}/dataset_{dataset.dataset_name}/')
    clusters_file_name = f'{alg_abv}_clusters_d{dataset_ver}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=cluster_save_path, if_create=True)

    rep_save_path = os.path.join(save_path,
                                 f'models/{constatnts.ALG_NAMES[alg_abv]}/rep/dataset_{dataset.dataset_name}/')
    rep_file_name = f'rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=rep_save_path, if_create=True)

    utils.verbose_print(f'Data {dataset_name} dv{dataset_ver} {alg_abv}_v{alg_ver}', verbose_arg['verbose_level'], 0)
    model = models.ComparableAlphaKmeans(n_clusters=dataset.n_clusters, random_state=seed,
                                         dist_metric=alg_arg['dist_function'], features_dim=dataset.embedding_size,
                                         strategy=alg_arg['init_strategy'],
                                         centroid_info=dataset.data, mu=alg_arg['mu'])

    existence = os.path.exists(os.path.join(cluster_save_path, clusters_file_name))
    if existence and run_mode == 'train':
        utils.verbose_print('The model with this parameters have already been trained', verbose_arg['verbose_level'], 0)
    else:
        if existence and run_mode == 'retrain':
            utils.verbose_print(
                'The model with this parameters have already been trained, BUT now it is going to retraining.',
                verbose_arg['verbose_level'], 0)
        elif existence and run_mode == 'debug':
            utils.verbose_print('The model with this parameters have already been trained. Debug mode.',
                                verbose_arg['verbose_level'], 0)

        optimizer = torch.optim.Adam(model.parameters(), lr=alg_arg['lr'])
        scheduler = run_util.set_lr_scheduler(optimizer=optimizer, scheduler_alg=alg_arg['scheduler_alg'],
                                              lr_scheduler=alg_arg['lr_s'], n_epochs=alg_arg['n_epochs'],
                                              learning_rate=alg_arg['lr'])
        earlystopper = run_util.EarlyStopper(tolerance=alg_arg['tolerance'], delta=alg_arg['delta'],
                                             off=not alg_arg['early_stop'])
        train_args = {
            'alphas': run_util.create_annealing_array(n_size=alg_arg['n_epochs'],
                                                      annealing_start_val=alg_arg['alpha_annealing_start'],
                                                      annealing_max_val=alg_arg['alpha_annealing_max'],
                                                      base=alg_arg['alpha_base']),
            'taus': run_util.create_annealing_array(n_size=alg_arg['n_epochs'],
                                                    annealing_start_val=alg_arg['tau_annealing_start'],
                                                    annealing_max_val=alg_arg['tau_annealing_max'],
                                                    base=alg_arg['tau_base']),
            'compute_rep_frequency': alg_arg['compute_rep_frequency'],
        }

        model, history, data_prob, rep_prob = train.train_ComparableAlphaKmeans(model=model, dataloader=dataloader,
                                                                                optimizer=optimizer,
                                                                                scheduler=scheduler,
                                                                                earlystopper=earlystopper,
                                                                                n_epochs=alg_arg['n_epochs'],
                                                                                train_args=train_args,
                                                                                verbose_level=verbose_arg[
                                                                                                  'verbose_level'] - 1,
                                                                                print_val=verbose_arg['print_val'])
        # if run_mode != 'debug':
        #    utils.save_model(model, None, None, os.path.join(model_save_path, model_file_name))

        with torch.no_grad():
            clusters = model.clusters(dataset.data, data_prob, rep_prob, train_args['alphas'][-1]).numpy()
        if run_mode != 'debug':
            utils.save_data(clusters, os.path.join(cluster_save_path, clusters_file_name), data_type='np')
            utils.save_data(model.cluster_rep.detach().numpy(), os.path.join(rep_save_path, rep_file_name),
                            data_type='np')


def run_MaskComparableAlphaKmeans(dataset_name, seed, **kwargs):
    alg_abv = 'mcak'
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    alg_arg = run_util.generate_version_info(alg_abv, **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    alg_ver, alg_arg = run_util.version_verification(alg_abv, alg_info=alg_arg,
                                                     input_version=kwargs.get('alg_ver', None),
                                                     mode_=kwargs.get('ver_mode', 'create'),
                                                     verbose_level=verbose_arg['verbose_level'] - 1)

    dataset_ver, dataset_arg = run_util.version_verification('dataset', alg_info=dataset_arg,
                                                             input_version=kwargs.get('dataset_ver', None),
                                                             mode_=kwargs.get('dataset_ver_mode', 'create'),
                                                             verbose_level=verbose_arg['verbose_level'] - 1)
    run_mode = kwargs.get('run_mode', 'train')

    dataset = DKMDataset(dataset_name=dataset_name, seed=seed,
                         dataset_path=dataset_arg['dataset_path'], embedding_path=dataset_arg['embedding_path'],
                         embedding_type=dataset_arg['embedding_type'], get_type=dataset_arg['get_type'],
                         emb_kwargs=kwargs)
    dataloader = DataLoader(dataset, batch_size=alg_arg['batch_size'], shuffle=alg_arg['shuffle'])

    cluster_save_path = os.path.join(save_path,
                                     f'clusters/{constatnts.ALG_NAMES[alg_abv]}/dataset_{dataset.dataset_name}/')
    clusters_file_name = f'{alg_abv}_clusters_d{dataset_ver}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=cluster_save_path, if_create=True)

    mask_save_path = os.path.join(save_path,
                                  f'models/{constatnts.ALG_NAMES[alg_abv]}/mask/dataset_{dataset.dataset_name}/')
    mask_file_name = f'mask_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=mask_save_path, if_create=True)

    rep_save_path = os.path.join(save_path,
                                 f'models/{constatnts.ALG_NAMES[alg_abv]}/rep/dataset_{dataset.dataset_name}/')
    rep_file_name = f'rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=rep_save_path, if_create=True)

    utils.verbose_print(f'Data {dataset_name} dv{dataset_ver} {alg_abv}_v{alg_ver}', verbose_arg['verbose_level'], 0)
    model = models.MaskComparableAlphaKmeans(n_clusters=dataset.n_clusters, random_state=seed,
                                             dist_metric=alg_arg['dist_function'], features_dim=dataset.embedding_size,
                                             strategy=alg_arg['init_strategy'], centroid_info=dataset.data,
                                             mu=alg_arg['mu'], eta=alg_arg['eta'],
                                             mask_update_type=alg_arg['mask_update_type'])

    existence = os.path.exists(os.path.join(cluster_save_path, clusters_file_name))
    if existence and run_mode == 'train':
        utils.verbose_print('The model with this parameters have already been trained', verbose_arg['verbose_level'], 0)
    else:
        if existence and run_mode == 'retrain':
            utils.verbose_print(
                'The model with this parameters have already been trained, BUT now it is going to retraining.',
                verbose_arg['verbose_level'], 0)
        elif existence and run_mode == 'debug':
            utils.verbose_print('The model with this parameters have already been trained. Debug mode.',
                                verbose_arg['verbose_level'], 0)

        optimizer = torch.optim.Adam(model.parameters(), lr=alg_arg['lr'])
        scheduler = run_util.set_lr_scheduler(optimizer=optimizer, scheduler_alg=alg_arg['scheduler_alg'],
                                              lr_scheduler=alg_arg['lr_s'], n_epochs=alg_arg['n_epochs'],
                                              learning_rate=alg_arg['lr'])
        earlystopper = run_util.EarlyStopper(tolerance=alg_arg['tolerance'], delta=alg_arg['delta'],
                                             off=not alg_arg['early_stop'])
        train_args = {
            'alphas': run_util.create_annealing_array(n_size=alg_arg['n_epochs'],
                                                      annealing_start_val=alg_arg['alpha_annealing_start'],
                                                      annealing_max_val=alg_arg['alpha_annealing_max'],
                                                      base=alg_arg['alpha_base']),
            'taus': run_util.create_annealing_array(n_size=alg_arg['n_epochs'],
                                                    annealing_start_val=alg_arg['tau_annealing_start'],
                                                    annealing_max_val=alg_arg['tau_annealing_max'],
                                                    base=alg_arg['tau_base']),
            'compute_rep_frequency': alg_arg['compute_rep_frequency'],
        }

        model, history, data_prob, rep_prob = train.train_MaskComparableAlphaKmeans(
            model=model, dataloader=dataloader, optimizer=optimizer, scheduler=scheduler, earlystopper=earlystopper,
            n_epochs=alg_arg['n_epochs'], train_args=train_args, verbose_level=verbose_arg['verbose_level'] - 1,
            print_val=verbose_arg['print_val'])
        # if run_mode != 'debug':
        #    utils.save_model(model, None, None, os.path.join(model_save_path, model_file_name))

        with torch.no_grad():
            clusters = model.clusters(dataset.data, data_prob, rep_prob, train_args['alphas'][-1]).numpy()
        if run_mode != 'debug':
            utils.save_data(clusters, os.path.join(cluster_save_path, clusters_file_name), data_type='np')
            utils.save_data(model.rep_mask.detach().numpy(), os.path.join(mask_save_path, mask_file_name),
                            data_type='np')
            utils.save_data(model.cluster_rep.detach().numpy(), os.path.join(rep_save_path, rep_file_name),
                            data_type='np')


def run_DeepAlphaKmeans(dataset_name, seed, **kwargs):
    alg_abv = 'dak'
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    alg_arg = run_util.generate_version_info(alg_abv, **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    alg_ver, alg_arg = run_util.version_verification(alg_abv, alg_info=alg_arg,
                                                     input_version=kwargs.get('alg_ver', None),
                                                     mode_=kwargs.get('ver_mode', 'create'),
                                                     verbose_level=verbose_arg['verbose_level'] - 1)

    dataset_ver, dataset_arg = run_util.version_verification('dataset', alg_info=dataset_arg,
                                                             input_version=kwargs.get('dataset_ver', None),
                                                             mode_=kwargs.get('dataset_ver_mode', 'create'),
                                                             verbose_level=verbose_arg['verbose_level'] - 1)
    run_mode = kwargs.get('run_mode', 'train')

    dataset = DKMDataset(dataset_name=dataset_name, seed=seed,
                         dataset_path=dataset_arg['dataset_path'], embedding_path=dataset_arg['embedding_path'],
                         embedding_type=dataset_arg['embedding_type'], get_type=dataset_arg['get_type'],
                         emb_kwargs=kwargs)
    dataloader = DataLoader(dataset, batch_size=alg_arg['batch_size'], shuffle=alg_arg['shuffle'])

    # model_save_path = os.path.join(save_path, f'models/AlphaKmeans/model/dataset_{dataset.dataset_name}/')
    # model_file_name = f'model_{dataset.dataset_name}_v{alg_ver}_s{seed}.pt'
    # utils.path_check(path=model_save_path, if_create=True)

    cluster_save_path = os.path.join(save_path,
                                     f'clusters/{constatnts.ALG_NAMES[alg_abv]}/dataset_{dataset.dataset_name}/')
    clusters_file_name = f'{alg_abv}_clusters_d{dataset_ver}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=cluster_save_path, if_create=True)

    rep_save_path = os.path.join(save_path,
                                 f'models/{constatnts.ALG_NAMES[alg_abv]}/rep/dataset_{dataset.dataset_name}/')
    rep_file_name = f'rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=rep_save_path, if_create=True)

    ls_save_path = os.path.join(save_path,
                                f'models/{constatnts.ALG_NAMES[alg_abv]}/latent_space/dataset_{dataset.dataset_name}/')
    ls_file_name = f'data_rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=ls_save_path, if_create=True)

    utils.verbose_print(f'Data {dataset_name} dv{dataset_ver} {alg_abv}_v{alg_ver}', verbose_arg['verbose_level'], 0)
    model = models.DeepAlphaKmeans(
        input_size=dataset.embedding_size, random_state=seed, n_clusters=dataset.n_clusters,
        lambda_=alg_arg['lambda_'], dist_metric=alg_arg['dist_function'],
        strategy=alg_arg['init_strategy'], centroid_info=dataset.data,
        one_encoder=alg_arg['one_encoder'],
        hidden_1_size=alg_arg['hidden_1_size'], hidden_2_size=alg_arg['hidden_2_size'],
        hidden_3_size=alg_arg['hidden_3_size'], embedding_dim=alg_arg['embedding_dim']
    )
    if alg_arg['pre_train']:
        state_dict, space_info = run_util.load_pretrain_(use_pre_train=alg_arg['pre_train'],
                                                         dataset_name=dataset.dataset_name, ae_ver=alg_arg["ae_ver"],
                                                         seed=seed)
        model.load_part_of_state_dict(state_dict)

    existence = os.path.exists(os.path.join(cluster_save_path, clusters_file_name))
    if existence and run_mode == 'train':
        utils.verbose_print('The model with this parameters have already been trained', verbose_arg['verbose_level'], 0)
    else:
        if existence and run_mode == 'retrain':
            utils.verbose_print(
                'The model with this parameters have already been trained, BUT now it is going to retraining.',
                verbose_arg['verbose_level'], 0)
        elif existence and run_mode == 'debug':
            utils.verbose_print('The model with this parameters have already been trained. Debug mode.',
                                verbose_arg['verbose_level'], 0)

        optimizer = torch.optim.Adam(model.parameters(), lr=alg_arg['lr'])
        scheduler = run_util.set_lr_scheduler(optimizer=optimizer, scheduler_alg=alg_arg['scheduler_alg'],
                                              lr_scheduler=alg_arg['lr_s'], n_epochs=alg_arg['n_epochs'],
                                              learning_rate=alg_arg['lr'])
        earlystopper = run_util.EarlyStopper(tolerance=alg_arg['tolerance'], delta=alg_arg['delta'],
                                             off=not alg_arg['early_stop'])
        train_args = {'alphas': run_util.create_annealing_array(
            n_size=alg_arg['n_epochs'],
            annealing_start_val=alg_arg['alpha_annealing_start'],
            annealing_max_val=alg_arg['alpha_annealing_max'],
            base=alg_arg['alpha_base'])
        }

        model, history = train.train_DeepAlphaKmeans(model=model, dataloader=dataloader, optimizer=optimizer,
                                                     scheduler=scheduler, earlystopper=earlystopper,
                                                     n_epochs=alg_arg['n_epochs'], train_args=train_args,
                                                     verbose_level=verbose_arg['verbose_level'] - 1,
                                                     print_val=verbose_arg['print_val'])

        with torch.no_grad():
            clusters = model.clusters(dataset.data, dataset.mask, train_args['alphas'][-1]).numpy()
            latent_space = model.forward(torch.tensor(dataset.data,
                                                      dtype=torch.float32), dataset.mask)[1].detach().numpy()
        if run_mode != 'debug':
            utils.save_data(clusters, os.path.join(cluster_save_path, clusters_file_name), data_type='np')
            utils.save_data(model.cluster_rep.detach().numpy(), os.path.join(rep_save_path, rep_file_name),
                            data_type='np')
            utils.save_data(latent_space, os.path.join(ls_save_path, ls_file_name), data_type='np')


def run_DeepMaskAlphaKmeans(dataset_name, seed, **kwargs):
    alg_abv = 'dmak'
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    alg_arg = run_util.generate_version_info(alg_abv, **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    alg_ver, alg_arg = run_util.version_verification(alg_abv, alg_info=alg_arg,
                                                     input_version=kwargs.get('alg_ver', None),
                                                     mode_=kwargs.get('ver_mode', 'create'),
                                                     verbose_level=verbose_arg['verbose_level'] - 1)

    dataset_ver, dataset_arg = run_util.version_verification('dataset', alg_info=dataset_arg,
                                                             input_version=kwargs.get('dataset_ver', None),
                                                             mode_=kwargs.get('dataset_ver_mode', 'create'),
                                                             verbose_level=verbose_arg['verbose_level'] - 1)
    run_mode = kwargs.get('run_mode', 'train')

    dataset = DKMDataset(dataset_name=dataset_name, seed=seed,
                         dataset_path=dataset_arg['dataset_path'], embedding_path=dataset_arg['embedding_path'],
                         embedding_type=dataset_arg['embedding_type'], get_type=dataset_arg['get_type'],
                         emb_kwargs=kwargs)
    dataloader = DataLoader(dataset, batch_size=alg_arg['batch_size'], shuffle=alg_arg['shuffle'])

    # model_save_path = os.path.join(save_path, f'models/AlphaKmeans/model/dataset_{dataset.dataset_name}/')
    # model_file_name = f'model_{dataset.dataset_name}_v{alg_ver}_s{seed}.pt'
    # utils.path_check(path=model_save_path, if_create=True)

    cluster_save_path = os.path.join(save_path,
                                     f'clusters/{constatnts.ALG_NAMES[alg_abv]}/dataset_{dataset.dataset_name}/')
    clusters_file_name = f'{alg_abv}_clusters_d{dataset_ver}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=cluster_save_path, if_create=True)

    mask_save_path = os.path.join(save_path,
                                  f'models/{constatnts.ALG_NAMES[alg_abv]}/mask/dataset_{dataset.dataset_name}/')
    mask_file_name = f'mask_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=mask_save_path, if_create=True)

    rep_save_path = os.path.join(save_path,
                                 f'models/{constatnts.ALG_NAMES[alg_abv]}/rep/dataset_{dataset.dataset_name}/')
    rep_file_name = f'rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=rep_save_path, if_create=True)

    ls_save_path = os.path.join(save_path,
                                f'models/{constatnts.ALG_NAMES[alg_abv]}/latent_space/dataset_{dataset.dataset_name}/')
    ls_file_name = f'data_rep_d{dataset_ver}_{alg_abv}_v{alg_ver}_s{seed}.npy'
    utils.path_check(path=ls_save_path, if_create=True)

    utils.verbose_print(f'Data {dataset_name} dv{dataset_ver} {alg_abv}_v{alg_ver}', verbose_arg['verbose_level'], 0)
    model = models.DeepMaskAlphaKmeans(
        input_size=dataset.embedding_size, random_state=seed, n_clusters=dataset.n_clusters,
        lambda_=alg_arg['lambda_'], dist_metric=alg_arg['dist_function'],
        strategy=alg_arg['init_strategy'], centroid_info=dataset.data,
        one_encoder=alg_arg['one_encoder'],
        hidden_1_size=alg_arg['hidden_1_size'], hidden_2_size=alg_arg['hidden_2_size'],
        hidden_3_size=alg_arg['hidden_3_size'], embedding_dim=alg_arg['embedding_dim'],
        eta=alg_arg['eta'], mask_update_type=alg_arg['mask_update_type']
    )
    if alg_arg['pre_train']:
        state_dict, space_info = run_util.load_pretrain_(use_pre_train=alg_arg['pre_train'], seed=seed,
                                                         dataset_name=dataset.dataset_name, ae_ver=alg_arg["ae_ver"])
        model.load_part_of_state_dict(state_dict)

    existence = os.path.exists(os.path.join(cluster_save_path, clusters_file_name))
    if existence and run_mode == 'train':
        utils.verbose_print('The model with this parameters have already been trained', verbose_arg['verbose_level'], 0)
    else:
        if existence and run_mode == 'retrain':
            utils.verbose_print(
                'The model with this parameters have already been trained, BUT now it is going to retraining.',
                verbose_arg['verbose_level'], 0)
        elif existence and run_mode == 'debug':
            utils.verbose_print('The model with this parameters have already been trained. Debug mode.',
                                verbose_arg['verbose_level'], 0)

        optimizer = torch.optim.Adam(model.parameters(), lr=alg_arg['lr'])
        scheduler = run_util.set_lr_scheduler(optimizer=optimizer, scheduler_alg=alg_arg['scheduler_alg'],
                                              lr_scheduler=alg_arg['lr_s'], n_epochs=alg_arg['n_epochs'],
                                              learning_rate=alg_arg['lr'])
        earlystopper = run_util.EarlyStopper(tolerance=alg_arg['tolerance'], delta=alg_arg['delta'],
                                             off=not alg_arg['early_stop'])
        train_args = {'alphas': run_util.create_annealing_array(
            n_size=alg_arg['n_epochs'], annealing_start_val=alg_arg['alpha_annealing_start'],
            annealing_max_val=alg_arg['alpha_annealing_max'], base=alg_arg['alpha_base'])
        }

        model, history = train.train_DeepMaskAlphaKmeans(model=model, dataloader=dataloader, optimizer=optimizer,
                                                         scheduler=scheduler, earlystopper=earlystopper,
                                                         n_epochs=alg_arg['n_epochs'], train_args=train_args,
                                                         verbose_level=verbose_arg['verbose_level'] - 1,
                                                         print_val=verbose_arg['print_val'])

        with torch.no_grad():
            clusters = model.clusters(dataset.data, dataset.mask, train_args['alphas'][-1]).numpy()
            latent_space = model.forward(torch.tensor(dataset.data,
                                                      dtype=torch.float32), dataset.mask)[1].detach().numpy()
        if run_mode != 'debug':
            utils.save_data(clusters, os.path.join(cluster_save_path, clusters_file_name), data_type='np')
            utils.save_data(model.rep_mask.detach().numpy(), os.path.join(mask_save_path, mask_file_name),
                            data_type='np')

            utils.save_data(model.cluster_rep.detach().numpy(), os.path.join(rep_save_path, rep_file_name),
                            data_type='np')
            utils.save_data(latent_space, os.path.join(ls_save_path, ls_file_name), data_type='np')


if __name__ == '__main__':
    print('ok')
