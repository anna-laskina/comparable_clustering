import os.path

import numpy as np
import torch
from torch.utils.data import DataLoader

from clustering_package import models, train, constatnts
from clustering_package.dataset import DKMDataset
from clustering_package.util_files import run_util, utils


def run_autoencoder(dataset_name, seed, **kwargs):
    verbose_arg = run_util.generate_arg('verbose', **kwargs)
    dataset_arg = run_util.generate_arg('dataset', **kwargs)
    ae_arg = run_util.generate_version_info('AE', **kwargs)
    save_path = kwargs.get('save_path', constatnts.SAVE_ROOT_PATH)
    ae_ver, ae_arg = run_util.version_verification('AE', alg_info=ae_arg, input_version=kwargs.get('ae_ver', None),
                                                   mode_=kwargs.get('ver_mode', 'create'),
                                                   verbose_level=verbose_arg['verbose_level'])
    dataset = DKMDataset(dataset_name=dataset_name, seed=seed,
                         language_1=dataset_arg['language_1'], language_2=dataset_arg['language_2'],
                             dataset_path=dataset_arg['dataset_path'], embedding_path=dataset_arg['embedding_path'],
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
        earlystopper = run_util.EarlyStopper(tolerance=ae_arg['tolerance'], delta=ae_arg['delta'], off=not ae_arg['early_stop'])

        model, history = train.train_AutoEncoder(dataloader=dataloader, model=model, optimizer=optimizer,
                                                 scheduler=scheduler, earlystopper=earlystopper,
                                                 n_epochs=ae_arg['n_epochs'], verbose_level=verbose_arg['verbose_level'],
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

