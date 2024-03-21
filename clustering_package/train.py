import torch
from tqdm import tqdm

from clustering_package.util_files import utils


def train_AutoEncoder(dataloader, model, optimizer, scheduler, earlystopper, n_epochs=100,
                      verbose_level=0, print_val=1):
    history = {'loss': [], 'lr': []}
    utils.verbose_print('Starting autoencoder training...', verbose_level, 0)

    for epoch in range(n_epochs) if verbose_level != 1 else tqdm(range(n_epochs)):
        utils.verbose_print(f'Training step: epoch {epoch:3d}', verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1, print_end=' ')
        running_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            data, mask, indices = batch
            optimizer.zero_grad()
            loss_ae, batch_embedding = model.compute_loss_ae(data.float(), mask)
            loss_ae.backward()
            optimizer.step()
            running_loss += loss_ae.item()
        scheduler.step()
        earlystopper(running_loss)
        if earlystopper.early_stop:
            utils.verbose_print(f'Early stopping after {epoch} epoch.', verbose_level, 1)
            break
        history['loss'].append(running_loss)
        history['lr'].append(optimizer.param_groups[0]["lr"])
        utils.verbose_print(' '.join([f'{k}: {v[-1]:.5f}' for k, v in history.items()]), verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1)
    return model, history


def train_BatchKmeans(model, dataloader, optimizer, scheduler, earlystopper, n_epochs,
                      verbose_level=0, print_val=1):
    history = {'loss': [], 'lr': []}
    utils.verbose_print('Start training...', verbose_level, 0)
    for epoch in range(n_epochs) if verbose_level != 1 else tqdm(range(n_epochs)):
        utils.verbose_print(f'Training step: epoch {epoch:3d}', verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1, ' ')
        running_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            data, mask, indices = batch
            optimizer.zero_grad()
            loss, dist = model.compute_loss(data.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        earlystopper(running_loss)
        if earlystopper.early_stop:
            utils.verbose_print(f'Early stopping after {epoch} epoch.', verbose_level, 1)
            break
        history['loss'].append(running_loss)
        history['lr'].append(optimizer.param_groups[0]["lr"])
        utils.verbose_print(' '.join([f'{k}: {v[-1]:.5f}' for k, v in history.items()]), verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1)
    return model, history


def train_AlphaKmeans(model, dataloader, optimizer, scheduler, earlystopper, n_epochs, train_args,
                      verbose_level=0, print_val=1):
    history = {'loss': [], 'lr': []}
    utils.verbose_print('Start training...', verbose_level, 0)
    for epoch in range(n_epochs) if verbose_level != 1 else tqdm(range(n_epochs)):
        utils.verbose_print(f'Training step: epoch {epoch:3d}', verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1, ' ')
        running_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            data, mask, indices = batch
            optimizer.zero_grad()
            loss, dist = model.compute_loss(data.float(), train_args['alphas'][epoch])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        earlystopper(running_loss)
        if earlystopper.early_stop:
            utils.verbose_print(f'Early stopping after {epoch} epoch.', verbose_level, 1)
            break
        history['loss'].append(running_loss)
        history['lr'].append(optimizer.param_groups[0]["lr"])
        utils.verbose_print(' '.join([f'{k}: {v[-1]:.5f}' for k, v in history.items()]), verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1)
    return model, history


def train_MaskAlphaKmeans(model, dataloader, optimizer, scheduler, earlystopper, n_epochs, train_args,
                          verbose_level=0, print_val=1):
    history = {'loss': [], 'lr': []}
    if not torch.is_tensor(dataloader.dataset.data):
        space = torch.tensor(dataloader.dataset.data, dtype=torch.float32)
    else:
        space = dataloader.dataset.data
    utils.verbose_print('Start training...', verbose_level, 0)
    for epoch in range(n_epochs) if verbose_level != 1 else tqdm(range(n_epochs)):
        utils.verbose_print(f'Training step: epoch {epoch:3d}', verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1, ' ')
        running_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            data, mask, indices = batch
            optimizer.zero_grad()
            loss, _ = model.compute_loss(data.float(), train_args['alphas'][epoch])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.update_rep_mask(space, alpha=train_args['alphas'][epoch])

        scheduler.step()
        earlystopper(running_loss)
        if earlystopper.early_stop:
            utils.verbose_print(f'Early stopping after {epoch} epoch.', verbose_level, 1)
            break
        history['loss'].append(running_loss)
        history['lr'].append(optimizer.param_groups[0]["lr"])
        utils.verbose_print(' '.join([f'{k}: {v[-1]:.5f}' for k, v in history.items()]), verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1)
    return model, history


def train_ComparableAlphaKmeans(model, dataloader, optimizer, scheduler, earlystopper, n_epochs, train_args,
                                verbose_level=0, print_val=1):
    history = {'loss': [], 'lr': []}
    space = torch.tensor(dataloader.dataset.data, dtype=torch.float32)
    utils.verbose_print('Start training...', verbose_level, 0)
    data_prob = model.update_data_prob(space=space, space_mask=dataloader.dataset.mask, tau=train_args['taus'][0])
    rep_prob = None
    if train_args['compute_rep_frequency'] == 'ones':
        rep_prob = model.update_rep_prob(space=space, space_mask=dataloader.dataset.mask, tau=train_args['taus'][0])
    for epoch in range(n_epochs) if verbose_level != 1 else tqdm(range(n_epochs)):
        utils.verbose_print(f'Training step: epoch {epoch:3d}', verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1, ' ')
        running_loss = 0.0
        if train_args['compute_rep_frequency'] == 'epoch':
            rep_prob = model.update_rep_prob(space=space, space_mask=dataloader.dataset.mask,
                                             tau=train_args['taus'][epoch])
        for i, batch in enumerate(dataloader, 0):
            if train_args['compute_rep_frequency'] == 'batch':
                rep_prob = model.update_rep_prob(space=space, space_mask=dataloader.dataset.mask,
                                                 tau=train_args['taus'][epoch])
            data, mask, indices = batch
            data_prob_batch = data_prob[indices]
            optimizer.zero_grad()
            loss, dist = model.compute_loss(data=data.float(), rep_prob=rep_prob, data_prob=data_prob_batch,
                                            alpha=train_args['alphas'][epoch])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        earlystopper(running_loss)
        if earlystopper.early_stop:
            utils.verbose_print(f'Early stopping after {epoch} epoch.', verbose_level, 1)
            break
        history['loss'].append(running_loss)
        history['lr'].append(optimizer.param_groups[0]["lr"])
        utils.verbose_print(' '.join([f'{k}: {v[-1]:.5f}' for k, v in history.items()]), verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1)
    return model, history, data_prob, rep_prob


def train_MaskComparableAlphaKmeans(model, dataloader, optimizer, scheduler, earlystopper, n_epochs, train_args,
                                    verbose_level=0, print_val=1):
    history = {'loss': [], 'lr': []}
    if not torch.is_tensor(dataloader.dataset.data):
        space = torch.tensor(dataloader.dataset.data, dtype=torch.float32)
    else:
        space = dataloader.dataset.data
    utils.verbose_print('Start training...', verbose_level, 0)
    data_prob = model.update_data_prob(space=space, space_mask=dataloader.dataset.mask, tau=train_args['taus'][0])
    rep_prob = None
    if train_args['compute_rep_frequency'] == 'ones':
        rep_prob = model.update_rep_prob(space=space, space_mask=dataloader.dataset.mask, tau=train_args['taus'][0])
    for epoch in range(n_epochs) if verbose_level != 1 else tqdm(range(n_epochs)):
        utils.verbose_print(f'Training step: epoch {epoch:3d}', verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1, ' ')
        running_loss = 0.0
        if train_args['compute_rep_frequency'] == 'epoch':
            rep_prob = model.update_rep_prob(space=space, space_mask=dataloader.dataset.mask,
                                             tau=train_args['taus'][epoch])
        for i, batch in enumerate(dataloader, 0):
            if train_args['compute_rep_frequency'] == 'batch':
                rep_prob = model.update_rep_prob(space=space, space_mask=dataloader.dataset.mask,
                                                 tau=train_args['taus'][epoch])
            data, mask, indices = batch
            data_prob_batch = data_prob[indices]
            optimizer.zero_grad()
            loss, dist = model.compute_loss(data=data.float(), rep_prob=rep_prob, data_prob=data_prob_batch,
                                            alpha=train_args['alphas'][epoch])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.update_rep_mask(space, alpha=train_args['alphas'][epoch])

        scheduler.step()
        earlystopper(running_loss)
        if earlystopper.early_stop:
            utils.verbose_print(f'Early stopping after {epoch} epoch.', verbose_level, 1)
            break
        history['loss'].append(running_loss)
        history['lr'].append(optimizer.param_groups[0]["lr"])
        utils.verbose_print(' '.join([f'{k}: {v[-1]:.5f}' for k, v in history.items()]), verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1)
    return model, history, data_prob, rep_prob


def train_DeepAlphaKmeans(model, dataloader, optimizer, scheduler, earlystopper, n_epochs, train_args,
                          verbose_level=0, print_val=1):
    history = {'loss': [], 'loss_ae': [], 'loss_kmeans': [], 'lr': []}
    utils.verbose_print('Start training...', verbose_level, 0)
    for epoch in range(n_epochs) if verbose_level != 1 else tqdm(range(n_epochs)):
        utils.verbose_print(f'Training step: epoch {epoch:3d}', verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1, ' ')
        running_loss = 0.0
        running_loss_ae = 0.0
        running_loss_kmeans = 0.0

        for i, batch in enumerate(dataloader, 0):
            data, mask, indices = batch
            optimizer.zero_grad()
            loss_ae, loss_kmeans, loss, _, _ = model.compute_loss(data.float(), mask, train_args['alphas'][epoch])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_ae += loss_ae.item()
            running_loss_kmeans += loss_kmeans.item()

        scheduler.step()
        earlystopper(running_loss)
        if earlystopper.early_stop:
            utils.verbose_print(f'Early stopping after {epoch} epoch.', verbose_level, 1)
            break
        history['loss'].append(running_loss)
        history['loss_ae'].append(running_loss_ae)
        history['loss_kmeans'].append(running_loss_kmeans)
        history['lr'].append(optimizer.param_groups[0]["lr"])
        utils.verbose_print(' '.join([f'{k}: {v[-1]:.5f}' for k, v in history.items()]), verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1)
    return model, history


def train_DeepMaskAlphaKmeans(model, dataloader, optimizer, scheduler, earlystopper, n_epochs, train_args,
                              verbose_level=0, print_val=1):
    history = {'loss': [], 'loss_ae': [], 'loss_kmeans': [], 'lr': []}
    if not torch.is_tensor(dataloader.dataset.data):
        space = torch.tensor(dataloader.dataset.data, dtype=torch.float32)
    else:
        space = dataloader.dataset.data

    utils.verbose_print('Start training...', verbose_level, 0)
    for epoch in range(n_epochs) if verbose_level != 1 else tqdm(range(n_epochs)):
        utils.verbose_print(f'Training step: epoch {epoch:3d}', verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1, ' ')
        running_loss = 0.0
        running_loss_ae = 0.0
        running_loss_kmeans = 0.0

        for i, batch in enumerate(dataloader, 0):
            data, mask, indices = batch
            optimizer.zero_grad()
            loss_ae, loss_kmeans, loss = model.compute_loss(data.float(), mask, train_args['alphas'][epoch])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_ae += loss_ae.item()
            running_loss_kmeans += loss_kmeans.item()

        model.update_rep_mask(space, dataloader.dataset.mask, alpha=train_args['alphas'][epoch])

        scheduler.step()
        earlystopper(running_loss)
        if earlystopper.early_stop:
            utils.verbose_print(f'Early stopping after {epoch} epoch.', verbose_level, 1)
            break
        history['loss'].append(running_loss)
        history['loss_ae'].append(running_loss_ae)
        history['loss_kmeans'].append(running_loss_kmeans)
        history['lr'].append(optimizer.param_groups[0]["lr"])
        utils.verbose_print(' '.join([f'{k}: {v[-1]:.5f}' for k, v in history.items()]), verbose_level, 1,
                            epoch % print_val == 0 or epoch == n_epochs - 1)
    return model, history


if __name__ == '__main__':
    print('ok')
