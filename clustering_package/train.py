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

