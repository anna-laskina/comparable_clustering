import os

import numpy as np
import torch

from clustering_package import constatnts
from clustering_package.util_files import utils


class EarlyStopper:
    def __init__(self, tolerance=10, delta=0.01, off=False):
        self.tolerance = tolerance
        self.min_delta = delta
        self.off = off
        self.counter = 0
        self.last_loss = 1e+10
        self.early_stop = False

    def __call__(self, curr_loss):
        if not self.off and ((self.last_loss - curr_loss) < self.min_delta):
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        self.last_loss = curr_loss


def set_lr_scheduler(optimizer, scheduler_alg='without', lr_scheduler=1.0,
                     n_epochs=None, learning_rate=None):
    if scheduler_alg == 'without':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 ** epoch)
    elif scheduler_alg == 'exp':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_scheduler ** epoch)
    elif scheduler_alg == 'cos':
        if n_epochs is None or learning_rate is None:
            print('Set the number of epoch and/or lr to define scheduler.')
            exit()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs,
                                                               eta_min=lr_scheduler * learning_rate)
    elif scheduler_alg == 'CoaAnne':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=100,
                                                                         T_mult=1,
                                                                         eta_min=lr_scheduler * learning_rate
                                                                         )
    elif scheduler_alg == 'triangular':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=learning_rate, max_lr=lr_scheduler * learning_rate,
                                                      step_size_up=30,
                                                      mode="triangular2", cycle_momentum=False)
    elif scheduler_alg == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)
    else:
        print(f'Unknown scheduler algorithm: {scheduler_alg}')
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 ** epoch)
    return scheduler


def version_verification(alg, alg_info=None, input_version=None, mode_='def', last_print=True, verbose_level=0):
    if alg in constatnts.MODEL_PARAMS.keys():
        save_path = f'{constatnts.SAVE_PATH_VER}/{constatnts.ALG_NAMES[alg]}/ver_info/'
        alg_name = alg
    elif alg in constatnts.DEF_ARG.keys():
        save_path = f'{constatnts.SAVE_PATH_VER}/vers/{alg}/'
        alg_name = alg
    else:
        save_path = constatnts.SAVE_PATH_VER
        alg_name = alg

    utils.path_check(save_path)
    work_version = input_version
    no_any_last_print = False
    if mode_ == 'create':
        if alg_info is None:
            utils.verbose_print('Please Determine alg_info parameter or choose anther mode_.', verbose_level, 0)
            exit()
        existing_versions = utils.get_last_param_from_path(directory_path=os.path.join(save_path, 'info_ver*.json'),
                                                           mark='ver', param_type=int, verbose=True)
        already_exist = False
        for possible_ver in existing_versions:
            info_ = utils.read_data(os.path.join(save_path, f'info_ver{possible_ver}.json'))
            if alg_info == info_:
                already_exist = True
                utils.verbose_print(f'This {alg_name} version already exists.', verbose_level, 0, print_end=' ')
                work_version = possible_ver
                break
        if not already_exist:
            work_version = max(existing_versions) + 1 if len(existing_versions) > 0 else 1
            utils.save_data(alg_info, os.path.join(save_path, f'info_ver{work_version}.json'))
            utils.verbose_print(f'The new {alg_name} version information has been saved.', verbose_level, 0,
                                print_end=' ')
    elif mode_ == 'read':
        if input_version is None:
            utils.verbose_print('Please determine input_version parameter or choose anther mode_.', verbose_level, 0)
            exit()
        try:
            alg_info = utils.read_data(os.path.join(save_path, f'info_ver{input_version}.json'))
            no_any_last_print = True
        except FileNotFoundError:
            utils.verbose_print(f'The {alg_name} version does not exist!', verbose_level, 0, print_end=' ')
            alg_info = None
    elif mode_ == 'save':
        if alg_info is None or input_version is None:
            utils.verbose_print('Please Determine alg_info and input_version parameters or choose anther mode_.',
                                verbose_level, 0)
            exit()

        utils.save_data(alg_info, os.path.join(save_path, f'info_ver{input_version}.json'))
        utils.verbose_print(f'The new {alg_name} version information has been saved.', verbose_level, 0, print_end=' ')
    elif mode_ == 'check':
        if alg_info is None or input_version is None:
            utils.verbose_print('Please Determine alg_info and input_version parameters or choose anther mode_.',
                                verbose_level, 0)
            exit()
        try:
            info_ = utils.read_data(os.path.join(save_path, f'info_ver{input_version}.json'))
            if alg_info != info_:
                utils.verbose_print(f'The {alg_name} version is incorrect!', verbose_level, 0)
                work_version, _ = version_verification(alg=alg, alg_info=alg_info, input_version=input_version,
                                                       mode_='create', last_print=False)
        except FileNotFoundError:
            utils.verbose_print(f'The {alg_name} version does not exist!', verbose_level, 0)
            work_version, _ = version_verification(alg=alg, alg_info=alg_info, input_version=input_version,
                                                   mode_='create', last_print=False)
    elif mode_ == 'try':
        work_version = -1
    else:
        utils.verbose_print(f'Unknown mode in version_verification function! The mode \'{mode_}\' was given, '
                            f'acceptable are: \'create\',\'save\', \'read\', \'check\', \'try\'.', verbose_level, 0)
        work_version = 0

    if no_any_last_print:
        return work_version, alg_info
    if last_print and input_version != work_version:
        utils.verbose_print(f'The {alg_name} version has been changed to {work_version}', verbose_level, 0)
    else:
        print()
    return work_version, alg_info


def check_arg_conditions(alg, alg_args, verbose_level=1):
    if alg == 'dak':
        if alg_args['pre_train'] is False and alg_args['ae_ver'] is not None:
            utils.verbose_print(f'For DAK without pre_train \'ae_ver\' should be None.', verbose_level, 0)
            alg_args['ae_ver'] = None
    if alg == 'assign':
        if alg_args['assignment_type'] == 'hard' and alg_args['alpha'] is not None:
            utils.verbose_print(f'For assign args with hard assignment type \'alpha\' should be None.',
                                verbose_level, 0)
            alg_args['alpha'] = None
        if alg_args['considering_cluster_type'] is False and alg_args['gamma'] is not None:
            utils.verbose_print(f'For assign args without considering_cluster_type \'gamma\' should be None.',
                                verbose_level, 0)
            alg_args['gamma'] = None
    return alg_args


def generate_version_info(alg, **kwargs):
    verbose_level = kwargs.get('verbose', 1)
    if alg not in constatnts.MODEL_PARAMS.keys():
        print(f'Unknown clustering algorithm. The algorithm \'{alg}\' was given, '
              f'acceptable are: {", ".join(list(constatnts.MODEL_PARAMS.keys()))}.')
        return {}
    alg_args = {alg_param: kwargs.get(alg_param, constatnts.MODEL_PARAMS[alg][alg_param])
                for alg_param in constatnts.MODEL_PARAMS[alg].keys()}
    alg_args = check_arg_conditions(alg, alg_args, verbose_level=verbose_level)
    return alg_args


def generate_arg(arg_type, **kwargs):
    verbose_level = kwargs.get('verbose', 1)
    if arg_type not in constatnts.DEF_ARG.keys():
        print(f'Unknown type of arguments. The type \'{arg_type}\' was given, '
              f'acceptable are: {", ".join(list(constatnts.DEF_ARG.keys()))}.')
        return {}
    alg_args = {alg_param: kwargs.get(alg_param, constatnts.DEF_ARG[arg_type][alg_param])
                for alg_param in constatnts.DEF_ARG[arg_type].keys()}
    alg_args = check_arg_conditions(arg_type, alg_args, verbose_level=verbose_level)
    return alg_args


def create_annealing_array(n_size, annealing_start_val=1.0, annealing_max_val=10.0, base=2.0):
    if n_size > 1 and annealing_max_val != annealing_start_val:
        annealing_array = np.zeros(n_size, dtype=float)
        annealing_array[0] = 1
        for i in range(1, n_size):
            annealing_array[i] = (base ** (1 / (np.log(i + 1)) ** 2)) * annealing_array[i - 1]
        annealing_array = (annealing_array - 1) * ((annealing_max_val - annealing_start_val) / (
                    annealing_array[n_size - 1] - annealing_array[0])) + annealing_start_val
    else:
        annealing_array = [annealing_start_val] * n_size
    return annealing_array


def load_pretrain_(use_pre_train, dataset_name, ae_ver, seed):
    if use_pre_train:
        state_dict, _, _ = utils.load_model(
            os.path.join(constatnts.SAVE_ROOT_PATH,
                         f'models/AE/model/dataset_{dataset_name}/model_{dataset_name}_ae{ae_ver}_s{seed}.pt'))

        pt_h = utils.read_data(
            os.path.join(constatnts.SAVE_ROOT_PATH,
                         f'emb/AE/dataset_{dataset_name}/embedding_ae{ae_ver}_s{seed}.npy'), data_type='np')
        space_info = pt_h
    else:
        state_dict = None
        space_info = None
    return state_dict, space_info


if __name__ == '__main__':
    print('ok')
