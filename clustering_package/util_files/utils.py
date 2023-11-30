import os
import glob
import json
import random
import numpy as np


def save_data(data, filename, data_type='json'):
    """Function to save files.

    :param data: data
    :param filename: path to save data.
    :param data_type: type of data (def. 'json').
    :return: None
    """
    if data_type == 'json':
        with open(filename, 'w') as f:
            json.dump(data, f)
    elif data_type == 'np':
        np.save(filename, data)


def read_data(filename, data_type='json'):
    """Function to loading data from a file.

    :param filename: path to the file.
    :param data_type: type of data (def. 'json').
    :return: data
    """
    if data_type == 'json':
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            dirpath = os.path.dirname(__file__)
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'r') as f:
                return json.load(f)
    elif data_type == 'np':
        return np.load(filename)


def save_model(model, optimizer=None, epoch=None, save_path='checkpoint'):
    """Save a torch model.

    :param model: torch model
    :param optimizer: torch optimizer
    :param epoch: number of epoch
    :param save_path: path to the file.
    :return: None
    """
    import torch
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(state, save_path)


def load_model(save_path='checkpoint'):
    """Load torch model.

    :param save_path: path to the file.
    :return: state_dict of the model, state_dict of the optimizer, number of epoch
    """
    import torch
    checkpoint = torch.load(save_path)
    return checkpoint['state_dict'], checkpoint['optimizer'], checkpoint['epoch']


def fix_seed(seed: int):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True     # For CuDNN backend
    torch.backends.cudnn.benchmark = False     # For CuDNN backend
    os.environ["PYTHONHASHSEED"] = str(seed)


def path_check(path, if_create=True):
    if path is not None and not os.path.exists(path):
        print(f'path: {path} dosn\'t exist')
        if if_create:
            os.makedirs(path)
            print(f"The directory {path} is created!")


def get_last_param_from_path(directory_path, mark, param_type=int, skip_end_part=-5, verbose=True):
    params = []
    for path_name in glob.glob(directory_path):
        try:
            param = param_type(path_name[path_name.rfind(mark) + len(mark): skip_end_part])
            params.append(param)
        except ValueError:
            if verbose:
                print(f'For path {path_name} failed to get the parameter.')
    return params


def verbose_print(print_line, current_verbose_level, allowed_verbose_level=-1,  additional_condition = True,
                  print_end='\n',):
    if (current_verbose_level > allowed_verbose_level) and additional_condition:
        print(print_line, end=print_end)
