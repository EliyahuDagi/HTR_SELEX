import os
from typing import List, Dict
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from balanced_sampler import BalancedSampler
import torch
import yaml


def parse_htr_selex_dir(root_path: str) -> List[Dict[int, str]]:
    res = dict()
    for file_path in os.listdir(root_path):
        file_name = os.path.splitext(file_path)[0] # remove extension
        seperator_index = file_name.index('_')
        rbp_index = int(file_name[3: seperator_index])
        rbp_cycle_index = int(file_name[seperator_index + 1:])
        if rbp_index not in res:
            res[rbp_index] = dict()
        res[rbp_index][rbp_cycle_index] = os.path.join(root_path, file_path)
    res = OrderedDict(sorted(res.items()))
    return list(res.values())


def parse_RNAcompete_intensities_dir(root_path: str) -> List[np.ndarray]:
    res = dict()
    for file_path in os.listdir(root_path):
        file_name = os.path.splitext(file_path)[0]  # remove extension
        rbp_index = int(file_name[3:])
        res[rbp_index] = np.loadtxt(os.path.join(root_path, file_path))
    res = list(OrderedDict(sorted(res.items())).values())
    return res


def read_htr_selex_cycle(cycle_path) -> List[str]:
    with open(cycle_path, 'r') as f:
        all_text = f.read().split('\n')
    res = [line.split(',')[0] for line in all_text]
    res = list(filter(len, res))
    return res


def create_loaders(cfg, dataset: Dataset):
    dataset_len = len(dataset)
    val_ratio = cfg['val_ratio']
    val_len = int(dataset_len * (1 - val_ratio))
    train_len = dataset_len - val_len
    train_dataset, val_datasets = random_split(dataset, [train_len, val_len])
    train_labels = train_dataset.dataset.y[train_dataset.indices]
    batch_sampler = BalancedSampler(labels=train_labels.numpy(),
                                    batch_size=cfg['batch_size'],
                                    num_class_in_batch=dataset.num_classes,
                                    seed=1)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_datasets, cfg['batch_size'] * 2, shuffle=False)
    return {'train': train_loader,
            'val': val_loader}


def read_rna_compete_rna_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        rna_sequences = f.read().split('\n')
    return rna_sequences


def get_device(cfg):
    user_device = cfg['device']
    if user_device:
        device = torch.device(user_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def read_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def dataset_to_loader(dataset: Dataset, cfg: Dict, **kwargs) -> DataLoader:
    return DataLoader(dataset=dataset, batch_size=cfg['batch_size'], **kwargs)


def pearson_correlation(a, b):
    correlation_matrix = np.corrcoef(a, b)
    return correlation_matrix[0, 1]


if __name__ == '__main__':
    htr_selex_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\htr-selex'
    rna_compete_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\RNAcompete_intensities'
    # res = parse_htr_selex_dir(htr_selex_path)
    res = parse_RNAcompete_intensities_dir(rna_compete_path)
    for rbp_item in res:
        print(rbp_item)
