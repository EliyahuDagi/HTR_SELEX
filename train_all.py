import copy

import numpy as np

from datasets import HtrSelexDataset, SimpleRnaDataset
from utils import parse_htr_selex_dir, parse_RNAcompete_intensities_dir,\
    create_loaders, read_cfg, get_device, dataset_to_loader, pearson_correlation
from train_model import train_model
from torch.optim.lr_scheduler import CyclicLR
from models.factory import create_factory, create_optimizer
import os
import torch


def run_full_train(dataset, model_name, cfg, model_out_dir):
    device = get_device(cfg)
    factory_type = cfg['factory_type']
    orig_dataset = copy.deepcopy(dataset)
    # First iteration map all cycles to class 1
    # and add random sequences as class 0

    # map all cycles to class 1
    dataset.map_labels({1: 1, 2: 1, 3: 1, 4: 1})
    # add random sequences as class 0
    dataset.add_random_samples()
    # factory responible to create Model, loss, and predictor by config
    factory = create_factory(factory_type, cfg, dataset.num_classes, device)
    model = factory.create_model()
    criterion = factory.create_loss()

    loaders = create_loaders(cfg, dataset)
    optimizer = create_optimizer(cfg, model)
    schedular = CyclicLR(optimizer=optimizer, base_lr=0.00001, max_lr=0.1,
                         step_size_up=len(loaders['train']),
                         mode='triangular',
                         cycle_momentum=False)

    os.makedirs(model_out_dir, exist_ok=True)
    model_out_path = os.path.join(model_out_dir, f'{model_name}.pth')
    max_no_progress = -1 if dataset.num_cycles == 1 else 1
    if cfg['skip_exist'] and os.path.exists(model_out_path):
        model.load_state_dict(torch.load(model_out_path, map_location=device), strict=False)
    else:
        model, _ = train_model(model, dataloaders=loaders, criterion=criterion,
                               optimizer=optimizer, device=device,
                               num_epochs=cfg['num_epochs'], model_name=model_name,
                               schedular=schedular, two_step_optimizer=cfg['optimizer'] == 'sam',
                               max_no_progress=max_no_progress, train_dir=model_out_dir)
        # Second iteration - distinguish low cycles from high cycles
        dataset = orig_dataset
        has_first_cycle = 1 in dataset.cycles
        has_second_cycle = 2 in dataset.cycles
        second_iter = dataset.num_cycles > 1 and (has_first_cycle or has_second_cycle)
        if second_iter:
            # distinguish between low cycles and high cycles
            if has_first_cycle:
                dataset.map_labels({1: 0, 2: -1, 3: 1, 4: 1})
            else:
                dataset.map_labels({1: -1, 2: 0, 3: 1, 4: 1})
            loaders = create_loaders(cfg, dataset)
            optimizer = create_optimizer(cfg, model)
            schedular = CyclicLR(optimizer=optimizer, base_lr=0.00001, max_lr=0.1, step_size_up=len(loaders['train']),
                                 mode='triangular',
                                 cycle_momentum=False)
            model, _ = train_model(model, dataloaders=loaders, criterion=criterion,
                                   optimizer=optimizer, device=device,
                                   num_epochs=cfg['num_epochs'], model_name=model_name,
                                   schedular=schedular, two_step_optimizer=cfg['optimizer'] == 'sam',
                                   max_no_progress=1, train_dir=model_out_dir)
    predictor = factory.create_predictor(model)
    return predictor


def run_predict(out_path, predictor, cfg, rna_compete_dataset):
    rna_compete_loader = dataset_to_loader(rna_compete_dataset, cfg, shuffle=False)
    predicted = predictor.predict_loader(rna_compete_loader)
    with open(out_path, 'w') as f:
        f.write('\n'.join(map(str, predicted.tolist())))
    return predicted


if __name__ == '__main__':
    htr_selex_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\htr-selex'
    rna_compete_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\RNAcompete_sequences_rc.txt'
    rna_compete_intensities_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for ' \
                                 r'biological data\Project\data\RNAcompete_intensities'

    cfg = read_cfg('cfg.yaml')
    htr_selex_data = parse_htr_selex_dir(htr_selex_path)
    rna_compete_data = parse_RNAcompete_intensities_dir(rna_compete_intensities_path)
    rna_compete_dataset = SimpleRnaDataset(rna_compete_path)
    correlations = []
    for i, rbp_data in enumerate(htr_selex_data):
        dataset = HtrSelexDataset(rbp_data)
        model_name = f'RBP_{i}'
        model_out_dir = os.path.join('results', model_name)
        predictor = run_full_train(dataset, model_name, cfg, model_out_dir)
        out_path = os.path.join(model_out_dir, model_name + '.txt')
        predicted = run_predict(out_path, predictor, cfg, rna_compete_dataset)
        if i < len(rna_compete_data):
            measured = rna_compete_data[i]
            correlation = pearson_correlation(predicted, measured)
            print(correlation)
            correlations.append(correlation)
    print(f'mean correlation {np.mean(correlations)}')

