import copy
from datasets import HtrSelexDataset, SimpleRnaDataset
from utils import parse_htr_selex_dir, parse_RNAcompete_intensities_dir,\
    create_loaders, read_cfg, get_device, dataset_to_loader, pearson_correlation
from train_model import train_model
from torch.optim.lr_scheduler import CyclicLR
from models.factory import create_factory, create_optimizer
import os
import torch


def run_full_train_step(dataset, model_name, cfg):
    device = get_device(cfg)
    factory_type = cfg['factory_type']
    orig_dataset = copy.deepcopy(dataset)
    dataset.map_labels({1: 1, 2: 1, 3: 1, 4: 1})
    dataset.add_random_samples()
    factory = create_factory(factory_type, cfg, dataset.num_classes, device)
    model = factory.create_model()
    criterion = factory.create_loss()

    loaders = create_loaders(cfg, dataset)
    optimizer = create_optimizer(cfg, model)
    schedular = CyclicLR(optimizer=optimizer, base_lr=0.00001, max_lr=0.1,
                         step_size_up=len(loaders['train']),
                         mode='triangular',
                         cycle_momentum=False)

    model_out_dir = os.path.join('results', model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    model_out_path = os.path.join(model_out_dir, f'{model_name}.pth')

    if cfg['skip_exist'] and os.path.exists(model_out_path):
        model.load_state_dict(torch.load(model_out_path, map_location=device))
    else:
        model, _ = train_model(model, dataloaders=loaders, criterion=criterion,
                               optimizer=optimizer, device=device,
                               num_epochs=cfg['num_epochs'], model_name=model_name,
                               schedular=schedular, two_step_optimizer=cfg['optimizer'] == 'sam')

        dataset = orig_dataset
        has_first_cycle = 1 in rbp_data.keys()
        has_second_cycle = 2 in rbp_data.keys()
        second_iter = dataset.num_cycles > 1 or has_first_cycle or has_second_cycle
        if second_iter:
            if dataset.num_cycles > 1 and (has_first_cycle or has_second_cycle):
                if has_first_cycle:
                    dataset.map_labels({1: 0, 2: -1, 3: 1, 4: 1})
                else:
                    dataset.map_labels({1: -1, 2: 0, 3: 1, 4: 1})
            if dataset.num_cycles == 1:
                factory = create_factory(factory_type, cfg, dataset.num_classes, device)
                model, criterion = factory.create_model(encoder=model.encoder)
            loaders = create_loaders(cfg, dataset)
            optimizer = create_optimizer(cfg, model)
            schedular = CyclicLR(optimizer=optimizer, base_lr=0.00001, max_lr=0.1, step_size_up=len(loaders['train']),
                                 mode='triangular',
                                 cycle_momentum=False)
            model, _ = train_model(model, dataloaders=loaders, criterion=criterion,
                                   optimizer=optimizer, device=device,
                                   num_epochs=cfg['num_epochs'], model_name=model_name,
                                   schedular=schedular, two_step_optimizer=cfg['optimizer'] == 'sam')
    return model, factory


if __name__ == '__main__':
    htr_selex_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\htr-selex'
    rna_compete_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\RNAcompete_sequences_rc.txt'
    rna_compete_intensities_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for ' \
                                 r'biological data\Project\data\RNAcompete_intensities'

    cfg = read_cfg('cfg.yaml')
    htr_selex_data = parse_htr_selex_dir(htr_selex_path)
    rna_compete_data = parse_RNAcompete_intensities_dir(rna_compete_intensities_path)
    rna_compete_dataset = SimpleRnaDataset(rna_compete_path)

    for i, rbp_data in enumerate(htr_selex_data):
        dataset = HtrSelexDataset(rbp_data)
        model_name = f'RBP_{i}'
        model, factory = run_full_train_step(dataset, model_name, cfg)
        predictor = factory.create_predictor(model)
        rna_compete_loader = dataset_to_loader(rna_compete_dataset, cfg, shuffle=False)
        predicted = predictor.predict_loader(rna_compete_loader)
        if i < len(rna_compete_data):
            measured = rna_compete_data[i]
            correlation = pearson_correlation(predicted, measured)
            print(correlation)


