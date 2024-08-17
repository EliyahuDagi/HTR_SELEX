from datasets import HtrSelexDataset, SimpleRnaDataset
from utils import parse_htr_selex_dir, parse_RNAcompete_intensities_dir,\
    read_rna_compete_rna_list,\
    create_model, create_loaders, read_cfg, \
    create_optimizer, get_device, create_predictor, dataset_to_loader, pearson_correlation
from train_model import train_model
import os
import torch


if __name__ == '__main__':
    htr_selex_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\htr-selex'
    rna_compete_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\RNAcompete_sequences_rc.txt'
    rna_compete_intensities_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for ' \
                                 r'biological data\Project\data\RNAcompete_intensities'

    cfg = read_cfg('cfg.yaml')
    htr_selex_data = parse_htr_selex_dir(htr_selex_path)
    rna_compete_data = parse_RNAcompete_intensities_dir(rna_compete_intensities_path)
    rna_compete_dataset = SimpleRnaDataset(rna_compete_path)
    device = get_device(cfg)

    for i, rbp_data in enumerate(htr_selex_data):
        model_name = f'RBP_{i}'
        dataset = HtrSelexDataset(rbp_data)
        model, criterion = create_model(cfg, dataset, device)
        loaders = create_loaders(cfg, dataset)
        optimizer = create_optimizer(cfg, model)

        model_out_path = os.path.join('models', model_name, f'{model_name}.pth')
        if cfg['skip_exist'] and os.path.exists(model_out_path):
            model.load_state_dict(torch.load(model_out_path, map_location=device))
        else:
            model = train_model(model, dataloaders=loaders, criterion=criterion,
                                optimizer=optimizer, device=device, num_epochs=10, model_name=model_name)

        predictor = create_predictor(dataset.num_cycles, model, loaders)
        rna_compete_loader = dataset_to_loader(rna_compete_dataset, cfg, shuffle=False)
        predicted = predictor.predict_loader(rna_compete_loader, device)
        measured = rna_compete_data[i]
        correlation = pearson_correlation(predicted, measured)
        print(correlation)

