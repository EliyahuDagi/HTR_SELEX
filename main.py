import os.path
from typing import List, Dict
import sys
from datasets import SimpleRnaDataset
from utils import htr_selex_info, read_cfg
from datasets import HtrSelexDataset
from train_all import run_full_train, run_predict


def cycles_dict(htr_selex_cycles: List[str]) -> Dict[int, str]:
    """
    :param htr_selex_cycles: list of htr selex paths assume RBP<Index>_<cycle>
    :return: dict with cycle index and path
    """
    selex_cycles_dict = dict()
    for cycle_path in htr_selex_cycles:
        cycle_file_name = os.path.basename(cycle_path)
        rbp_index, rbp_cycle_index = htr_selex_info(cycle_file_name)
        selex_cycles_dict[rbp_cycle_index] = cycle_path
    return selex_cycles_dict


def train_single(out_path: str, rna_compete_path: str, htr_selex_cycles: List[str], cfg):
    rna_compete_dataset = SimpleRnaDataset(rna_compete_path)
    rbp_index = 0
    selex_cycles_dict = cycles_dict(htr_selex_cycles)
    dataset = HtrSelexDataset(selex_cycles_dict)
    model_name = f'RBP_{rbp_index}'
    model_out_dir = os.path.dirname(out_path)
    predictor = run_full_train(dataset, model_name, cfg, model_out_dir)
    run_predict(out_path, predictor, cfg, rna_compete_dataset)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python main.py <ofile> <RNCMPT> <SELEX1> <SELEX2> … <SELEX4>")
        sys.exit(1)
    out_path = sys.argv[1]
    rna_compete_path = sys.argv[2]
    selex_cycles = sys.argv[3:]
    cfg = read_cfg('cfg.yaml')
    train_single(out_path=out_path, rna_compete_path=rna_compete_path,
                 htr_selex_cycles=selex_cycles, cfg=cfg)



