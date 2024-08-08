import os
from typing import List, Dict
from collections import OrderedDict


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


def parse_RNAcompete_intensities_dir(root_path: str) -> List[str]:
    res = dict()
    for file_path in os.listdir(root_path):
        file_name = os.path.splitext(file_path)[0] # remove extension
        rbp_index = int(file_name[3:])
        res[rbp_index] = file_name# os.path.join(root_path, file_path)
    res = list(OrderedDict(sorted(res.items())).values())
    return res


def read_htr_selex_cycle(cycle_path) -> List[str]:
    with open(cycle_path, 'r') as f:
        all_text = f.read().split('\n')
    res = [line.split(',')[0] for line in all_text]
    res = list(filter(len, res))
    return res


if __name__ == '__main__':
    htr_selex_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\htr-selex'
    rna_compete_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\RNAcompete_intensities'
    # res = parse_htr_selex_dir(htr_selex_path)
    res = parse_RNAcompete_intensities_dir(rna_compete_path)
    for rbp_item in res:
        print(rbp_item)