from torch.utils.data import Dataset
from typing import Dict

from torch.utils.data.dataset import T_co

from utils import parse_htr_selex_dir, read_htr_selex_cycle


class HtrSelexDataset(Dataset):

    def __init__(self, cycles: Dict[int, str], padding_strategy: str):
        super().__init__()
        self.x = []
        self.y = []
        self.cycles = []
        for cycle_index, cycle_path in cycles.items():
            cur_rna = read_htr_selex_cycle(cycle_path)
            self.x += cur_rna
            self.y += [cycle_index] * len(cur_rna)
            self.cycles.append(cycle_index)
        self.cycles.sort()
            
    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]


if __name__ == '__main__':
    htr_selex_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\htr-selex'
    rna_compete_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\RNAcompete_intensities'
    res = parse_htr_selex_dir(htr_selex_path)
    dataset = HtrSelexDataset(res[0], '')
    for item in dataset:
        print(item)