import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Union, List
import torch
from torch.utils.data.dataset import T_co
import random
from utils import parse_htr_selex_dir, read_htr_selex_cycle, read_rna_compete_rna_list


class RnaEncoder:
    def __init__(self):
        self.rna_char2class_map = dict(zip(list('NACGT'), list(range(5))))  # 'Add N for padding and unknown'
        self.max_len = 41

    def pad(self, rna):
        if len(rna) > self.max_len:
            print(len(rna))
        if len(rna) == self.max_len:
            return rna
        pad_len = self.max_len - len(rna)
        # pad_before = pad_len // 2
        # pad_after = pad_len - pad_before
        # return 'P' * pad_before + rna + 'P' * pad_after
        return rna + 'N' * pad_len

    def encode_single_unit(self, single_unit):
        return self.rna_char2class_map[single_unit]

    def encode_rna(self, rna):
        padded = self.pad(rna)
        encoded = [self.encode_single_unit(char) for char in padded]
        return encoded


class HtrSelexDataset(Dataset):
    def __init__(self, cycles: Dict[int, str], padding_strategy: str = 'zeros'):
        super().__init__()
        self.encoder = RnaEncoder()
        self.x = []
        self.y = []
        self.cycles = []
        # self.pad_data = self.create_padding(padding_strategy)
        self.cycles = cycles.values()
        lengths = []
        for cycle_index, cycle_path in cycles.items():
            cur_cycle = read_htr_selex_cycle(cycle_path)
            # cur_label = np.zeros(self.num_cycles, dtype=np.float32)
            # cur_label[:cycle_index] = 1
            cur_label = cycle_index
            lengths.append(len(cur_cycle))
            for cur_rna in cur_cycle:
                encoded = self.encoder.encode_rna(cur_rna)
                self.x.append(encoded)
                self.y.append(cur_label)
        mean_cycle_len = int(np.mean(lengths))
        zero_lists = [[0] * self.encoder.max_len for _ in range(mean_cycle_len)]
        self.x += zero_lists
        self.y += [0] * mean_cycle_len
        self.x = torch.from_numpy(np.array(self.x, dtype=np.int64))
        self.y = torch.from_numpy(np.array(self.y, dtype=np.int64))

    # def create_padding(padding_strategy: str):
    #     if padding_strategy == 'zeros':
    #         return list('P' * 40)

    def generate_random_sequence(self):
        length = random.randint(31, self.encoder.max_len)
        sequence = ''.join(random.choices('ACGT', k=length))
        return self.encoder.encode_rna(sequence)

    def __getitem__(self, index) -> T_co:
        label = self.y[index]
        if label == 0:
            rna = self.generate_random_sequence()
            rna = torch.from_numpy(np.array(rna, dtype=np.int64))
        else:
            rna = self.x[index]
        return rna, label

    def __len__(self):
        return len(self.y)
    @property
    def cycles(self):
        return self._cycles

    @cycles.setter
    def cycles(self, value):
        self._cycles = value

    @property
    def num_cycles(self):
        return len(self.cycles)


class SimpleRnaDataset(Dataset):
    def __init__(self, rna_sequences: Union[str, List[str]]):
        if isinstance(rna_sequences, str):
            rna_sequences = read_rna_compete_rna_list(rna_sequences)
        encoder = RnaEncoder()
        self.encoded_rna = [encoder.encode_rna(rna) for rna in rna_sequences if rna.strip()]

    def __getitem__(self, index) -> T_co:
        return torch.from_numpy(np.array(self.encoded_rna[index], dtype=np.int64))

    def __len__(self):
        return len(self.encoded_rna)

    @property
    def encoded_rna(self):
        return self._encoded_rna

    @encoded_rna.setter
    def encoded_rna(self, value):
        self._encoded_rna = value



if __name__ == '__main__':
    htr_selex_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\htr-selex'
    rna_compete_path = r'C:\Users\eli.dagi\OneDrive - AU10TIX\Documents\Courses\year 2\b\Deep for biological data\Project\data\RNAcompete_intensities'
    res = parse_htr_selex_dir(htr_selex_path)
    dataset = HtrSelexDataset(res[0], '')
    for item in dataset:
        print(item)