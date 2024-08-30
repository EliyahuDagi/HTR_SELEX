import random
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co


class BalancedSampler(Sampler):
    """
    Sample batch with is balanced in amount of each class inside the batch
    """
    def __init__(self, labels: np.ndarray, batch_size, num_class_in_batch, seed):
        super().__init__(None)
        self.labels = labels
        self.batch_size = batch_size
        self.num_class_in_bacth = num_class_in_batch
        self.per_class_in_batch = batch_size // num_class_in_batch
        self.left_overs = batch_size % num_class_in_batch
        self.groups = np.unique(self.labels).tolist()
        self.group_curser = 0
        self.per_group_samples_indices = dict()
        self.per_group_curser = dict()
        for label in self.groups:
            group_indices = np.where(self.labels == label)[0].tolist()
            self.per_group_samples_indices[label] = group_indices
            self.restart_group(label)
        random.seed(seed)

    def __iter__(self) -> Iterator[T_co]:
        for batch_index in range(self.__len__()):
            indices = self.generate_batch()
            yield indices

    def __len__(self):
        return self.labels.shape[0] // self.batch_size

    def shuffle_groups(self):
        random.shuffle(self.groups)

    def shuffle_group(self, group_name):
        random.shuffle(self.per_group_samples_indices[group_name])

    def generate_batch(self):
        indices = []
        for _ in range(self.num_class_in_bacth):
            class_name = self.generate_class()
            indices += self.generate_class_samples(class_name)
        for _ in range(self.left_overs):
            class_name = self.generate_class(promote=False)
            indices.append(self.generate_class_sample(class_name))

        return indices

    def generate_class(self, promote=True):
        cur_class = self.groups[self.group_curser]
        if promote:
            self.promote_groups()
        return cur_class

    def promote_groups(self):
        self.group_curser += 1
        if self.group_curser == len(self.groups):
            self.restart_groups()

    def restart_groups(self):
        random.shuffle(self.groups)
        self.group_curser = 0

    def generate_class_samples(self, class_name):
        return [self.generate_class_sample(class_name) for _ in range(self.per_class_in_batch)]

    def generate_class_sample(self, class_name):
        indices = self.per_group_samples_indices[class_name]
        cur_index = indices[self.per_group_curser[class_name]]
        self.promote_group_curser(class_name)
        return cur_index

    def promote_group_curser(self, group_name):
        cur_len = len(self.per_group_samples_indices[group_name])
        self.per_group_curser[group_name] += 1
        if self.per_group_curser[group_name] == cur_len:
            self.restart_group(group_name)

    def restart_group(self, group_name):
        self.per_group_curser[group_name] = 0
        self.shuffle_group(group_name)
