import numpy as np
import torch
import random
import os

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.utils.data.dataset import random_split
import torchvision.transforms

from .generate_dataset import generate_dataset

dir_path = os.path.dirname(os.path.realpath(__file__))


class ReferentialDataset:
    """
    Referential Game Dataset
    """

    def __init__(self, data):
        self.data = data.astype(np.float32)

    def __getitem__(self, indices):

        target_idx = indices[0]
        distractors_idxs = indices[1:]

        distractors = []
        for d_idx in distractors_idxs:
            distractors.append(self.data[d_idx])

        target_img = self.data[target_idx]

        return (target_img, distractors)

    def __len__(self):
        return self.data.shape[0]


class ReferentialSampler(Sampler):
    def __init__(self, data_source, k=3):
        self.data_source = data_source
        self.n = len(data_source)
        self.k = k
        assert self.k < self.n

    def __iter__(self):
        indices = []
        for t in range(self.n):
            # target in first position with k random distractors following
            indices.append(
                np.array(
                    [t]
                    + random.sample(
                        list(range(t - 1)) + list(range(t, self.n)), self.k
                    ),
                    dtype=int,
                )
            )
        return iter(indices)

    def __len__(self):
        return self.n


def get_referential_dataloader(file_name: str, batch_size: int = 32):
    """
    Splits a pytorch dataset into different sizes of dataloaders
    """
    # load if already exists
    file_path = dir_path + "/" + file_name

    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        data = generate_dataset()
        # save locally
        np.save(file_path, data)

    dataset = ReferentialDataset(data)
    return DataLoader(
        dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ReferentialSampler(dataset), batch_size=batch_size, drop_last=False
        ),
    )
