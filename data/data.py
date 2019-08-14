import numpy as np
import torch
import random

from torch.utils.data.sampler import Sampler
import torchvision.transforms


class ReferentialDataset:
    """
    Referential Game Dataset
    """

    def __init__(self, features):
        self.features = features

    def __getitem__(self, indices):

        target_idx = indices[0]
        distractors_idxs = indices[1:]

        distractors = []
        for d_idx in distractors_idxs:
            distractors.append(self.features[d_idx])

        target_img = self.features[target_idx]

        return (target_img, distractors, indices, 0)

    def __len__(self):
        return self.features.shape[0]


class ReferentialSampler(Sampler):
    def __init__(self, data_source, k=3):
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


def split_dataset_into_dataloaders(dataset, sizes=[], batch_size=32, sampler=None):
    """
    Splits a pytorch dataset into different sizes of dataloaders
    """

    # 50 % of dataset used in train
    train_length = int(0.5 * len(dataset))
    # 10 % of dataset used in validation set
    valid_length = int(0.1 * len(dataset))
    # rest used in test set
    test_length = len(dataset) - train_length - valid_length

    if len(sizes) == 0:
        sizes = [train_length, valid_length, test_length]

    datasets = random_split(dataset, sizes)

    return (
        DataLoader(
            d,
            batch_size=batch_size if sampler is None else False,
            batch_sampler=BatchSampler(
                sampler(d), batch_size=batch_size, drop_last=False
            )
            if sampler is not None
            else False,
        )
        for d in datasets
    )
