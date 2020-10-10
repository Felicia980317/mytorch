from random import shuffle

import numpy as np
import torch


class DataLoader(object):
    def __init__(self, dataset, batch_size, use_shuffle=True):
        super(DataLoader, self).__init__()

        self.dataset = dataset
        self.use_shuffle = use_shuffle
        self.batch_size = batch_size

        self.n_samples = len(self.dataset)
        self.iter_per_epoch = self.n_samples // self.batch_size

        self.ep = 0
        self.i = 0
        self.random_pool = [_ for _ in range(self.n_samples)]
        if use_shuffle:
            shuffle(self.random_pool)

    def __iter__(self):
        return self

    def __next__(self):
        img_batch, label_batch = [], []
        if self.i + self.batch_size > self.n_samples:
            self.ep += 1
            self.i = 0
            shuffle(self.random_pool)

        data = []
        for i in range(self.i, self.i + self.batch_size, 1):
            idx = self.random_pool[i]
            _data = self.dataset[idx]
            data.append(_data)
        self.i += self.batch_size

        out = []
        for key in data[0].keys():
            out.append([_data[key] for _data in data])
        return (torch.stack(_out) for _out in out)

    def __len__(self):
        return self.iter_per_epoch


if __name__ == "__main__":
    pass
