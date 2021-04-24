import numpy as np
import torch


class MultiviewTensorDataset:
    def __init__(self, Xs, y, batch_size=1, stratified=False, shuffle=False):
        self.Xs = Xs
        self.y = y
        self.batch_size = batch_size
        self.stratified = stratified
        self.shuffle = shuffle
        self.indices = np.arange(len(self.y))
        self.iter_id = 0
        self.reset()

    def __getitem__(self, index):
        return_indices = self.indices[self.iter_id * self.batch_size:(self.iter_id + 1) * self.batch_size]
        Xs = [X[return_indices] for X in self.Xs]
        y = self.y[return_indices]
        return Xs, y

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.iter_id < len(self):
            return_indices = self.indices[self.iter_id * self.batch_size:(self.iter_id + 1) * self.batch_size]
            self.iter_id += 1
            Xs = [X[return_indices] for X in self.Xs]
            y = self.y[return_indices]
            return Xs, y
        else:
            raise StopIteration

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def reset(self):
        self.indices = np.arange(len(self.y))
        if self.shuffle:
            self.indices = np.random.permutation(self.indices)
        if self.stratified:
            y_count = np.bincount(self.y)
            y_per_batch = y_count.astype(np.float32) / len(self)
            if np.any(y_per_batch < 1):
                raise ValueError("Some classes cannot be distributed to all batches.")
            y_mask = [torch.where(self.y[self.indices].eq(i))[0].tolist() for i in self.y.unique()]
            stratified_indices = []
            y_per_batch_ceil = np.ceil(y_per_batch).astype(np.int)
            for batch_id in range(len(self)):
                for ci in range(len(y_mask)):
                    start_ind = round(batch_id * y_per_batch_ceil[ci])
                    end_ind = round((batch_id + 1) * y_per_batch_ceil[ci])
                    stratified_indices.extend(y_mask[ci][start_ind:end_ind])
            self.indices = self.indices[stratified_indices]
        self.iter_id = 0
