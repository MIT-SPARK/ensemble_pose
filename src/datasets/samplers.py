import logging
import torch
from torch.utils.data import Sampler


class PartialSampler(Sampler):
    def __init__(self, ds, epoch_size, randomize=True):
        self.n_items = len(ds)
        if epoch_size < 0:
            self.epoch_size = len(ds)
        else:
            if len(ds) < epoch_size:
                logging.info("User-specified epoch size greater than dataset. Setting to original dataset length.")
            self.epoch_size = min(epoch_size, len(ds))
        self.randomize = randomize
        super().__init__(None)

    def __len__(self):
        return self.epoch_size

    def _random_iter(self):
        return (i.item() for i in torch.randperm(self.n_items)[: len(self)])

    def _sequential_iter(self):
        return (i for i in range(0, self.n_items)[: len(self)])

    def __iter__(self):
        if self.randomize:
            return self._random_iter()
        else:
            return self._sequential_iter()


class ListSampler(Sampler):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)
