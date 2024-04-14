import numpy as np
import torch
import random
import os.path as osp
import pathlib


class NBodyDataset():
    def __init__(self, partition='train', max_samples=1e8, dataset_name="dataset"):
        self.partition = partition
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        self.suffix += "_gravity5_initvel1small"

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        dir = pathlib.Path(__file__).parent.absolute()
        loc = np.load(osp.join(dir, 'dataset_gravity', 'loc_' + self.suffix + '.npy'))
        vel = np.load(osp.join(dir, 'dataset_gravity', 'vel_' + self.suffix + '.npy'))
        loc, vel = self.preprocess(loc, vel)
        return (loc, vel), None

    def preprocess(self, loc, vel):
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory

        return torch.Tensor(loc), torch.Tensor(vel)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()
    
    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel = self.data
        loc, vel = loc[i], vel[i]

        loc = torch.transpose(loc, 1, 2)
        vel = torch.transpose(vel, 1, 2)
        return loc, vel, loc

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


if __name__ == "__main__":
    NBodyDataset()
