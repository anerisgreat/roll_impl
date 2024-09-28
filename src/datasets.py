import scipy
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer

import torch
import torchvision
import torchvision.transforms as transforms

class ForestCoverDataset:
    def __init__(self):
        PATH = '/home/aner/.data/forestcover/cover.mat'

        mat = scipy.io.loadmat(PATH)

        x = mat['X']
        y = mat['y']

        self.x = \
            torch.tensor(PowerTransformer().fit_transform(x),
                         dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)

class Cifar10Dataset:
    def __init__(self):
        PATH = '/home/aner/.data/cifar10'
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self._dset = torchvision.datasets.CIFAR10(
            root=PATH, train=True, download=True, transform=transform)
        self.y = torch.tensor(torch.tensor(self._dset.targets) == 1, dtype = torch.float64)

    def __getitem__(self, i):
        x, _ = list(zip(*[self._dset[j] for j in i])) if len(i.shape) > 0 \
            else self._dset[i]
        print(x)
        return torch.stack(x, 0), self.y[i]

    def __len__(self):
        return len(self._dset)
