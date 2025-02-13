import scipy
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer

#for Adult
from adult import Adult

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
        self._transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self._dset = torchvision.datasets.CIFAR10(
            root=PATH, train=True, download=True, transform=self._transform)
        self.y = torch.tensor(torch.tensor(self._dset.targets) == 1, dtype = torch.float64)

    def __getitem__(self, i):
        x = self._dset.data[i]
        x = torch.tensor(x / 255, dtype = torch.float32)
        x = torch.movedim(x, -1, -3)
        return x, self.y[i]


        # if(type(i) == int):
        #     return self._dset.data[i], self.y[i]
        # if(len(i.shape) > 0):
        #     x, __= list(zip(*[self._dset.data[j] for j in i]))
        #     return torch.stack(x, 0), self.y[i]
        # else:
        #     return self._dset.data[i], self.y[i]


    def __len__(self):
        return len(self._dset)

class TestGaussianDataset:
    def __init__(self,
            loc_false, scale_false, n_false,
            loc_true, scale_true, n_true):
        false_s = np.random.normal(loc_false, scale_false,
                                   np.array([n_false, len(loc_false)]))
        true_s = np.random.normal(loc_true, scale_true,
                                  np.array([n_true, len(loc_true)]))

        samps = np.concatenate((false_s, true_s))

        labels = np.concatenate((
            np.zeros(n_false, dtype = bool),
            np.ones(n_true, dtype = bool)))

        self.x = torch.tensor(samps, dtype = torch.float32)
        self.y = torch.tensor(labels, dtype = torch.float32)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)

class AdultDataset:
    def __init__(self):
        self._data = Adult(root = 'datasets', download = True)
        self.x, self.y = self._data[:]
        self.y = self.y.float()

    def __getitem__(self, i):
        retx, rety = self._data[i]
        return retx, rety.float()

    def __len__(self, i):
        return len(self._data)
