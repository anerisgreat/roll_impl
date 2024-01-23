import scipy
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
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
