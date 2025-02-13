import torch
from functools import partial
from torch import nn
import numpy as np
from functools import partial
import torch.nn.functional as F

from torchvision import datasets, transforms
import logging
from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, CRBasedCriteriorator, \
    oneshot_datasplitter
from src.datasets import ForestCoverDataset, Cifar10Dataset, TestGaussianDataset
from src.utils import init_experiment
from src.roll import roll_loss_from_fpr

run_dir = init_experiment('results', 'forest')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.flatten(x)

device = torch.device('cpu')
dataset = Cifar10Dataset()

configurations = [
    ExperimentConfiguration(
        name = 'BCE',
        model_creator_func = Net,
        data_splitter = basic_data_splitter,
        optim_class = torch.optim.Adam,
        optim_args = {'lr' : 0.1},
        criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), 100),
        n_episodes = 5
    )] + [
            ExperimentConfiguration(
            name = f'roll-{rr:0.2f}',
            model_creator_func = Net,
            data_splitter = partial(basic_data_splitter, is_oneshot = False),
            optim_class = torch.optim.Adam,
            optim_args = {'lr' : 0.1},
            criteriorator = CRBasedCriteriorator(
                roll_loss_from_fpr(rr), 100, [rr]),
            n_episodes = 5) \
        for rr in [0.05, 0.02]
    ]

logging.info('Starting experiment!')

run_configurations(run_dir, configurations, dataset)

logging.info('Script completed!')
