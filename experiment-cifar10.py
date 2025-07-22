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
from src.roll import roll_loss_from_fpr, roll_beta_loss_from_fpr, roll_beta_aoc_loss, kernelized_roll_fpr
import logging

MAX_ITERS = 200
N_EPISODES = 3

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
if __name__ == '__main__':
    run_dir = init_experiment('results', 'cifar10', console_level = logging.DEBUG)
    device = torch.device('cpu')
    dataset = Cifar10Dataset()

    configurations = [
                ExperimentConfiguration(
                name = f'roll-kernel-{rr:0.2f}',
                model_creator_func = Net,
                data_splitter = partial(basic_data_splitter, batch_size = 2048, is_balanced = True),
                optim_class = torch.optim.Adam,
                optim_args = {'lr' : 1e-3, 'weight_decay' : 1e-4},
                # criteriorator = CRBasedCriteriorator(
                #     kernelized_roll_fpr(rr), MAX_ITERS, [rr]),
                criteriorator = BasicCriteriorator(
                    kernelized_roll_fpr(rr), MAX_ITERS),
                    n_episodes = N_EPISODES) \
            for rr in [0.1, 0.3]
        ] + [ExperimentConfiguration(
                name = 'BCE',
                model_creator_func = Net,
                data_splitter = partial(basic_data_splitter, batch_size = 2048, is_balanced = True),
                optim_class = torch.optim.Adam,
                optim_args = {'lr' : 1e-3, 'weight_decay' : 1e-4},
                criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), MAX_ITERS),
                n_episodes = N_EPISODES
            )]

    logging.info('Starting experiment!')

    run_configurations(run_dir, configurations, dataset, is_mp = True)

    logging.info('Script completed!')
