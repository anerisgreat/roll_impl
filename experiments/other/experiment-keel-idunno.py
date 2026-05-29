import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from functools import partial
from torch import nn
import numpy as np
import torch.nn.functional as F

from torchvision import datasets, transforms
import logging
from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, CRBasedCriteriorator, \
    oneshot_datasplitter
from src.datasets import get_keel_dataset
from src.utils import init_experiment
from src.roll import roll_loss_from_fpr, roll_beta_loss_from_fpr, roll_beta_aoc_loss, kernelized_roll_fpr
import logging

MAX_ITERS = 2000
N_EPISODES = 1

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        block = lambda a, b: [
            nn.Conv2d(a, b, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ]
        layers = \
            block(3, 3) + \
            block(3, 3) + \
            block(3, 3) + [
                nn.Flatten(),
                nn.Linear(3*4, 3),
                nn.ReLU(),
                nn.Linear(3, 1)
                ]
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.flatten(self._layers(x))

if __name__ == '__main__':
    run_dir = init_experiment('results', 'cifar10', console_level = logging.DEBUG)
    device = torch.device('cpu')
    dataset = get_keel_dataset()

    configurations = [
                ExperimentConfiguration(
                name = f'roll-kernel-{rr:0.2f}',
                model_creator_func = Net,
                data_splitter = partial(basic_data_splitter, batch_size = 256, is_balanced = True),
                optim_class = torch.optim.Adam,
                optim_args = {'lr' : 1e-4},
                criteriorator = BasicCriteriorator(
                    kernelized_roll_fpr(rr), MAX_ITERS),
                    n_episodes = N_EPISODES) \
            for rr in [0.1, 0.3]
        ] + [ExperimentConfiguration(
                name = 'BCE',
                model_creator_func = Net,
                data_splitter = partial(basic_data_splitter, batch_size = 256, is_balanced = True),
                optim_class = torch.optim.Adam,
                optim_args = {'lr' : 1e-4},
                criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), MAX_ITERS),
                n_episodes = N_EPISODES
            )]

    logging.info('Starting experiment!')

    run_configurations(run_dir, configurations, dataset, is_mp = False)

    logging.info('Script completed!')
