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
from src.datasets import KeelDataset
from src.utils import init_experiment
from src.roll import roll_loss_from_fpr, roll_beta_loss_from_fpr, roll_beta_aoc_loss, kernelized_roll_fpr
import logging

MAX_ITERS = 5000
N_EPISODES = 3

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
                nn.Linear(9, 9),
                nn.ReLU(),
                nn.Linear(9, 9),
                nn.ReLU(),
                nn.Linear(9, 9),
                nn.ReLU(),
                nn.Linear(9, 9),
                nn.ReLU(),
                nn.Linear(9, 1)
                ]
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.flatten(self._layers(x))

if __name__ == '__main__':
    run_dir = init_experiment('results', 'banana')
    dataset = KeelDataset('banana')

    configurations = [
                ExperimentConfiguration(
                name = f'roll-kernel-{rr:0.2f}',
                model_creator_func = Net,
                data_splitter = partial(basic_data_splitter, batch_size = 128, is_balanced = True),
                optim_class = torch.optim.Adam,
                optim_args = {'lr' : 1e-4},
                criteriorator = BasicCriteriorator(
                    kernelized_roll_fpr(rr), MAX_ITERS),
                    n_episodes = N_EPISODES) \
            for rr in [0.02, 0.05]
        ] + [ExperimentConfiguration(
                name = 'BCE',
                model_creator_func = Net,
                data_splitter = partial(basic_data_splitter, batch_size = 128, is_balanced = True),
                optim_class = torch.optim.Adam,
                optim_args = {'lr' : 1e-4},
                criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), MAX_ITERS),
                n_episodes = N_EPISODES
            )]

    logging.info('Starting experiment!')

    run_configurations(run_dir, configurations, dataset, is_mp = False)

    logging.info('Script completed!')
