import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from functools import partial
from torch import nn
import numpy as np

import logging

from adult import Adult

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, CRBasedCriteriorator, \
    oneshot_datasplitter, KernelScheduler
from src.datasets import AdultDataset
from src.utils import init_experiment
from src.roll import roll_loss_from_fpr, roll_beta_loss_from_fpr, roll_beta_aoc_loss, kernelized_roll_fpr, kernelized_roll_tpr

MAX_ITERS = 5000
N_EPISODES = 1

class MyNet(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        layers = \
            [nn.Linear(104, 15), nn.ReLU(), nn.Dropout(dropout)] + \
            [ x for y in [ \
                    [nn.Linear(15, 15), nn.ReLU(), nn.Dropout(dropout)] for _ in range(2)]
                for x in y] + \
            [nn.Linear(15, 1)]
        # layers = [nn.Linear(104, 1)]
        self._layers = nn.Sequential(
            *layers)

    def forward(self, x):
        return self._layers(x).squeeze()

if __name__ == '__main__':
    run_dir = init_experiment('results', 'adult')
    dataset = AdultDataset()

    num_true = dataset.y.sum().item()
    num_false = (1 - dataset.y).sum().item()
    imbalance_weight = torch.tensor([num_false / num_true])

    def roll_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=partial(MyNet, dropout=0.2),
            data_splitter=partial(basic_data_splitter, is_oneshot=False, batch_size=2048, is_balanced=True),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-4},
            criteriorator=BasicCriteriorator(loss_func, MAX_ITERS, 5.0,
                kernel_scheduler=KernelScheduler(initial_gamma=100, decay_every=100)),
            n_episodes=N_EPISODES)

    def bce_config(name, pos_weight=None):
        loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.clone() if pos_weight is not None else None)
        return ExperimentConfiguration(
            name=name,
            model_creator_func=partial(MyNet, dropout=0.2),
            data_splitter=partial(basic_data_splitter, batch_size=2048, is_oneshot=False, is_balanced=True),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-4},
            criteriorator=BasicCriteriorator(loss, MAX_ITERS),
            n_episodes=N_EPISODES)

    configurations = [
        roll_config('roll+bce-weighted',
            kernelized_roll_tpr(0.95, bce_weight=0.5, bce_pos_weight=num_false / num_true)),
    ]

    logging.info('Starting experiment!')
    run_configurations(run_dir, configurations, dataset)
    logging.info('Script completed!')
