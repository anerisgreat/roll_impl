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

MAX_ITERS = 10000
N_EPISODES = 3

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
    # configurations = [ ExperimentConfiguration(
    #             name = f'beta-roll-{rr:0.2f}',
    #             model_creator_func = MyNet,
    #             data_splitter = partial(basic_data_splitter, is_oneshot = False, batch_size = 512),
    #             optim_class = torch.optim.RMSprop,
    #             optim_args = {'lr' : 0.01},
    #             criteriorator = CRBasedCriteriorator(
    #                 roll_beta_loss_from_fpr(rr), MAX_ITERS, [rr]),
    #             n_episodes = N_EPISODES) \
    #         for rr in [0.05, 0.1, 0.2]
        # ] + [
    configurations = [
        ExperimentConfiguration(
            name = f'GR{str(optim.__name__)}',
            model_creator_func = partial(MyNet, dropout=0.2),
            data_splitter = partial(basic_data_splitter, is_oneshot = False, batch_size = 2048, is_balanced = True),
            optim_class = optim,
            optim_args = {'lr' : 0.00001, 'weight_decay' : 1e-4},
            criteriorator = BasicCriteriorator(kernelized_roll_tpr(0.95), MAX_ITERS, 5.0, kernel_scheduler = KernelScheduler(decay_every = 500)),
            n_episodes = N_EPISODES) for optim in [
                torch.optim.Adam,
            ]] + [
            ExperimentConfiguration(
            name = 'BCE',
            model_creator_func = partial(MyNet, dropout=0.2),
            data_splitter = partial(basic_data_splitter, batch_size = 2048, is_oneshot = False),
            optim_class = torch.optim.RMSprop,
            optim_args = {'lr' : 0.00001, 'weight_decay' : 1e-4},
            criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), MAX_ITERS),
            n_episodes = N_EPISODES),
        ]

    logging.info('Starting experiment!')

    run_configurations(run_dir, configurations, AdultDataset(), is_mp = False)

    logging.info('Script completed!')
