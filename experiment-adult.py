import torch
from functools import partial
from torch import nn
import numpy as np

import logging

from adult import Adult

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, CRBasedCriteriorator, \
    oneshot_datasplitter
from src.datasets import AdultDataset
from src.utils import init_experiment
from src.roll import roll_loss_from_fpr, roll_beta_loss_from_fpr, roll_beta_aoc_loss, kernelized_roll_fpr

MAX_ITERS = 2000
N_EPISODES = 1 #7 #3

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = \
            [nn.Linear(104, 30), nn.ReLU()] + \
            [ x for y in [ \
                    [nn.Linear(30, 30), nn.ReLU()] for _ in range(1)]
                for x in y] + \
            [nn.Linear(30, 1), nn.Sigmoid()]
        # layers = [nn.Linear(104, 1)]
        self._layers = nn.Sequential(
            *layers)

    def forward(self, x):
        return self._layers(x).squeeze()

if __name__ == '__main__':
    run_dir = init_experiment('results', 'adult', console_level = logging.DEBUG)
    device = torch.device('cpu')
    # configurations = [ ExperimentConfiguration(
    #             name = f'beta-roll-{rr:0.2f}',
    #             model_creator_func = MyNet,
    #             data_splitter = partial(basic_data_splitter, is_oneshot = False, batch_size = 512),
    #             optim_class = torch.optim.Adam,
    #             optim_args = {'lr' : 0.01},
    #             criteriorator = CRBasedCriteriorator(
    #                 roll_beta_loss_from_fpr(rr), MAX_ITERS, [rr]),
    #             n_episodes = N_EPISODES) \
    #         for rr in [0.05, 0.1, 0.2]
        # ] + [
    configurations = [
        ExperimentConfiguration(
            name = f'GR{r}',
            model_creator_func = MyNet,
            data_splitter = partial(basic_data_splitter, is_oneshot = False, batch_size = 2048, is_balanced = True),
            optim_class = torch.optim.Adam,
            optim_args = {'lr' : 0.01},
            criteriorator = BasicCriteriorator(kernelized_roll_fpr(r), MAX_ITERS),
            n_episodes = N_EPISODES) for r in [0.2, 0.3, 0.4]] + [
            ExperimentConfiguration(
            name = 'BCE',
            model_creator_func = MyNet,
            data_splitter = partial(basic_data_splitter, batch_size = 2048, is_oneshot = False),
            optim_class = torch.optim.Adam,
            optim_args = {'lr' : 0.01, 'weight_decay' : 1e-4},
            criteriorator = BasicCriteriorator(torch.nn.BCELoss(), MAX_ITERS),
            n_episodes = N_EPISODES),
        ]

    logging.info('Starting experiment!')

    run_configurations(run_dir, configurations, AdultDataset(), is_mp = False)

    logging.info('Script completed!')
