import torch
from functools import partial
from torch import nn
import numpy as np
from functools import partial
import logging

import logging
from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, CRBasedCriteriorator, \
    oneshot_datasplitter
from src.datasets import ForestCoverDataset
from src.utils import init_experiment

from src.roll import roll_loss_from_fpr, roll_beta_loss_from_fpr

run_dir = init_experiment('results', 'forest', console_level = logging.DEBUG)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            x for y in [
                [nn.Linear(10, 10), nn.ReLU()] for _ in range(0)]
            for x in y] + [nn.Sigmoid()]
        self._layers = nn.Sequential(
            *layers, nn.Linear(10, 1))

    def forward(self, x):
        return self._layers(x)

device = torch.device('cpu')
dataset = ForestCoverDataset()

configurations = [
    # ExperimentConfiguration(
    #     name = 'BCE',
    #     model_creator_func = MyNet,
    #     data_splitter = basic_data_splitter,
    #     optim_class = torch.optim.SGD,
    #     optim_args = {'lr' : 0.1},
    #     criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), 100),
    #     n_episodes = 5
    # )] + [
            ExperimentConfiguration(
            name = f'roll-beta-{rr:0.2f}',
            model_creator_func = MyNet,
            data_splitter = partial(basic_data_splitter, is_oneshot = True),
            optim_class = torch.optim.SGD,
            optim_args = {'lr' : 0.1},
            criteriorator = CRBasedCriteriorator(
                roll_beta_loss_from_fpr(rr), 100, [rr]),
            n_episodes = 5) \
        for rr in [0.05, 0.025]
    ]

logging.info('Starting experiment!')

run_configurations(run_dir, configurations, dataset)

logging.info('Script completed!')
