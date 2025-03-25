import torch
from functools import partial
from torch import nn
import numpy as np

from torchvision import datasets, transforms
import logging

from adult import Adult

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, CRBasedCriteriorator, \
    oneshot_datasplitter
from src.datasets import ForestCoverDataset, Cifar10Dataset, TestGaussianDataset, AdultDataset
from src.utils import init_experiment
from src.roll import roll_loss_from_fpr

run_dir = init_experiment('results', 'adult')

MAX_ITERS = 10
N_EPISODES = 5

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = \
            [nn.Linear(104, 20), nn.ReLU()] + \
            [ x for y in [ \
                    [nn.Linear(20, 20), nn.ReLU()] for _ in range(1)]
                for x in y] + \
            [nn.Linear(20, 1)]
        self._layers = nn.Sequential(
            *layers)

    def forward(self, x):
        return self._layers(x).squeeze()

device = torch.device('cpu')
configurations = [
    # ExperimentConfiguration(
    #     name = 'BCE',
    #     model_creator_func = MyNet,
    #     data_splitter = basic_data_splitter,
    #     optim_class = torch.optim.Adam,
    #     optim_args = {'lr' : 0.1},
    #     criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), MAX_ITERS),
    #     n_episodes = N_EPISODES),
    ] + [ ExperimentConfiguration(
            name = f'roll-{rr:0.2f}',
            model_creator_func = MyNet,
            data_splitter = partial(basic_data_splitter, is_oneshot = True),
            optim_class = torch.optim.SGD,
            optim_args = {'lr' : 0.1},
            criteriorator = CRBasedCriteriorator(
                roll_loss_from_fpr(rr), MAX_ITERS, [rr]),
            n_episodes = N_EPISODES) \
        for rr in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]

logging.info('Starting experiment!')

run_configurations(run_dir, configurations, AdultDataset())

logging.info('Script completed!')
