import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from functools import partial
from torch import nn
import numpy as np

from torchvision import datasets, transforms
import logging
from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, CRBasedCriteriorator, \
    oneshot_datasplitter
from src.datasets import ForestCoverDataset, Cifar10Dataset, TestGaussianDataset
from src.utils import init_experiment
from src.roll import roll_loss_from_fpr, KernelizedROLLoss, kernelized_roll_fpr


N_EPOCHS = 200
# N_EPOCHS = 1
N_EPISODES = 3

class DumbLinear(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, 1)]
        self._layers = nn.Sequential(
            *layers)

    def forward(self, x):
        return self._layers(x).squeeze()

if __name__ == '__main__':
    run_dir = init_experiment('results', 'gaussian')
    dataset = TestGaussianDataset(
        loc_false = (0, 0), scale_false = (0, 1),
        n_false = 10000,
        loc_true = (1, 1),
        scale_true = (1, 0),
        n_true = 10000)

    configurations = [
        ExperimentConfiguration(
            name = f'GR{r}',
            model_creator_func = DumbLinear,
            data_splitter = partial(basic_data_splitter, is_oneshot = False, batch_size = 256, is_balanced = True),
            optim_class = torch.optim.SGD,
            optim_args = {'lr' : 0.01},
            # criteriorator = BasicCriteriorator(GaussianRollLoss(0.2, 0.1), N_EPOCHS),
            criteriorator = BasicCriteriorator(kernelized_roll_fpr(r), N_EPOCHS),
            n_episodes = N_EPISODES)
        for r in [0.1, 0.2]] + [
        ExperimentConfiguration(
            name = 'BCE',
            model_creator_func = DumbLinear,
            data_splitter = basic_data_splitter,
            optim_class = torch.optim.Adam,
            optim_args = {'lr' : 0.1},
            criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), N_EPOCHS),
            n_episodes = N_EPISODES),
        ]

    logging.info('Starting experiment!')

    run_configurations(run_dir, configurations, dataset, is_mp = True)

    logging.info('Script completed!')
