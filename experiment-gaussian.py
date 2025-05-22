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
from src.roll import roll_loss_from_fpr


N_EPOCHS = 100
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
    device = torch.device('cpu')
    dataset = TestGaussianDataset(
        loc_false = (0, 0), scale_false = (0, 1),
        N_false = 10000,
        loc_true = (1, 1),
        scale_true = (1, 0),
        n_true = 10000)

    configurations = [
        ExperimentConfiguration(
            name = 'BCE',
            model_creator_func = DumbLinear,
            data_splitter = basic_data_splitter,
            optim_class = torch.optim.Adam,
            optim_args = {'lr' : 0.1},
            criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), N_EPOCHS),
            n_episodes = N_EPISODES),
        ] + [ ExperimentConfiguration(
                name = f'roll-{rr:0.2f}',
                model_creator_func = DumbLinear,
                data_splitter = partial(basic_data_splitter, is_oneshot = True),
                optim_class = torch.optim.Adam,
                optim_args = {'lr' : 0.1},
                criteriorator = CRBasedCriteriorator(
                    roll_loss_from_fpr(rr), N_EPOCHS, [rr]),
                n_episodes = N_EPISODES) \
            for rr in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        ]

    logging.info('Starting experiment!')

    run_configurations(run_dir, configurations, dataset, is_mp = True)

    logging.info('Script completed!')
