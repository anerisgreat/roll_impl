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

from src.roll import roll_loss_from_fpr, roll_beta_loss_from_fpr, roll_beta_aoc_loss


N_EPISODES = 7
N_EPOCHS = 1000

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.Linear(10, 1),
            # nn.ReLU(),
            # nn.Linear(3, 1)
            ]
        # layers = [
        #     x for y in [
        #         [nn.Linear(10, 10), nn.ReLU()] for _ in range(1)]
        #     for x in y] + [nn.Linear(10, 1)]
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)

if __name__ == '__main__':
    run_dir = init_experiment('results', 'forest', console_level = logging.DEBUG)
    device = torch.device('cpu')
    dataset = ForestCoverDataset()

    configurations = [
        ExperimentConfiguration(
            name = f'roll-beta-{rr:0.2f}',
            model_creator_func = MyNet,
            data_splitter = partial(basic_data_splitter, is_oneshot = False, is_balanced = True, batch_size = 2048),
            optim_class = torch.optim.SGD,
            optim_args = {'lr' : 0.01},
            criteriorator = CRBasedCriteriorator(
                roll_beta_loss_from_fpr(rr), N_EPOCHS, [rr]),
            n_episodes = N_EPISODES) \
        for rr in [0.05, 0.025]] + [
            ExperimentConfiguration(
                name = f'beta-aoc',
                model_creator_func = MyNet,
                data_splitter = partial(basic_data_splitter, is_oneshot = False, is_balanced = True, batch_size = 2048),
                optim_class = torch.optim.Adam,
                optim_args = {'lr' : 0.01},
                criteriorator = CRBasedCriteriorator(
                    roll_beta_aoc_loss, N_EPOCHS, [0.05]),
                n_episodes = N_EPISODES),
        ExperimentConfiguration(
            name = 'BCE',
            model_creator_func = MyNet,
            data_splitter = partial(basic_data_splitter, is_oneshot = False, batch_size = 2048, is_balanced = True),
            optim_class = torch.optim.Adam,
            optim_args = {'lr' : 0.01},
            criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), N_EPOCHS),
            n_episodes = N_EPISODES
        )]

    logging.info('Starting experiment!')

    run_configurations(run_dir, configurations, dataset, is_mp = True)

    logging.info('Script completed!')
