import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from functools import partial
from torch import nn
import logging

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration
from src.datasets import HomeCreditDataset
from src.utils import init_experiment
from src.roll import kernelized_roll_fpr

MAX_ITERS = 500
N_EPISODES = 3

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return torch.flatten(self._layers(x))

if __name__ == '__main__':
    run_dir = init_experiment('results', 'homecredit', console_level=logging.DEBUG)
    dataset = HomeCreditDataset()

    input_dim = dataset.x.shape[1]
    print(f"Input dimension: {input_dim}, samples: {len(dataset)}")

    def make_net():
        return Net(input_dim)

    configurations = [
        ExperimentConfiguration(
            name=f'roll-kernel-{rr:0.2f}',
            model_creator_func=make_net,
            data_splitter=partial(basic_data_splitter, batch_size=512, is_balanced=True),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-4},
            criteriorator=BasicCriteriorator(kernelized_roll_fpr(rr), MAX_ITERS),
            n_episodes=N_EPISODES)
        for rr in [0.02, 0.05]
    ] + [ExperimentConfiguration(
        name='BCE',
        model_creator_func=make_net,
        data_splitter=partial(basic_data_splitter, batch_size=512, is_balanced=True),
        optim_class=torch.optim.Adam,
        optim_args={'lr': 1e-4},
        criteriorator=BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), MAX_ITERS),
        n_episodes=N_EPISODES
    )]

    logging.info('Starting experiment!')
    run_configurations(run_dir, configurations, dataset, is_mp=False)
    logging.info('Script completed!')
