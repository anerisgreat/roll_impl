import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from functools import partial
from torch import nn
import logging

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, KernelScheduler
from src.datasets import HiggsDataset
from src.utils import init_experiment
from src.roll import kernelized_roll_tpr

MAX_ITERS = 2000
N_EPISODES = 1

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 64),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return torch.flatten(self._layers(x))

if __name__ == '__main__':
    run_dir = init_experiment('results', 'higgs')
    dataset = HiggsDataset(n_samples=500_000)

    input_dim = dataset.x.shape[1]
    print(f"Input dimension: {input_dim}, samples: {len(dataset)}")

    num_true = dataset.y.sum().item()
    num_false = (1 - dataset.y).sum().item()
    imbalance_ratio = num_false / num_true

    make_net = partial(Net, input_dim)

    def roll_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=make_net,
            data_splitter=partial(basic_data_splitter, batch_size=256, is_balanced=True),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-2, 'weight_decay': 0.001},
            criteriorator=BasicCriteriorator(loss_func, MAX_ITERS,
                kernel_scheduler=KernelScheduler(126.0, decay_every=20),
                patience=100),
            lr_scheduler_class=None,
            n_episodes=N_EPISODES)

    def bce_config(name, pos_weight=None):
        loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]) if pos_weight is not None else None)
        return ExperimentConfiguration(
            name=name,
            model_creator_func=make_net,
            data_splitter=partial(basic_data_splitter, batch_size=2048, is_balanced=True),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-3},
            criteriorator=BasicCriteriorator(loss, MAX_ITERS, patience=100),
            lr_scheduler_class=None,
            n_episodes=N_EPISODES)

    configurations = [
        roll_config('roll', kernelized_roll_tpr(0.95)),
        roll_config('roll+bce', kernelized_roll_tpr(0.95, bce_weight=0.5, bce_pos_weight=imbalance_ratio)),
        bce_config('bce'),
        bce_config('bce-weighted', pos_weight=imbalance_ratio),
    ]

    logging.info('Starting experiment!')
    run_configurations(run_dir, configurations, dataset, tpr_summary=0.95)
    logging.info('Script completed!')
