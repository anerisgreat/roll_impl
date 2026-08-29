import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from functools import partial
from torch import nn
import logging

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, KernelScheduler
from src.datasets import CreditCardFraudDataset
from src.utils import init_experiment
from src.roll import kernelized_roll_aoc, kernelized_roll_tpr, mae_loss, gce_loss, \
    focal_loss, asymmetric_loss

MAX_ITERS_ROLL = 1500
MAX_ITERS_BASE = 5000
N_EPISODES     = 5

LR_SCHEDULER      = torch.optim.lr_scheduler.StepLR
LR_SCHEDULER_ARGS = {'step_size': 50, 'gamma': 0.5}


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
    run_dir = init_experiment('results', 'creditcard')
    dataset = CreditCardFraudDataset()

    input_dim = dataset.x.shape[1]
    num_true = int(dataset.y.sum().item())
    num_false = int((1 - dataset.y).sum().item())
    imbalance_ratio = num_false / num_true
    print(f"Input dim: {input_dim}, samples: {len(dataset)}, IR: {imbalance_ratio:.1f}")

    make_net = partial(Net, input_dim)
    splitter = partial(basic_data_splitter, batch_size=8096, is_balanced=True)

    def roll_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=make_net,
            data_splitter=splitter,
            optim_class=torch.optim.RMSprop,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss_func, MAX_ITERS_ROLL,
                kernel_scheduler=KernelScheduler(32.0, decay_every=20),
                patience=100, grace_period=50),
            lr_scheduler_class=LR_SCHEDULER,
            lr_scheduler_args=LR_SCHEDULER_ARGS,
            n_episodes=N_EPISODES)

    def base_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=make_net,
            data_splitter=splitter,
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss_func, MAX_ITERS_BASE,
                patience=500, grace_period=50),
            lr_scheduler_class=LR_SCHEDULER,
            lr_scheduler_args=LR_SCHEDULER_ARGS,
            n_episodes=N_EPISODES)

    configurations = [
        roll_config('roll-aoc',    kernelized_roll_aoc()),
        roll_config('roll-tpr90',  kernelized_roll_tpr(tpr=0.9)),
        base_config('bce-weighted', nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([imbalance_ratio]))),
        base_config('gce-0.7',        gce_loss(q=0.7, pos_weight=imbalance_ratio)),
        base_config('mae',             mae_loss(pos_weight=imbalance_ratio)),
        base_config('focal-loss',      focal_loss(gamma=2.0, pos_weight=imbalance_ratio)),
        base_config('asymmetric-loss', asymmetric_loss(pos_weight=imbalance_ratio)),
    ]

    logging.info('Starting experiment!')
    run_configurations(run_dir, configurations, dataset, is_mp=False, sequential_episodes=True)
    logging.info('Script completed!')
