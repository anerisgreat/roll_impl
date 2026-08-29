"""
Add focal-loss and asymmetric-loss baselines to the CIFAR-10N run.
Runs into the existing results dir alongside the original 5 configs.
Results are combined via scripts/regen_cifar10n_summaries.py after the run.

Usage:
    python experiments/other/experiment-cifar10n-focal-asym.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch import nn
from functools import partial

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration
from src.datasets import Cifar10NDataset
from src.utils import logging_get_default_config
from src.roll import focal_loss, asymmetric_loss
from src.networks import ConvNet
import logging
import logging.config as loggingconfig

MAX_ITERS = 5000
N_EPISODES = 3

RUN_DIR = 'results/cifar10n/2026-07-19-06-52'


def make_configs(net_creator, data_splitter, noise_type, imbalance_ratio, n_episodes):
    def base_config(name, loss_func):
        return ExperimentConfiguration(
            name=f'{noise_type}-{name}',
            model_creator_func=net_creator,
            data_splitter=data_splitter,
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss_func, MAX_ITERS,
                patience=500, grace_period=50),
            lr_scheduler_class=None,
            n_episodes=n_episodes)

    return [
        base_config('focal-loss', focal_loss(gamma=2.0, pos_weight=imbalance_ratio)),
        base_config('asymmetric-loss', asymmetric_loss(pos_weight=imbalance_ratio)),
    ]


if __name__ == '__main__':
    loggingconfig.dictConfig(logging_get_default_config(
        debug_fname=os.path.join(RUN_DIR, 'debug-focal-asym.log')))

    net_creator = ConvNet
    splitter = partial(basic_data_splitter, batch_size=256, is_balanced=True)

    for noise_type in ['clean', 'aggre', 'worse']:
        dataset = Cifar10NDataset(noise_type=noise_type)
        n_pos = int(dataset.y.sum().item())
        n_neg = int((1 - dataset.y).sum().item())
        imbalance_ratio = n_neg / max(n_pos, 1)
        logging.info(f'{noise_type}: {n_pos} pos / {n_neg} neg  (IR {imbalance_ratio:.1f})')

        configs = make_configs(net_creator, splitter, noise_type, imbalance_ratio, N_EPISODES)
        run_configurations(RUN_DIR, configs, dataset, is_mp=False)

    logging.info('focal + asymmetric runs complete')
