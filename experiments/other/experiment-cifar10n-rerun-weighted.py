"""
Re-run gce-0.7, mae, focal-loss, asymmetric-loss on CIFAR-10N with proper
pos_weight class balancing across all 3 noise types. Writes directly into
results-final/cifar10n/, overwriting the previously unweighted results.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import logging
import torch
from functools import partial

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration
from src.datasets import Cifar10NDataset
from src.roll import mae_loss, gce_loss, focal_loss, asymmetric_loss
from src.networks import ConvNet

MAX_ITERS_BASE = 5000
N_EPISODES     = 3

LR_SCHEDULER      = torch.optim.lr_scheduler.StepLR
LR_SCHEDULER_ARGS = {'step_size': 50, 'gamma': 0.5}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')

    run_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'results-final', 'cifar10n'))
    os.makedirs(run_dir, exist_ok=True)

    net_creator = ConvNet
    splitter = partial(basic_data_splitter, batch_size=256, is_balanced=True)

    for noise_type in ['clean', 'aggre', 'worse']:
        dataset = Cifar10NDataset(noise_type=noise_type)
        n_pos = int(dataset.y.sum().item())
        n_neg = int((1 - dataset.y).sum().item())
        imbalance_ratio = n_neg / max(n_pos, 1)
        print(f'{noise_type}: {n_pos} pos / {n_neg} neg  (IR {imbalance_ratio:.1f})')

        def base_config(name, loss_func):
            return ExperimentConfiguration(
                name=f'{noise_type}-{name}',
                model_creator_func=net_creator,
                data_splitter=splitter,
                optim_class=torch.optim.Adam,
                optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
                criteriorator=BasicCriteriorator(loss_func, MAX_ITERS_BASE,
                    patience=500, grace_period=50),
                lr_scheduler_class=LR_SCHEDULER,
                lr_scheduler_args=LR_SCHEDULER_ARGS,
                n_episodes=N_EPISODES)

        configs = [
            base_config('gce-0.7',        gce_loss(q=0.7, pos_weight=imbalance_ratio)),
            base_config('mae',             mae_loss(pos_weight=imbalance_ratio)),
            base_config('focal-loss',      focal_loss(gamma=2.0, pos_weight=imbalance_ratio)),
            base_config('asymmetric-loss', asymmetric_loss(pos_weight=imbalance_ratio)),
        ]

        run_configurations(run_dir, configs, dataset,
                           is_mp=False, sequential_episodes=True)
