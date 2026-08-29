"""
Food-101N: baseline suite on web-crawled noisy labels.
(roll-aoc, bce-weighted, mae, gce-0.7, libauc-auroc, focal-loss, asymmetric-loss)

~310K training images, resized to 64×64 RGB, ~18.4% web noise.
Positive class: class 0 (alphabetically first) vs rest — natural IR ~100:1.

Reference: Kuang-Huai Lee et al., CVPR 2018.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch import nn
from functools import partial

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, KernelScheduler
from src.datasets import Food101NDataset
from src.utils import init_experiment
from src.roll import kernelized_roll_aoc, mae_loss, gce_loss, libauc_auc_loss, \
    focal_loss, asymmetric_loss
from src.networks import ConvNet

MAX_ITERS_ROLL = 1500
MAX_ITERS_BASE = 5000
N_EPISODES     = 3

LR_SCHEDULER      = torch.optim.lr_scheduler.StepLR
LR_SCHEDULER_ARGS = {'step_size': 50, 'gamma': 0.5}


def make_configs(net_creator, data_splitter, imbalance_ratio, n_episodes):
    imratio = 1.0 / (1.0 + imbalance_ratio)

    def roll_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=net_creator,
            data_splitter=data_splitter,
            optim_class=torch.optim.RMSprop,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss_func, MAX_ITERS_ROLL,
                kernel_scheduler=KernelScheduler(32.0, decay_every=20),
                patience=100, grace_period=50),
            lr_scheduler_class=LR_SCHEDULER,
            lr_scheduler_args=LR_SCHEDULER_ARGS,
            n_episodes=n_episodes)

    def base_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=net_creator,
            data_splitter=data_splitter,
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss_func, MAX_ITERS_BASE,
                patience=500, grace_period=50),
            lr_scheduler_class=LR_SCHEDULER,
            lr_scheduler_args=LR_SCHEDULER_ARGS,
            n_episodes=n_episodes)

    loss_fn = libauc_auc_loss(margin=1.0, imratio=imratio)
    libauc_cfg = ExperimentConfiguration(
        name='libauc-auroc',
        model_creator_func=net_creator,
        data_splitter=data_splitter,
        opt_factory=loss_fn.pesg_opt_factory(lr=1e-3),
        criteriorator=BasicCriteriorator(loss_fn, MAX_ITERS_BASE,
            patience=500, grace_period=50),
        lr_scheduler_class=LR_SCHEDULER,
        lr_scheduler_args=LR_SCHEDULER_ARGS,
        n_episodes=n_episodes)

    configs = [
        roll_config('roll-aoc', kernelized_roll_aoc()),
        base_config('bce-weighted',
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([imbalance_ratio]))),
        base_config('mae', mae_loss(pos_weight=imbalance_ratio)),
        base_config('gce-0.7', gce_loss(q=0.7, pos_weight=imbalance_ratio)),
        base_config('focal-loss', focal_loss(gamma=2.0, pos_weight=imbalance_ratio)),
        base_config('asymmetric-loss', asymmetric_loss(pos_weight=imbalance_ratio)),
        libauc_cfg,
    ]
    return configs


if __name__ == '__main__':
    run_dir = init_experiment('results', 'food101n')

    net_creator = partial(ConvNet, image_size=64)
    splitter = partial(basic_data_splitter, batch_size=256, is_balanced=True)

    dataset = Food101NDataset()
    n_pos = int(dataset.y.sum().item())
    n_neg = int((1 - dataset.y).sum().item())
    imbalance_ratio = n_neg / max(n_pos, 1)
    print(f'food101n: {n_pos} pos / {n_neg} neg  (IR {imbalance_ratio:.1f})')

    configs = make_configs(net_creator, splitter, imbalance_ratio, N_EPISODES)

    # CUDA available on beacon-wsl — all configs including libauc run on GPU
    run_configurations(run_dir, configs, dataset, is_mp=False)
