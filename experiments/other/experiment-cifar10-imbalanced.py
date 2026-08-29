"""
CIFAR-10 with controlled imbalance ratios, using the same baseline suite as
the KEEL poisoned experiments: roll-aoc, bce-weighted, mae, gce-0.7, libauc-auroc.

IR 9 = natural binary (automobile vs rest, no subsampling).
IR 50/100/200 match long-tail regimes from LDAM-DRW / MiSLAS papers.

Positive class: class 1 (automobile), ~5000 samples in CIFAR-10 train.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch import nn
from functools import partial

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, KernelScheduler
from src.datasets import ImbalancedCifar10Dataset
from src.utils import init_experiment
from src.roll import kernelized_roll_aoc, mae_loss, gce_loss, libauc_auc_loss
from src.networks import ConvNet

MAX_ITERS_ROLL = 1500
MAX_ITERS_BASE = 5000
N_EPISODES = 3


def make_configs(net_creator, data_splitter, imbalance_ratio, n_episodes):
    imratio = 1.0 / (1.0 + imbalance_ratio)  # fraction of positives

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
            lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_args={'step_size': 50, 'gamma': 0.5},
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
            lr_scheduler_class=None,
            n_episodes=n_episodes)

    loss_fn = libauc_auc_loss(margin=1.0, imratio=imratio)
    libauc_cfg = ExperimentConfiguration(
        name='libauc-auroc',
        model_creator_func=net_creator,
        data_splitter=data_splitter,
        opt_factory=loss_fn.pesg_opt_factory(lr=1e-3),
        criteriorator=BasicCriteriorator(loss_fn, MAX_ITERS_BASE,
            patience=500, grace_period=50),
        lr_scheduler_class=None,
        n_episodes=n_episodes)

    return [
        roll_config('roll-aoc', kernelized_roll_aoc()),
        base_config('bce-weighted',
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([imbalance_ratio]))),
        base_config('mae', mae_loss(pos_weight=imbalance_ratio)),
        base_config('gce-0.7', gce_loss(q=0.7, pos_weight=imbalance_ratio)),
        libauc_cfg,
    ]


if __name__ == '__main__':
    run_dir = init_experiment('results', 'cifar10-imbalanced')

    imbalance_ratios = [9, 50, 100, 200]
    net_creator = ConvNet
    splitter = partial(basic_data_splitter, batch_size=256, is_balanced=True)

    for ir in imbalance_ratios:
        dataset = ImbalancedCifar10Dataset(target_ir=ir)
        n_pos = int(dataset.y.sum().item())
        n_neg = int((1 - dataset.y).sum().item())
        actual_ir = n_neg / max(n_pos, 1)
        print(f'IR={ir}: {n_pos} pos / {n_neg} neg  (actual {actual_ir:.1f})')

        configs = make_configs(net_creator, splitter, actual_ir, N_EPISODES)
        run_configurations(run_dir, configs[:-1], dataset, is_mp=False)
        run_configurations(run_dir, configs[-1:], dataset,
                           device=torch.device('cpu'), is_mp=False)
