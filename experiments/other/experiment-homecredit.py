"""
Home Credit Default Risk: binary classification on loan application data.
(Kaggle competition: home-credit-default-risk)

~307K applications, ~106 numeric features, IR ~11:1 (~8% default rate).
Label noise: self-reported income, document flags, and derived ratios from
credit bureaus introduce real-world annotation noise.

Network: 256→128→1 MLP (2 hidden layers, ReLU, Dropout 0.3).
libauc-auroc runs on CPU (LibAUC does not support MPS).

Download:
  kaggle competitions download -c home-credit-default-risk -f application_train.csv \
    -p ~/.data/homecredit --unzip
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import logging
import torch
from torch import nn
from functools import partial

from src.experiment import run_configurations, BasicCriteriorator, \
    ExperimentConfiguration, KernelScheduler, make_shuffled_splitter
from src.datasets import HomeCreditDataset
from src.roll import kernelized_roll_aoc, kernelized_roll_tpr, mae_loss, gce_loss, \
    libauc_auc_loss, focal_loss, asymmetric_loss

MAX_ITERS_ROLL = 1500
MAX_ITERS_BASE = 5000
N_EPISODES     = 5

LR_SCHEDULER      = torch.optim.lr_scheduler.StepLR
LR_SCHEDULER_ARGS = {'step_size': 50, 'gamma': 0.5}


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return torch.flatten(self._layers(x))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')

    run_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'results-final', 'homecredit'))
    os.makedirs(run_dir, exist_ok=True)

    dataset = HomeCreditDataset()
    n_pos = int(dataset.y.sum().item())
    n_neg = int((1 - dataset.y).sum().item())
    imbalance_ratio = n_neg / max(n_pos, 1)
    imratio = n_pos / len(dataset)
    print(f'home-credit: {n_pos} pos / {n_neg} neg  (IR {imbalance_ratio:.1f})')

    make_net = partial(Net, dataset.x.shape[1])

    def roll_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=make_net,
            data_splitter=make_shuffled_splitter(batch_size=256, is_balanced=True),
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
            data_splitter=make_shuffled_splitter(batch_size=256, is_balanced=True),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss_func, MAX_ITERS_BASE,
                patience=500, grace_period=50),
            lr_scheduler_class=LR_SCHEDULER,
            lr_scheduler_args=LR_SCHEDULER_ARGS,
            n_episodes=N_EPISODES)

    loss_fn = libauc_auc_loss(margin=1.0, imratio=imratio)
    libauc_cfg = ExperimentConfiguration(
        name='libauc-auroc',
        model_creator_func=make_net,
        data_splitter=make_shuffled_splitter(batch_size=256, is_balanced=True),
        opt_factory=loss_fn.pesg_opt_factory(lr=0.01),
        criteriorator=BasicCriteriorator(loss_fn, MAX_ITERS_BASE,
            patience=500, grace_period=50),
        lr_scheduler_class=LR_SCHEDULER,
        lr_scheduler_args=LR_SCHEDULER_ARGS,
        n_episodes=N_EPISODES)

    mps_configs = [
        roll_config('roll-aoc',  kernelized_roll_aoc()),
        roll_config('roll-tpr90', kernelized_roll_tpr(tpr=0.9)),
        base_config('bce-weighted',
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([imbalance_ratio]))),
        base_config('gce-0.7',        gce_loss(q=0.7, pos_weight=imbalance_ratio)),
        base_config('mae',             mae_loss(pos_weight=imbalance_ratio)),
        base_config('focal-loss',      focal_loss(gamma=2.0, pos_weight=imbalance_ratio)),
        base_config('asymmetric-loss', asymmetric_loss(pos_weight=imbalance_ratio)),
    ]

    run_configurations(run_dir, mps_configs, dataset,
                       is_mp=False, sequential_episodes=True)
    run_configurations(run_dir, [libauc_cfg], dataset,
                       device=torch.device('cpu'), is_mp=False, sequential_episodes=True)
