import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch import nn
from functools import partial
import logging

from src.datasets import KeelDataset
from src.networks import KeelNet
from src.experiment import run_configurations, basic_data_splitter, \
    make_poisoned_splitter, BasicCriteriorator, ExperimentConfiguration, KernelScheduler
from src.utils import init_experiment
from src.roll import kernelized_roll_tpr, kernelized_roll_fpr, kernelized_roll_aoc, \
    mae_loss, gce_loss, libauc_auc_loss

TARGET_TPR = 0.95

def run_keel_experiment(dataset_name, n_hidden_layers=3, hidden_size=None,
                        dropout_p=0.1, max_iters=1500, n_episodes=5, batch_size=256,
                        n_poison_duplicates=0, results_dir='results', is_mp=True):
    run_dir = init_experiment(results_dir, dataset_name, console_level=logging.INFO)
    dataset = KeelDataset(dataset_name)
    input_size = dataset.x.shape[1]

    num_true = dataset.y.sum().item()
    num_false = (1 - dataset.y).sum().item()
    imbalance_ratio = num_false / num_true

    net_creator = partial(KeelNet, input_size, n_hidden_layers, hidden_size, dropout_p)

    if n_poison_duplicates > 0:
        data_splitter = make_poisoned_splitter(n_poison_duplicates, batch_size=batch_size, is_balanced=True)
    else:
        data_splitter = partial(basic_data_splitter, batch_size=batch_size, is_balanced=True)

    def roll_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=net_creator,
            data_splitter=data_splitter,
            optim_class=torch.optim.RMSprop,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss_func, max_iters,
                kernel_scheduler=KernelScheduler(32.0, decay_every=20),
                patience=100, grace_period=50),
            lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_args={'step_size': 50, 'gamma': 0.5},
            n_episodes=n_episodes)

    def bce_config(name, pos_weight=None):
        loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]) if pos_weight is not None else None)
        return ExperimentConfiguration(
            name=name,
            model_creator_func=net_creator,
            data_splitter=data_splitter,
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss, 5000, patience=500, grace_period=50),
            lr_scheduler_class=None,
            n_episodes=n_episodes)

    def robust_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=net_creator,
            data_splitter=data_splitter,
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss_func, 5000, patience=500, grace_period=50),
            lr_scheduler_class=None,
            n_episodes=n_episodes)

    def libauc_config(name):
        imratio = num_true / (num_true + num_false)
        loss_fn = libauc_auc_loss(margin=1.0, imratio=imratio)
        return ExperimentConfiguration(
            name=name,
            model_creator_func=net_creator,
            data_splitter=data_splitter,
            opt_factory=loss_fn.pesg_opt_factory(lr=0.1),
            criteriorator=BasicCriteriorator(loss_fn, 5000, patience=500, grace_period=50),
            lr_scheduler_class=None,
            n_episodes=n_episodes)

    if n_poison_duplicates > 0:
        configurations = [
            roll_config('roll-aoc', kernelized_roll_aoc()),
            bce_config('bce-weighted', pos_weight=imbalance_ratio),
            robust_config('mae', mae_loss),
            robust_config('gce-0.7', gce_loss(q=0.7)),
            libauc_config('libauc-auroc'),
        ]
    else:
        configurations = [
            roll_config('roll', kernelized_roll_tpr(TARGET_TPR)),
            roll_config('roll+bce', kernelized_roll_tpr(TARGET_TPR, bce_weight=0.5, bce_pos_weight=imbalance_ratio)),
            roll_config('roll-fpr-0.40', kernelized_roll_fpr(0.40)),
            roll_config('roll-fpr-0.40+bce', kernelized_roll_fpr(0.40, bce_weight=0.5, bce_pos_weight=imbalance_ratio)),
            bce_config('bce'),
            bce_config('bce-weighted', pos_weight=imbalance_ratio),
            libauc_config('libauc-auroc'),
        ]

    logging.info('Starting experiment!')
    run_configurations(run_dir, configurations, dataset, tpr_summary=TARGET_TPR, is_mp=is_mp)
    logging.info('Script completed!')
