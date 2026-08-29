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
    make_poisoned_splitter, make_random_poisoned_splitter, make_noisy_splitter, \
    BasicCriteriorator, ExperimentConfiguration, KernelScheduler
from src.utils import init_experiment
from src.roll import kernelized_roll_aoc, kernelized_roll_tpr, mae_loss, gce_loss, libauc_auc_loss, \
    focal_loss, asymmetric_loss


def run_keel_experiment(dataset_name, n_hidden_layers=3, hidden_size=None,
                        dropout_p=0.1, max_iters=1500, n_episodes=5, batch_size=256,
                        n_poison_duplicates=0, noise_rate=0.0, results_dir='results', is_mp=False,
                        force_cpu=False):
    run_dir = init_experiment(results_dir, dataset_name, console_level=logging.INFO)
    dataset = KeelDataset(dataset_name)
    input_size = dataset.x.shape[1]

    num_true = dataset.y.sum().item()
    num_false = (1 - dataset.y).sum().item()

    net_creator = partial(KeelNet, input_size, n_hidden_layers, hidden_size, dropout_p)

    if noise_rate > 0.0:
        def new_splitter():
            return make_noisy_splitter(noise_rate, batch_size=batch_size, is_balanced=True)
    elif n_poison_duplicates > 0:
        def new_splitter():
            return make_random_poisoned_splitter(n_poison_duplicates, batch_size=batch_size, is_balanced=True)
    else:
        def new_splitter():
            return partial(basic_data_splitter, batch_size=batch_size, is_balanced=True)

    imbalance_ratio = num_false / num_true

    def roll_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=net_creator,
            data_splitter=new_splitter(),
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
            data_splitter=new_splitter(),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
            criteriorator=BasicCriteriorator(loss, 5000, patience=500, grace_period=50),
            lr_scheduler_class=None,
            n_episodes=n_episodes)

    def robust_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=net_creator,
            data_splitter=new_splitter(),
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
            data_splitter=new_splitter(),
            opt_factory=loss_fn.pesg_opt_factory(lr=0.1),
            criteriorator=BasicCriteriorator(loss_fn, 5000, patience=500, grace_period=50),
            lr_scheduler_class=None,
            n_episodes=n_episodes)

    configurations = [
        roll_config('roll-aoc', kernelized_roll_aoc()),
        roll_config('roll-tpr90', kernelized_roll_tpr(tpr=0.9)),
        bce_config('bce-weighted', pos_weight=imbalance_ratio),
        libauc_config('libauc-auroc'),
        robust_config('gce-0.7', gce_loss(q=0.7, pos_weight=imbalance_ratio)),
        robust_config('mae', mae_loss(pos_weight=imbalance_ratio)),
        robust_config('focal-loss', focal_loss(gamma=2.0, pos_weight=imbalance_ratio)),
        robust_config('asymmetric-loss', asymmetric_loss(pos_weight=imbalance_ratio)),
    ]

    cpu = torch.device('cpu')
    device = cpu if force_cpu else None
    non_libauc = [c for c in configurations if c.name != 'libauc-auroc']
    libauc = [c for c in configurations if c.name == 'libauc-auroc']

    logging.info('Starting experiment!')
    if non_libauc:
        run_configurations(run_dir, non_libauc, dataset, device=device,
                           is_mp=is_mp, sequential_episodes=True)
    if libauc:
        run_configurations(run_dir, libauc, dataset,
                           device=cpu, is_mp=False, sequential_episodes=True)
    logging.info('Script completed!')
