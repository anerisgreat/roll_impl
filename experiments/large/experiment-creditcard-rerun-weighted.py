"""
Re-run gce-0.7, mae, focal-loss, asymmetric-loss on credit card fraud with
proper pos_weight class balancing. Writes directly into results-final/creditcard/,
overwriting the previously unweighted results, then regenerates auc.csv.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from functools import partial
from torch import nn
import logging
import pickle
import csv

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration
from src.datasets import CreditCardFraudDataset
from src.utils import init_experiment
from src.roll import mae_loss, gce_loss, focal_loss, asymmetric_loss

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


def regen_auc_csv(results_dir):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rows = []
    for config in sorted(os.listdir(results_dir)):
        config_dir = os.path.join(results_dir, config)
        if not os.path.isdir(config_dir):
            continue
        for ep in sorted(os.listdir(config_dir)):
            pkl = os.path.join(config_dir, ep, 'test-res.pkl')
            if not os.path.isfile(pkl):
                continue
            with open(pkl, 'rb') as f:
                res = pickle.load(f)
            test = res.split_results.get('test')
            if test is None:
                continue
            auc = roc_auc_score(np.array(test.y).ravel(), np.array(test.yh).ravel())
            rows.append((config, ep, auc))

    out = os.path.join(results_dir, 'auc.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'episode', 'test_auc'])
        w.writerows(rows)
    print(f'Regenerated {out} ({len(rows)} rows)')


if __name__ == '__main__':
    run_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'results-final', 'creditcard'))
    os.makedirs(run_dir, exist_ok=True)

    dataset = CreditCardFraudDataset()
    num_true  = int(dataset.y.sum().item())
    num_false = int((1 - dataset.y).sum().item())
    imbalance_ratio = num_false / num_true
    print(f"samples: {len(dataset)}, IR: {imbalance_ratio:.1f}")

    make_net = partial(Net, dataset.x.shape[1])
    splitter = partial(basic_data_splitter, batch_size=8096, is_balanced=True)

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
        base_config('gce-0.7',        gce_loss(q=0.7, pos_weight=imbalance_ratio)),
        base_config('mae',             mae_loss(pos_weight=imbalance_ratio)),
        base_config('focal-loss',      focal_loss(gamma=2.0, pos_weight=imbalance_ratio)),
        base_config('asymmetric-loss', asymmetric_loss(pos_weight=imbalance_ratio)),
    ]

    logging.info('Re-running 4 weighted configs...')
    run_configurations(run_dir, configurations, dataset,
                       is_mp=False, sequential_episodes=True)
    logging.info('Done. Regenerating auc.csv...')
    regen_auc_csv(run_dir)
    logging.info('Complete.')
