"""
Run libauc-auroc on credit card fraud — CPU only (LibAUC does not support MPS).
Writes results directly into results-final/creditcard/ alongside existing configs,
then regenerates auc.csv for the full set.
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
from src.roll import libauc_auc_loss

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
    """Regenerate auc.csv from all test-res.pkl files under results_dir."""
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
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    run_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'results-final', 'creditcard')
    run_dir = os.path.normpath(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    dataset = CreditCardFraudDataset()
    num_true  = int(dataset.y.sum().item())
    num_false = int((1 - dataset.y).sum().item())
    imratio   = num_true / (num_true + num_false)
    print(f"samples: {len(dataset)}, IR: {num_false/num_true:.1f}, imratio: {imratio:.5f}")

    make_net = partial(Net, dataset.x.shape[1])
    splitter = partial(basic_data_splitter, batch_size=8096, is_balanced=True)

    loss_fn = libauc_auc_loss(margin=1.0, imratio=imratio)
    libauc_cfg = ExperimentConfiguration(
        name='libauc-auroc',
        model_creator_func=make_net,
        data_splitter=splitter,
        opt_factory=loss_fn.pesg_opt_factory(lr=0.01),
        criteriorator=BasicCriteriorator(loss_fn, MAX_ITERS_BASE,
            patience=500, grace_period=50),
        lr_scheduler_class=LR_SCHEDULER,
        lr_scheduler_args=LR_SCHEDULER_ARGS,
        n_episodes=N_EPISODES)

    logging.info('Running libauc-auroc on CPU...')
    run_configurations(run_dir, [libauc_cfg], dataset,
                       device=torch.device('cpu'),
                       is_mp=False, sequential_episodes=True)
    logging.info('Done. Regenerating auc.csv...')
    regen_auc_csv(run_dir)
    logging.info('Complete.')
