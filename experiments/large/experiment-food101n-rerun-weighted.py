"""
Re-run food101n baselines with proper pos_weight class balancing.
Skips roll-aoc (already complete). Writes directly to results-final/food101n/.
Runs on CUDA (beacon-wsl).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import logging
import torch
from torch import nn
from functools import partial

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration
from src.datasets import Food101NDataset
from src.roll import mae_loss, gce_loss, libauc_auc_loss, focal_loss, asymmetric_loss
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
        os.path.dirname(__file__), '..', '..', 'results-final', 'food101n'))
    os.makedirs(run_dir, exist_ok=True)

    net_creator = partial(ConvNet, image_size=64)
    splitter = partial(basic_data_splitter, batch_size=256, is_balanced=True)

    dataset = Food101NDataset()
    n_pos = int(dataset.y.sum().item())
    n_neg = int((1 - dataset.y).sum().item())
    imbalance_ratio = n_neg / max(n_pos, 1)
    imratio = 1.0 / (1.0 + imbalance_ratio)
    print(f'food101n: {n_pos} pos / {n_neg} neg  (IR {imbalance_ratio:.1f})')

    def base_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=net_creator,
            data_splitter=splitter,
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
        model_creator_func=net_creator,
        data_splitter=splitter,
        opt_factory=loss_fn.pesg_opt_factory(lr=1e-3),
        criteriorator=BasicCriteriorator(loss_fn, MAX_ITERS_BASE,
            patience=500, grace_period=50),
        lr_scheduler_class=LR_SCHEDULER,
        lr_scheduler_args=LR_SCHEDULER_ARGS,
        n_episodes=N_EPISODES)

    configs = [
        base_config('bce-weighted',
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([imbalance_ratio]))),
        base_config('mae',             mae_loss(pos_weight=imbalance_ratio)),
        base_config('gce-0.7',        gce_loss(q=0.7, pos_weight=imbalance_ratio)),
        base_config('focal-loss',      focal_loss(gamma=2.0, pos_weight=imbalance_ratio)),
        base_config('asymmetric-loss', asymmetric_loss(pos_weight=imbalance_ratio)),
        libauc_cfg,
    ]

    run_configurations(run_dir, configs, dataset, is_mp=False)

    # ── Log complete AUC summary to MLflow (all configs, including prior roll-aoc) ──
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if tracking_uri:
        import pickle
        import glob
        import mlflow
        import numpy as np
        from sklearn.metrics import roc_auc_score

        logging.info('Logging AUC summary to MLflow...')
        configs_aucs = {}
        for pkl_path in sorted(glob.glob(os.path.join(run_dir, '**', 'test-res.pkl'), recursive=True)):
            parts = pkl_path[len(run_dir):].strip('/').split('/')
            if len(parts) < 2:
                continue
            cfg_name = parts[0]
            try:
                with open(pkl_path, 'rb') as f:
                    ep_res = pickle.load(f)
                test = ep_res.split_results.get('test')
                if test is None or np.isnan(test.yh).any() or np.isinf(test.yh).any():
                    continue
                auc = roc_auc_score(test.y, test.yh)
                configs_aucs.setdefault(cfg_name, []).append(auc)
            except Exception as e:
                logging.warning(f'Could not load {pkl_path}: {e}')

        mlflow.set_experiment('food101n')
        with mlflow.start_run(run_name='auc-summary'):
            for cfg_name, aucs in sorted(configs_aucs.items()):
                mlflow.log_metric(f'{cfg_name}/mean_auc',    float(np.mean(aucs)))
                mlflow.log_metric(f'{cfg_name}/std_auc',     float(np.std(aucs)))
                mlflow.log_metric(f'{cfg_name}/n_episodes',  float(len(aucs)))
                for i, auc in enumerate(aucs):
                    mlflow.log_metric(f'{cfg_name}/ep{i}_auc', float(auc))
        logging.info('AUC summary logged to MLflow experiment "food101n"')
    else:
        logging.warning('MLFLOW_TRACKING_URI not set — skipping MLflow summary')
