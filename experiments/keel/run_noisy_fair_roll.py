"""
Re-run roll-aoc for all KEEL datasets × noise levels with a fair config:
  max_iters=5000, patience=500  (matching the baselines)
  optimizer, lr, KernelScheduler, StepLR unchanged from original roll_config.

Writes flat to results-fair-roll/keel-noise-{05,10,20}/<dataset>/roll-aoc/
so results can be dropped into results-final later.

Run on Mac:
  python experiments/keel/run_noisy_fair_roll.py
"""
import sys, os
os.environ.pop('MLFLOW_TRACKING_URI', None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import logging
import torch
from functools import partial

from src.datasets import KeelDataset
from src.networks import KeelNet
from src.experiment import (run_configurations, make_noisy_splitter,
    BasicCriteriorator, ExperimentConfiguration, KernelScheduler)
from src.roll import kernelized_roll_aoc

DATASETS = [
    'glass0', 'glass1', 'glass2', 'glass6',
    'haberman', 'iris0', 'new-thyroid1', 'pima',
    'vehicle2', 'vowel0', 'wisconsin', 'yeast3',
]
NOISE_LEVELS = [('05', 0.05), ('10', 0.10), ('20', 0.20)]

N_EPISODES  = 10
MAX_ITERS   = 5000
PATIENCE    = 500
GRACE       = 50
BATCH_SIZE  = 256

OUT_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'results-fair-roll'))


def _all_episodes_done(config_dir, n):
    return all(
        os.path.isfile(os.path.join(config_dir, str(i), 'test-res.pkl'))
        for i in range(n)
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S')

    device = torch.device('cpu')
    logging.info('Forcing CPU (6.6x faster than MPS for KEEL batch sizes)')

    total = len(NOISE_LEVELS) * len(DATASETS)
    done  = 0

    for noise_tag, noise_rate in NOISE_LEVELS:
        for ds_name in DATASETS:
            done += 1
            run_dir = os.path.join(OUT_BASE, f'keel-noise-{noise_tag}', ds_name)
            roll_dir = os.path.join(run_dir, 'roll-aoc')

            if _all_episodes_done(roll_dir, N_EPISODES):
                logging.info(f'[{done}/{total}] {ds_name} noise={noise_rate} — already done, skipping')
                continue

            logging.info(f'[{done}/{total}] {ds_name} noise={noise_rate}')
            os.makedirs(run_dir, exist_ok=True)

            dataset = KeelDataset(ds_name)
            input_size = dataset.x.shape[1]
            net_creator = partial(KeelNet, input_size, 3, None, 0.1)
            splitter = make_noisy_splitter(noise_rate, batch_size=BATCH_SIZE, is_balanced=True)

            config = ExperimentConfiguration(
                name='roll-aoc',
                model_creator_func=net_creator,
                data_splitter=splitter,
                optim_class=torch.optim.RMSprop,
                optim_args={'lr': 1e-3, 'weight_decay': 1e-5},
                criteriorator=BasicCriteriorator(
                    kernelized_roll_aoc(), MAX_ITERS,
                    kernel_scheduler=KernelScheduler(32.0, decay_every=20),
                    patience=PATIENCE, grace_period=GRACE),
                lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
                lr_scheduler_args={'step_size': 50, 'gamma': 0.5},
                n_episodes=N_EPISODES)

            run_configurations(run_dir, [config], dataset,
                               device=device, is_mp=False, sequential_episodes=True)

    logging.info('All done.')
