"""
Regenerate auc.csv and per-noise-condition ROC HTML files for a cifar10n run dir.

Usage:
    python scripts/regen_cifar10n_summaries.py results/cifar10n/2026-07-19-06-52
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import pandas as pd

import torch
import torch.serialization as _ts

# Remap MPS tensors to CPU when loading on non-MPS machines
_orig_restore = _ts.default_restore_location
def _cpu_restore(storage, location):
    return _orig_restore(storage, 'cpu' if location == 'mps' else location)
_ts.default_restore_location = _cpu_restore

from src.experiment import EpisodeResult, ModelDataResult, MultiEpisodeResult
from src.summary import _gen_roc_to_file, write_auc_csv

# ── stub so pkl loads work ────────────────────────────────────────────────────
class ConvNet(nn.Module): pass

CONFIG_ORDER = ['roll-aoc', 'bce-weighted', 'mae', 'gce-0.7', 'libauc-auroc',
                'focal-loss', 'asymmetric-loss']
NOISE_TYPES  = ['clean', 'aggre', 'worse']
N_EPISODES   = 3


def load_multi_ep(run_dir, config_name):
    eps = []
    for ep in range(N_EPISODES):
        path = os.path.join(run_dir, config_name, str(ep), 'test-res.pkl')
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            eps.append(pickle.load(f))
    return MultiEpisodeResult(eps)


def main(run_dir):
    # ── collect all results ───────────────────────────────────────────────────
    all_multi_eps = {}   # config_name -> MultiEpisodeResult
    for noise in NOISE_TYPES:
        for cfg in CONFIG_ORDER:
            name = f'{noise}-{cfg}'
            mer = load_multi_ep(run_dir, name)
            if mer is not None:
                all_multi_eps[name] = mer
            else:
                print(f'  skipping {name} (incomplete)')

    # ── regenerate auc.csv ────────────────────────────────────────────────────
    class _FakeCfg:
        def __init__(self, name): self.name = name

    # wipe existing so we rebuild from scratch
    auc_path = os.path.join(run_dir, 'auc.csv')
    if os.path.exists(auc_path):
        os.remove(auc_path)

    write_auc_csv(run_dir,
                  list(all_multi_eps.values()),
                  [_FakeCfg(n) for n in all_multi_eps])
    print(f'auc.csv written ({len(all_multi_eps)} configs)')

    # ── regenerate per-noise ROC HTML ─────────────────────────────────────────
    for noise in NOISE_TYPES:
        names, mers = [], []
        for cfg in CONFIG_ORDER:
            name = f'{noise}-{cfg}'
            if name in all_multi_eps:
                names.append(name)
                mers.append(all_multi_eps[name])

        if not mers:
            print(f'  no data for {noise}, skipping ROC')
            continue

        fname = os.path.join(run_dir, f'roc-{noise}.html')
        _gen_roc_to_file(fname=fname, multi_ep_results=mers,
                         names=names, disabled_modes=['Train', 'Val'])
        print(f'roc-{noise}.html written ({len(mers)} configs)')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
