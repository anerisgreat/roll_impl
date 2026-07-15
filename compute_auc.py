"""Compute test-split AUC from poisoned experiment result pickles.

Usage:
    python compute_auc.py [results_root]   (default: poison)

For each dataset, picks the latest run directory, loads test-res.pkl per
episode per config, and prints mean ± std AUC across episodes.
"""
import os
import sys
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

results_root = sys.argv[1] if len(sys.argv) > 1 else 'poison'


def latest_run(dataset_dir):
    """Return the path to the most recent timestamped run folder."""
    entries = [
        e for e in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, e)) and e[0].isdigit()
    ]
    if not entries:
        return None
    return os.path.join(dataset_dir, sorted(entries)[-1])


def auc_for_config(run_dir, config):
    config_dir = os.path.join(run_dir, config)
    if not os.path.isdir(config_dir):
        return None
    aucs = []
    ep_dirs = [e for e in os.listdir(config_dir) if e.isdigit()]
    for ep in sorted(ep_dirs, key=int):
        pkl = os.path.join(config_dir, ep, 'test-res.pkl')
        if not os.path.exists(pkl):
            continue
        with open(pkl, 'rb') as f:
            ep_res = pickle.load(f)
        test = ep_res.split_results['test']
        y, yh = test.y, test.yh
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, yh))
    return np.array(aucs) if aucs else None


datasets = sorted(os.listdir(results_root))
rows = []
all_configs = set()

for ds in datasets:
    ds_dir = os.path.join(results_root, ds)
    if not os.path.isdir(ds_dir):
        continue
    run_dir = latest_run(ds_dir)
    if run_dir is None:
        continue
    configs = [
        e for e in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, e)) and not e[0].isdigit()
    ]
    row = {'dataset': ds}
    for cfg in configs:
        all_configs.add(cfg)
        aucs = auc_for_config(run_dir, cfg)
        if aucs is not None and len(aucs) > 0:
            row[cfg] = (aucs.mean(), aucs.std(), len(aucs))
    rows.append(row)

configs_sorted = sorted(all_configs)

# Header
col_w = 14
header = f"{'Dataset':<16}" + ''.join(f"{c:^{col_w}}" for c in configs_sorted)
print(header)
print('-' * len(header))

for row in rows:
    line = f"{row['dataset']:<16}"
    for cfg in configs_sorted:
        if cfg in row:
            mean, std, n = row[cfg]
            line += f"  {mean:.3f}±{std:.3f} ({n})"
        else:
            line += f"{'—':^{col_w}}"
    print(line)
