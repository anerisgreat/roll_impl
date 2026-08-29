"""
Generate auc.csv for each KEEL dataset in results-final/keel/ from existing test-res.pkl files.
Run from the impl/ directory: nix develop --command python3 scripts/gen_keel_auc_csv.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import pandas as pd
import numpy as np
import torch
import torch.serialization
from sklearn.metrics import roc_auc_score
from src.experiment import EpisodeResult

# pkl files were saved on Mac (MPS); remap tensors to CPU on non-MPS machines
if not torch.backends.mps.is_available():
    _orig_restore = torch.serialization.default_restore_location
    def _mps_to_cpu(storage, location):
        return _orig_restore(storage, 'cpu' if location.startswith('mps') else location)
    torch.serialization.default_restore_location = _mps_to_cpu

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results-final', 'keel')

def auc_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        ep: EpisodeResult = pickle.load(f)
    test = ep.split_results.get('test')
    if test is None:
        return None
    y = np.array(test.y)
    yh = np.array(test.yh)
    if len(np.unique(y)) < 2 or np.isnan(yh).any() or np.isinf(yh).any():
        return None
    return float(roc_auc_score(y, yh))

def gen_dataset(ds_path):
    rows = []
    for config in sorted(os.listdir(ds_path)):
        config_path = os.path.join(ds_path, config)
        if not os.path.isdir(config_path):
            continue
        for ep_dir in sorted(os.listdir(config_path)):
            pkl_path = os.path.join(config_path, ep_dir, 'test-res.pkl')
            if not os.path.isfile(pkl_path):
                continue
            auc = auc_from_pkl(pkl_path)
            if auc is not None:
                rows.append({'config': config, 'episode': int(ep_dir), 'test_auc': auc})
    return pd.DataFrame(rows)

for ds_name in sorted(os.listdir(RESULTS_DIR)):
    ds_path = os.path.join(RESULTS_DIR, ds_name)
    if not os.path.isdir(ds_path):
        continue
    df = gen_dataset(ds_path)
    if df.empty:
        print(f'{ds_name}: no results found')
        continue
    out = os.path.join(ds_path, 'auc.csv')
    df.to_csv(out, index=False)
    print(f'{ds_name}: wrote {len(df)} rows -> {out}')
