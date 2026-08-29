"""
Regenerate the unified ROC graph.html from pickled episode results in a results-final dataset dir.

Usage:
    python scripts/regen_roc.py results-final/bank-marketing
"""
import sys, os, pickle, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import io
import torch
import torch.nn as nn
import torch.storage
from src.experiment import MultiEpisodeResult
from src.summary import _gen_roc_to_file

# Models may have been saved on MPS (Mac) — remap tensors to CPU on load
_orig_load_from_bytes = torch.storage._load_from_bytes
torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b), weights_only=False, map_location='cpu'
)

# Experiment scripts define Net (and other models) in __main__, so pickle
# can't find them when loading from a different entry point. Register a
# dummy so unpickling succeeds — we only need split_results, not the model.
class _DummyModule(nn.Module):
    pass

class _PermissiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except AttributeError:
            return _DummyModule

def _load_pkl(path):
    with open(path, 'rb') as f:
        return _PermissiveUnpickler(f).load()

def load_multi_ep_result(config_dir):
    episode_dirs = sorted(
        d for d in os.listdir(config_dir)
        if os.path.isdir(os.path.join(config_dir, d)) and d.isdigit()
    )
    episodes = []
    for ep in episode_dirs:
        pkl_path = os.path.join(config_dir, ep, 'test-res.pkl')
        if not os.path.exists(pkl_path):
            print(f"  WARNING: missing {pkl_path}, skipping episode {ep}")
            continue
        episodes.append(_load_pkl(pkl_path))
    if not episodes:
        return None
    return MultiEpisodeResult(episodes)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_dir = sys.argv[1]
    config_names = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )

    multi_ep_results = []
    names = []
    for name in config_names:
        config_dir = os.path.join(dataset_dir, name)
        result = load_multi_ep_result(config_dir)
        if result is None:
            print(f"Skipping {name} (no episodes found)")
            continue
        print(f"Loaded {name}: {len(result.episode_results)} episodes")
        multi_ep_results.append(result)
        names.append(name)

    out = os.path.join(dataset_dir, 'graph.html')
    _gen_roc_to_file(
        fname=out,
        multi_ep_results=multi_ep_results,
        names=names,
        disabled_modes=['Train', 'Val'],
    )
    print(f"\nWrote unified ROC to {out}")

if __name__ == '__main__':
    main()
