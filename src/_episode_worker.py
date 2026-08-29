"""
Subprocess entry point for parallel episode execution.

Called by _perform_multiple_episodes when running on MPS (which cannot share
tensors across processes via mp.Pool). Each worker runs a single episode and
saves its result to disk via the normal summarize_episode path.

Invoked as:
    python -m src._episode_worker <config_pkl> <dataset_pkl> <episode_idx> <summary_dir> <device>
"""
import sys, os, pickle, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.experiment import _perform_episode, joinmakedir


def main():
    config_path, dataset_path, episode_idx, summary_dir, device_str = sys.argv[1:]
    episode_idx = int(episode_idx)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    device = torch.device(device_str)

    logging.basicConfig(level=logging.WARNING)

    train_loader, val_loader, test_loader = config.data_splitter(dataset)
    _perform_episode(
        summary_dir=joinmakedir(summary_dir, str(episode_idx)),
        data_loaders={'train': train_loader, 'val': val_loader, 'test': test_loader},
        logger=None,
        device=device,
        config=config,
    )


if __name__ == '__main__':
    main()
