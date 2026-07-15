import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from functools import partial
from torch import nn
import logging

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration, KernelScheduler
from src.datasets import HomeCreditDataset
from src.utils import init_experiment
from src.roll import kernelized_roll_fpr

MAX_ITERS = 500
N_EPISODES = 3

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

if __name__ == '__main__':
    run_dir = init_experiment('results', 'homecredit')
    dataset = HomeCreditDataset()

    input_dim = dataset.x.shape[1]
    print(f"Input dimension: {input_dim}, samples: {len(dataset)}")

    num_true = dataset.y.sum().item()
    num_false = (1 - dataset.y).sum().item()
    imbalance_ratio = num_false / num_true

    make_net = partial(Net, input_dim)

    def roll_config(name, loss_func):
        return ExperimentConfiguration(
            name=name,
            model_creator_func=make_net,
            data_splitter=partial(basic_data_splitter, batch_size=512, is_balanced=True),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-4},
            criteriorator=BasicCriteriorator(loss_func, MAX_ITERS,
                kernel_scheduler=KernelScheduler(initial_gamma=100, decay_every=100)),
            n_episodes=N_EPISODES)

    def bce_config(name, pos_weight=None):
        loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]) if pos_weight is not None else None)
        return ExperimentConfiguration(
            name=name,
            model_creator_func=make_net,
            data_splitter=partial(basic_data_splitter, batch_size=512, is_balanced=True),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-4},
            criteriorator=BasicCriteriorator(loss, MAX_ITERS),
            n_episodes=N_EPISODES)

    fprs = [0.02, 0.05]
    configurations = (
        [roll_config(f'roll-{rr:0.2f}', kernelized_roll_fpr(rr)) for rr in fprs] +
        [roll_config(f'roll+bce-{rr:0.2f}',
            kernelized_roll_fpr(rr, bce_weight=0.5, bce_pos_weight=imbalance_ratio))
            for rr in fprs] +
        [bce_config('bce'),
         bce_config('bce-weighted', pos_weight=imbalance_ratio)]
    )

    logging.info('Starting experiment!')
    run_configurations(run_dir, configurations, dataset)
    logging.info('Script completed!')
