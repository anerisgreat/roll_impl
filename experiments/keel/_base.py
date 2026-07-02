import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch import nn
from functools import partial
import logging

from src.datasets import KeelDataset
from src.networks import KeelNet
from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration
from src.utils import init_experiment
from src.roll import kernelized_roll_fpr


def run_keel_experiment(dataset_name, n_hidden_layers=3, hidden_size=None,
                        dropout_p=0.0, fprs=(0.02, 0.05),
                        max_iters=1500, n_episodes=3, batch_size=128):
    run_dir = init_experiment('results', dataset_name)
    dataset = KeelDataset(dataset_name)
    input_size = dataset.x.shape[1]

    net_creator = lambda: KeelNet(input_size, n_hidden_layers, hidden_size, dropout_p)

    configurations = [
        ExperimentConfiguration(
            name=f'roll-kernel-{rr:0.2f}',
            model_creator_func=net_creator,
            data_splitter=partial(basic_data_splitter, batch_size=batch_size, is_balanced=True),
            optim_class=torch.optim.Adam,
            optim_args={'lr': 1e-4},
            criteriorator=BasicCriteriorator(kernelized_roll_fpr(rr), max_iters),
            n_episodes=n_episodes)
        for rr in fprs
    ] + [ExperimentConfiguration(
        name='BCE',
        model_creator_func=net_creator,
        data_splitter=partial(basic_data_splitter, batch_size=batch_size, is_balanced=True),
        optim_class=torch.optim.Adam,
        optim_args={'lr': 1e-4},
        criteriorator=BasicCriteriorator(nn.BCEWithLogitsLoss(), max_iters),
        n_episodes=n_episodes,
    )]

    logging.info('Starting experiment!')
    run_configurations(run_dir, configurations, dataset, is_mp=False)
    logging.info('Script completed!')
