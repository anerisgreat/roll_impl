import torch
from functools import partial
from torch import nn

from torchvision import datasets, transforms

from src.experiment import _perform_multiple_episodes
from src.experiment import *

from src.datasets import ForestCoverDataset
from src.summary import _gen_roc_to_file

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            x for y in [
                [nn.Linear(10, 10), nn.ReLU()] for _ in range(1)]
            for x in y]
        self._layers = nn.Sequential(
            *layers, nn.Linear(10, 1))

    def forward(self, x):
        return self._layers(x)

def _torch_mean_std(yh):
    yhmean = torch.mean(yh)
    yhstd = torch.std(yh)
    return yhmean, yhstd

def _torch_normal_fit(yh):
    yhmean, yhstd = _torch_mean_std(yh)
    return torch.distributions.normal.Normal(yhmean, yhstd)

def _split_true_false(yh, y):
    true_indeces = torch.argwhere(y)[:,0]
    false_indeces = torch.argwhere(torch.logical_not(y))[:,0]

    true_yh = yh[true_indeces]
    false_yh = yh[false_indeces]

    return true_yh, false_yh

def roll_loss_from_fpr(fpr):
    def _partial(yh, y):
        true_yh, false_yh = _split_true_false(yh, y)
        true_normal = _torch_normal_fit(true_yh)
        false_normal = _torch_normal_fit(false_yh)
        return true_normal.cdf(false_normal.icdf(torch.tensor(1 - fpr)))
    return _partial

summary_dir = 'tmp'
device = torch.device('cpu')
dataset = ForestCoverDataset()

oneshot_datasplitter = partial(basic_data_splitter, is_oneshot = True)

configurations = [
    ExperimentConfiguration(
        name = 'BCE',
        model_creator_func = MyNet,
        dataset = dataset,
        data_splitter = basic_data_splitter,
        optim_class = torch.optim.Adam,
        optim_args = {'lr' : 0.001},
        criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), 100),
        device = device,
        n_episodes = 5),
    ExperimentConfiguration(
        name = 'roll-0.04',
        model_creator_func = MyNet,
        dataset = dataset,
        data_splitter = oneshot_datasplitter,
        optim_class = torch.optim.SGD,
        optim_args = {'lr' : 0.001},
        criteriorator = BasicCriteriorator(roll_loss_from_fpr(0.04), 100),
        device = device,
        n_episodes = 5),
    ExperimentConfiguration(
        name = 'roll-0.02',
        model_creator_func = MyNet,
        dataset = dataset,
        data_splitter = oneshot_datasplitter,
        optim_class = torch.optim.SGD,
        optim_args = {'lr' : 0.001},
        criteriorator = BasicCriteriorator(roll_loss_from_fpr(0.02), 100),
        device = device,
        n_episodes = 5),
    ]

res_c = _perform_multiple_episodes(
    summary_dir = summary_dir,
    model_creator_func = model_creator_func,
    dataset = dataset,
    data_splitter = basic_data_splitter,
    optim_class = torch.optim.Adam,
    optim_args = {'lr' : 0.001},
    criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), 100),
    device = device,
    n_episodes = 5)

_gen_roc_to_file(
    fname='./graph.html',
    multi_ep_results = [res_a, res_b, res_c],
    names = ['fpr0.4', 'fpr0.2', 'bce'],
    disabled_modes = ['Train', 'Validation'])
