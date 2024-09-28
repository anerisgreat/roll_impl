import torch
from functools import partial
from torch import nn

from torchvision import datasets, transforms
import logging

from src.experiment import run_configurations, basic_data_splitter, \
    BasicCriteriorator, ExperimentConfiguration
from src.datasets import ForestCoverDataset, Cifar10Dataset
from src.utils import init_experiment

run_dir = init_experiment('tmp', 'test_experiment')

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            x for y in [
                [nn.Linear(10, 10), nn.ReLU()] for _ in range(0)]
            for x in y]
        self._layers = nn.Sequential(
            *layers, nn.Linear(10, 1))

    def forward(self, x):
        return self._layers(x)

class BasicConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = nn.Sequential(*[
            transforms.CenterCrop(5),
            nn.Conv2d(3, 3, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3, 1)])

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
# dataset = Cifar10Dataset()
dataset = ForestCoverDataset()

oneshot_datasplitter = partial(basic_data_splitter, is_oneshot = True)

logging.debug('Just before starting!')

configurations = [
    ExperimentConfiguration(
        name = 'BCE',
        model_creator_func = MyNet,
        data_splitter = basic_data_splitter,
        optim_class = torch.optim.Adam,
        optim_args = {'lr' : 0.001},
        criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), 10),
        # device = device,
        n_episodes = 1),
    # ExperimentConfiguration(
    #     name = 'roll-0.5',
    #     model_creator_func = BasicConvNet,
    #     data_splitter = oneshot_datasplitter,
    #     optim_class = torch.optim.SGD,
    #     optim_args = {'lr' : 0.01},
    #     criteriorator = BasicCriteriorator(roll_loss_from_fpr(0.5), 300),
    #     n_episodes = 5),
    # ExperimentConfiguration(
    #     name = 'roll-0.8',
    #     model_creator_func = BasicConvNet,
    #     data_splitter = oneshot_datasplitter,
    #     optim_class = torch.optim.SGD,
    #     optim_args = {'lr' : 0.01},
    #     criteriorator = BasicCriteriorator(roll_loss_from_fpr(0.8), 300),
    #     n_episodes = 5),
    ]

logging.info('Starting experiment!')

run_configurations(run_dir, configurations, dataset)

logging.info('Script completed!')
