import torch
from torch import nn

from torchvision import datasets, transforms

from src.experiment import _perform_multiple_episodes
from src.experiment import *

from src.datasets import ForestCoverDataset

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            x for y in [
                [nn.Linear(10, 10), nn.ReLU()] for _ in range(10)]
            for x in y]
        self._layers = nn.Sequential(
            *layers, nn.Linear(10, 1))

    def forward(self, x):
        return self._layers(x)

summary_dir = 'tmp'
device = torch.device('cpu')
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset = ForestCoverDataset()

model_creator_func = MyNet

_perform_multiple_episodes(
    summary_dir = summary_dir,
    model_creator_func = model_creator_func,
    dataset = dataset,
    data_splitter = basic_data_splitter,
    optim_class = torch.optim.Adam,
    optim_args = {'lr' : 0.001},
    criteriorator = BasicCriteriorator(torch.nn.BCEWithLogitsLoss(), 1),
    device = device,
    n_episodes = 2)
