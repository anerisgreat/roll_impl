import torch
from torch import nn


class KeelNet(nn.Module):
    """
    Parameterized MLP for KEEL imbalanced datasets.

    Builds: [Dropout?] → Linear(input, hidden) → (ReLU → Linear(hidden, hidden)) × (n_hidden_layers-1)
            → ReLU → Linear(hidden, 1)

    n_hidden_layers=3 matches the glass0/glass1 architecture.
    n_hidden_layers=4, dropout_p=0.5 matches the wisconsin architecture.
    hidden_size defaults to input_size; override when they differ (e.g. banana: input=2, hidden=9).
    """
    def __init__(self, input_size, n_hidden_layers=3, hidden_size=None, dropout_p=0.0):
        super().__init__()
        hidden_size = hidden_size if hidden_size is not None else input_size
        layers = []
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.flatten(self._layers(x))
