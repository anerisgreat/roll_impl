import torch
from torch import nn


class ConvNet(nn.Module):
    """Small CNN for RGB images.

    Three conv blocks (Conv→BN→ReLU→MaxPool2d) reduce spatial dims by 8×,
    then two FC layers → scalar logit.

    image_size=32 (CIFAR): 64*4*4=1024 features.
    image_size=64 (ANIMAL-10N, Food-101N): 64*8*8=4096 features.
    """
    def __init__(self, image_size=32):
        super().__init__()
        feature_size = 64 * (image_size // 8) ** 2
        block = lambda c_in, c_out: [
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ]
        self._layers = nn.Sequential(
            *block(3, 32),
            *block(32, 64),
            *block(64, 64),
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self._layers(x).squeeze(1)


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
