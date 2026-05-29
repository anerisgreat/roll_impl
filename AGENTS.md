# Agent Guidelines for ROLL

## Project Overview

ROLL (Ranking via Optimized Label Learning) is a PyTorch-based machine learning research project implementing loss functions for binary classification with kernel density estimation and ROC optimization.

## Development Environment

### Setup
```bash
# Enter the development environment via Nix
nix develop

# Or with direnv (auto-activates)
direnv allow

# Alternative: pip-based setup
pip install -r requirements.txt
```

### Running Experiments

Each experiment is a standalone script. Run them directly:
```bash
python experiment-gaussian.py
python experiment-adult.py
python experiment-cifar10.py
python experiment-forest.py
python experiment-keel-wisconsin.py
python experiment-keel-yeast.py
```

## Code Style Guidelines

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Variables | `snake_case` | `learning_rate`, `true_indeces` |
| Functions | `snake_case` | `split_true_false()`, `kernelized_roll_fpr()` |
| Classes | `CamelCase` | `BasicCriteriorator`, `ModelDataResult` |
| Constants | `UPPER_SNAKE_CASE` | `N_EPOCHS`, `N_MP_WORKERS` |
| Private members | Leading underscore | `_loss_func`, `_calc_moments()` |

### Import Organization
```python
# Standard library
import torch
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Any

# Third-party
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from KDEpy.bw_selection import improved_sheather_jones

# Local application
from .utils import joinmakedir
from .summary import summarize_episode
```

### Error Handling
- Use exceptions for unexpected errors: `raise ValueError("message")`
- Use `logging` for expected conditions and debugging
- Set appropriate log levels: debug/info/warning/error

### Logging Setup
```python
from src.utils import init_experiment
run_dir = init_experiment('results', 'experiment_name', console_level=logging.DEBUG)
```

## File Structure

```
src/
├── roll.py              # ROLL loss implementations (kernelized, beta, normal)
├── experiment.py        # Training loop, evaluation, data splitting
├── datasets.py          # Dataset loaders (Adult, CIFAR-10, Forest, KEEL, Gaussian, Bank)
├── summary.py           # Visualization (Plotly)
├── utils.py             # Logging, directory utilities
└── beta_dist.py         # Beta distribution for Torch

experiment-*.py          # Standalone experiment scripts
datasets/                # Dataset storage
results/                 # Experiment outputs
```

## Key Components

### Loss Functions (`src/roll.py`)
- `roll_loss_from_fpr(fpr)` - Normal distribution-based ROLL loss
- `roll_beta_loss_from_fpr(fpr)` - Beta distribution-based ROLL loss
- `kernelized_roll_fpr(fpr)` - Kernelized ROLL with custom autograd (KernelizedROLLoss)
- `kernelized_roll_tpr(tpr)` - Kernelized ROLL for TPR optimization

### Experiments (`src/experiment.py`)
- `ExperimentConfiguration` - dataclass for experiment setup
- `Criteriorator` (ABC) - base class for loss/criteria generation
- `BasicCriteriorator` - simple loss-based stopping
- `CRBasedCriteriorator` - criteria with FPR/TPR metrics
- `run_configurations()` - execute multiple experiment configs

### Datasets (`src/datasets.py`)
- `TestGaussianDataset` - synthetic Gaussian data
- `AdultDataset` - Adult income dataset
- `ForestCoverDataset` - Forest cover type
- `Cifar10Dataset` - CIFAR-10 (binary: class 1 vs rest)
- `KeelDataset` - Generic KEEL dataset loader
- `BankMarketingDataset` - UCI Bank Marketing
- `TorchStandardScaler` - PyTorch-compatible standard scaler

### Visualization (`src/summary.py`)
- ROC curve generation with confidence bands
- Score distribution plots
- ROLL CDF visualization

## Common Tasks

### Adding a New Experiment
1. Copy an existing `experiment-*.py` file
2. Modify model architecture, dataset, or loss function
3. Update `ExperimentConfiguration` with parameters
4. Run and save results for comparison

### Adding a New Dataset
1. Add loader class to `src/datasets.py`
2. Implement `__getitem__`, `__len__`, and `x`, `y` attributes
3. Test with existing experiment pattern

### Modifying Loss Functions
1. Core implementations in `src/roll.py`
2. Create partial function wrapper for experiment compatibility
3. Test with multiple FPR values

## Important Notes

- **GPU Support**: Disabled in flake.nix (`cudaSupport = false`). All development uses CPU.
- **Multiprocessing**: Uses `torch.multiprocessing` with spawn start method
- **Dataset Paths**: KEEL/UCI datasets fetched via Nix shellHook exports
- **No Linting/Type Checking**: Write clean, readable code
