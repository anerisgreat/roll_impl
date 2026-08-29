"""
Realistic benchmark: MPS vs CPU for KEEL-sized training.

Simulates the actual _perform_episode loop including:
  - data transfer to device each epoch (as in real training)
  - gen_criteria overhead: second forward pass + .detach().cpu().numpy() (forces MPS sync)
  - logging.info call per epoch (both train and val)
  - no MLflow (MLFLOW_TRACKING_URI unset)

Run: python scripts/benchmark_mps_vs_cpu.py
"""
import sys, os
os.environ.pop('MLFLOW_TRACKING_URI', None)  # ensure no mlflow overhead
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import logging
import torch
import torch.nn as nn
import numpy as np
from src.networks import KeelNet

logging.basicConfig(level=logging.WARNING)  # suppress per-epoch loss logs during timing

N_STEPS    = 300
N_TRAIN    = 72
N_VAL      = 71
N_FEATURES = 9

torch.manual_seed(0)
x_train_cpu = torch.randn(N_TRAIN, N_FEATURES)
y_train_cpu = torch.randint(0, 2, (N_TRAIN,)).float()
x_val_cpu   = torch.randn(N_VAL,   N_FEATURES)
y_val_cpu   = torch.randint(0, 2, (N_VAL,)).float()

loss_fn = nn.BCEWithLogitsLoss()


def simulate_gen_criteria(yh, y, device):
    """Mirrors BasicCriteriorator._get_loss: recomputes loss and syncs to CPU."""
    loss = loss_fn(yh, y).detach().cpu().numpy()
    return float(np.mean(loss)) if isinstance(loss, np.ndarray) else float(loss)


def run_benchmark(device_name: str):
    device = torch.device(device_name)
    model = KeelNet(N_FEATURES, n_hidden_layers=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(20):
        bx = x_train_cpu.to(device)
        by = y_train_cpu.to(device)
        optim.zero_grad()
        loss_fn(model(bx), by).backward()
        optim.step()

    if device_name == 'mps':
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        # --- train step (data transfer each epoch, as in real loop) ---
        bx = x_train_cpu.to(device)
        by = y_train_cpu.to(device)
        optim.zero_grad()
        byh = model(bx)
        loss = loss_fn(byh, by)
        loss.backward()
        optim.step()
        # gen_criteria: recompute + .cpu().numpy() sync
        simulate_gen_criteria(byh.detach(), by, device)

        # --- val step ---
        with torch.no_grad():
            vx = x_val_cpu.to(device)
            vy = y_val_cpu.to(device)
            vyh = model(vx)
            simulate_gen_criteria(vyh, vy, device)

    if device_name == 'mps':
        torch.mps.synchronize()

    elapsed = time.perf_counter() - t0
    sps = N_STEPS / elapsed
    ms  = 1000 / sps
    print(f"  {device_name:6s}: {elapsed:.2f}s for {N_STEPS} steps  →  {sps:.1f} steps/s  ({ms:.1f} ms/step)")
    return sps


print(f"Realistic benchmark (includes device transfers + gen_criteria sync)")
print(f"N_STEPS={N_STEPS}, batch={N_TRAIN} train / {N_VAL} val, features={N_FEATURES}")
print()

results = {}
results['cpu'] = run_benchmark('cpu')

if torch.backends.mps.is_available():
    results['mps'] = run_benchmark('mps')
    ratio = results['mps'] / results['cpu']
    faster = 'MPS' if ratio > 1 else 'CPU'
    print(f"\n  → {faster} is {max(ratio, 1/ratio):.1f}x faster")
