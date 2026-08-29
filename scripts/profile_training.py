"""
Profile training loop breakdown: data loading, CPU→device transfer,
forward pass, loss, backward, optimizer step.

Usage:
    python scripts/profile_training.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from functools import partial

from src.datasets import Cifar10NDataset
from src.experiment import ExperimentDataLoader, split_dataset_indeces
from src.utils import get_device

N_BATCHES = 50
WARMUP_BATCHES = 5  # let torch.compile finish JIT-ing before timing


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        block = lambda c_in, c_out: [
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ]
        self._layers = nn.Sequential(
            *block(3, 32), *block(32, 64), *block(64, 64),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 1),
        )
    def forward(self, x):
        return self._layers(x).squeeze(1)


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def mean_ms(times):
    return f"{1000 * sum(times) / len(times):.2f}ms"


def run_batches(loader, model, optimizer, loss_fn, device, n_batches, warmup):
    t_load, t_transfer, t_forward, t_loss, t_backward, t_step = [], [], [], [], [], []
    batch_iter = iter(loader)

    for i in range(n_batches + warmup):
        t0 = time.perf_counter()
        try:
            x_cpu, y_cpu = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            x_cpu, y_cpu = next(batch_iter)
        t1 = time.perf_counter()

        x = x_cpu.to(device); y = y_cpu.to(device); sync(device)
        t2 = time.perf_counter()

        yh = model(x); sync(device)
        t3 = time.perf_counter()

        loss = loss_fn(yh, y); sync(device)
        t4 = time.perf_counter()

        optimizer.zero_grad()
        loss.backward(); sync(device)
        t5 = time.perf_counter()

        optimizer.step(); sync(device)
        t6 = time.perf_counter()

        if i < warmup:
            continue  # discard warmup batches
        t_load.append(t1 - t0)
        t_transfer.append(t2 - t1)
        t_forward.append(t3 - t2)
        t_loss.append(t4 - t3)
        t_backward.append(t5 - t4)
        t_step.append(t6 - t5)

    total = [sum(v) for v in zip(t_load, t_transfer, t_forward, t_loss, t_backward, t_step)]
    return t_load, t_transfer, t_forward, t_loss, t_backward, t_step, total


def print_results(t_load, t_transfer, t_forward, t_loss, t_backward, t_step, total):
    rows = [
        ("data load",      t_load),
        ("CPU→device",     t_transfer),
        ("forward",        t_forward),
        ("loss",           t_loss),
        ("backward",       t_backward),
        ("optim step",     t_step),
        ("TOTAL",          total),
    ]
    total_mean = sum(total) / len(total)
    col_w = max(len(r[0]) for r in rows)
    print(f"  {'Step':<{col_w}}  {'mean':>8}  {'% total':>8}")
    print(f"  {'-'*(col_w+20)}")
    for label, times in rows:
        m = sum(times) / len(times)
        pct = 100 * m / total_mean if label != "TOTAL" else 100.0
        print(f"  {label:<{col_w}}  {1000*m:>7.2f}ms  {pct:>7.1f}%")
    iters_per_sec = 1 / total_mean
    # one epoch = all training batches; estimate mins per episode
    # (approximate: 33K train samples / batch_size batches/epoch, patience ~500 epochs)
    print(f"\n  → {iters_per_sec:.1f} batches/sec  |  total mean {1000*total_mean:.1f}ms/batch")
    return total_mean


def profile(device, batch_size, use_compile):
    tag = f"batch={batch_size}, compile={'yes' if use_compile else 'no'}"
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")

    dataset = Cifar10NDataset(noise_type='clean')
    train_idx, _, _ = split_dataset_indeces(dataset, 0.33, 0.33)
    loader = ExperimentDataLoader(dataset, train_idx, batch_size=batch_size, is_balanced=True)

    model = ConvNet().to(device)
    if use_compile:
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    results = run_batches(loader, model, optimizer, loss_fn, device,
                          n_batches=N_BATCHES, warmup=WARMUP_BATCHES)
    return print_results(*results)


if __name__ == '__main__':
    device = get_device()
    print(f"Device: {device}  |  profiling {N_BATCHES} batches (warmup {WARMUP_BATCHES})")

    totals = {}
    for batch_size in [256, 512]:
        for use_compile in [False, True]:
            key = (batch_size, use_compile)
            totals[key] = profile(device, batch_size, use_compile)

    print(f"\n{'='*60}")
    print("  SUMMARY  (mean ms/batch, speedup vs baseline 256/no-compile)")
    print(f"{'='*60}")
    baseline = totals[(256, False)]
    for (bs, comp), t in totals.items():
        speedup = baseline / t
        print(f"  batch={bs}, compile={'yes' if comp else 'no ':3s}  →  {1000*t:6.1f}ms  ({speedup:.2f}x)")
