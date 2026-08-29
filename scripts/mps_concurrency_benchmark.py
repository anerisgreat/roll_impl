"""
Benchmark: how many concurrent MPS training processes can the Mac handle?

Launches N independent subprocesses each training a ConvNet on CIFAR-10N for
a fixed number of batches, and measures total wall-clock time. Perfect scaling
= same wall-clock as N=1; GPU saturation = wall-clock grows with N.

Usage:
    python scripts/mps_concurrency_benchmark.py
"""
import sys, os, time, subprocess, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BATCHES_PER_WORKER = 200
CONCURRENCY_LEVELS = [1, 2, 3, 4, 5, 6, 7]
VENV_PYTHON = os.path.expanduser('~/roll-venv/bin/python3')

# ── Worker script (written to a temp file and executed as subprocess) ─────────
WORKER_CODE = '''
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.datasets import Cifar10NDataset
from src.experiment import ExperimentDataLoader, split_dataset_indeces
from src.utils import get_device

N_BATCHES = {n_batches}
BATCH_SIZE = 256

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        block = lambda c_in, c_out: [
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2, 2),
        ]
        self._layers = nn.Sequential(
            *block(3, 32), *block(32, 64), *block(64, 64),
            nn.Flatten(), nn.Linear(64*4*4, 128), nn.ReLU(), nn.Linear(128, 1),
        )
    def forward(self, x): return self._layers(x).squeeze(1)

device = get_device()
dataset = Cifar10NDataset(noise_type='clean')
train_idx, _, _ = split_dataset_indeces(dataset, 0.33, 0.33)
loader = ExperimentDataLoader(dataset, train_idx, batch_size=BATCH_SIZE, is_balanced=True)

model = ConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

batch_iter = iter(loader)
t0 = time.perf_counter()
for i in range(N_BATCHES):
    try:
        x, y = next(batch_iter)
    except StopIteration:
        batch_iter = iter(loader)
        x, y = next(batch_iter)
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

elapsed = time.perf_counter() - t0
print(f"worker_done elapsed={{elapsed:.2f}}")
sys.stdout.flush()
'''


def run_n_workers(n, python_bin, worker_script):
    """Launch n independent subprocesses, return wall-clock seconds until all finish."""
    procs = []
    t0 = time.perf_counter()
    for _ in range(n):
        p = subprocess.Popen(
            [python_bin, worker_script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=os.path.join(os.path.dirname(__file__), '..')
        )
        procs.append(p)

    elapsed_per_worker = []
    for p in procs:
        stdout, stderr = p.communicate()
        for line in stdout.decode().splitlines():
            if line.startswith('worker_done elapsed='):
                elapsed_per_worker.append(float(line.split('=')[1]))
        if not elapsed_per_worker and stderr:
            print(f"\nWorker error:\n{stderr.decode()[:500]}", flush=True)

    wall_clock = time.perf_counter() - t0
    return wall_clock, elapsed_per_worker


def main():
    python_bin = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

    # Write worker script to a temp file inside scripts/
    worker_path = os.path.join(os.path.dirname(__file__), '_bench_worker.py')
    with open(worker_path, 'w') as f:
        f.write(WORKER_CODE.format(n_batches=BATCHES_PER_WORKER))

    print(f"Benchmark: {BATCHES_PER_WORKER} batches/worker, concurrency levels {CONCURRENCY_LEVELS}")
    print(f"Python: {python_bin}\n")

    results = {}
    baseline = None

    for n in CONCURRENCY_LEVELS:
        print(f"N={n}: launching {n} worker(s)...", end='', flush=True)
        wall, per_worker = run_n_workers(n, python_bin, worker_path)
        results[n] = (wall, per_worker)
        if baseline is None:
            baseline = wall
        efficiency = baseline / wall * 100  # % of perfect scaling
        throughput = n * BATCHES_PER_WORKER / wall
        print(f"  wall={wall:.1f}s  per-worker avg={sum(per_worker)/len(per_worker):.1f}s  "
              f"throughput={throughput:.1f} batches/s  efficiency={efficiency:.0f}%")

    os.remove(worker_path)

    print(f"\n{'N':>4}  {'wall (s)':>10}  {'throughput':>12}  {'efficiency':>12}  {'vs N=1':>8}")
    print("-" * 55)
    baseline_wall = results[1][0]
    baseline_tp = BATCHES_PER_WORKER / baseline_wall
    for n, (wall, _) in results.items():
        tp = n * BATCHES_PER_WORKER / wall
        eff = baseline_wall / wall * 100
        speedup = wall / baseline_wall
        print(f"{n:>4}  {wall:>10.1f}  {tp:>11.1f}b/s  {eff:>11.0f}%  {speedup:>7.2f}x slower")


if __name__ == '__main__':
    main()
