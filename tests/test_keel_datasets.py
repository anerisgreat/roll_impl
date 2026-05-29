"""
Smoke-tests for all KEEL datasets mapped via flake.nix shellHook.

Each dataset is expected to have a corresponding env var:
  keel_<name_with_dashes_as_underscores>_dir  →  path to the .dat file

Run from the project root inside the nix devshell:
  python tests/test_keel_datasets.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import KeelDataset

# Mirrors keelDerivationsToHash keys in flake.nix
KEEL_DATASET_NAMES = [
    "wisconsin",
    "pima",
    "iris0",
    "haberman",
    "vehicle2",
    "new-thyroid1",
    "yeast3",
    "vowel0",
    "led7digit-0-2-4-5-6-7-8-9_vs_1",
    "ecoli-0-1_vs_5",
    "cleveland-0_vs_4",
    "glass4",
    "page-blocks-1-3_vs_4",
    "glass0",
    "glass1",
    "glass2",
    "glass5",
    "glass6",
]


def env_var_for(name: str) -> str:
    return "keel_" + name.replace("-", "_") + "_dir"


def check_env_var(name: str) -> str:
    """Returns the path if set, raises if missing."""
    var = env_var_for(name)
    path = os.getenv(var)
    if path is None:
        raise EnvironmentError(f"env var {var!r} is not set — run inside nix devshell")
    return path


def check_file_readable(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"file does not exist: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"file is empty: {path}")
    with open(path, "r") as f:
        header = f.read(256)
    if "@relation" not in header.lower():
        raise ValueError(f"file does not look like a KEEL .dat file: {path}")


def check_loads_as_dataset(name: str) -> tuple:
    """Instantiates KeelDataset and returns (n_samples, n_features)."""
    ds = KeelDataset(name)
    n = len(ds)
    if n == 0:
        raise ValueError(f"dataset {name!r} loaded but has 0 samples")
    x, y = ds[0]
    return n, x.shape[-1]


def run_tests():
    passed = []
    failed = []

    for name in KEEL_DATASET_NAMES:
        var = env_var_for(name)
        print(f"  {name:<45}", end="", flush=True)
        try:
            path = check_env_var(name)
            check_file_readable(path)
            n_samples, n_features = check_loads_as_dataset(name)
            print(f"OK  ({n_samples} samples, {n_features} features)")
            passed.append(name)
        except Exception as e:
            print(f"FAIL")
            print(f"    {type(e).__name__}: {e}")
            failed.append((name, e))

    print()
    print(f"Results: {len(passed)}/{len(KEEL_DATASET_NAMES)} passed")

    if failed:
        print("\nFailed datasets:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    print(f"Testing {len(KEEL_DATASET_NAMES)} KEEL datasets...\n")
    run_tests()
