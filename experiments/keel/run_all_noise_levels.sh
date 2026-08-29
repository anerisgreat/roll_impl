#!/usr/bin/env bash
# Run KEEL suite at 5%, 10%, and 20% label noise sequentially.
# Results land in results-final/keel-noise-{05,10,20}/.
#
# Usage:
#   ./scripts/run_remote.sh experiments/keel/run_all_noise_levels.sh --detach

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_EPISODES=10
CPU_FLAG="--cpu"    # CPU is 6.6x faster than MPS for KEEL-sized batches (72 samples)
unset MLFLOW_TRACKING_URI  # MLflow per-epoch writes dominate tiny-dataset wall time

for noise_pct in 05 10 20; do
    noise_rate="0.${noise_pct}"
    results_dir="results-final/keel-noise-${noise_pct}"
    echo ""
    echo "======================================================================"
    echo "=== KEEL noise=${noise_rate} → ${results_dir}"
    echo "======================================================================"
    bash "$SCRIPT_DIR/run_all_noisy.sh" "$noise_rate" "$N_EPISODES" "$results_dir" "$CPU_FLAG"
done

echo ""
echo "=== All noise levels complete ==="
