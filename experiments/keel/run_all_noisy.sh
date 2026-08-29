#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOISE_RATE="${1:-0.05}"
N_EPISODES="${2:-15}"
RESULTS_DIR="${3:-noise}"
CPU_FLAG="${4:-}"   # pass "--cpu" as 4th arg to force CPU

datasets=(
    glass0
    glass1
    glass2
    glass6
    haberman
    iris0
    new-thyroid1
    pima
    vehicle2
    vowel0
    wisconsin
    yeast3
)

total=${#datasets[@]}
for i in "${!datasets[@]}"; do
    ds="${datasets[$i]}"
    echo "=== [$(( i + 1 ))/$total] $ds (noise ${NOISE_RATE}) ==="
    python "$SCRIPT_DIR/run_noisy.py" "$ds" --noise-rate "$NOISE_RATE" --n-episodes "$N_EPISODES" \
        --results-dir "$RESULTS_DIR" --no-mp $CPU_FLAG \
        || echo "!!! FAILED: $ds, continuing..."
done

echo "=== All done ==="
