#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_DUPLICATES="${1:-3}"

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
    echo "=== [$(( i + 1 ))/$total] $ds (poison x${N_DUPLICATES}) ==="
    python "$SCRIPT_DIR/run_poisoned.py" "$ds" "$N_DUPLICATES" \
        || echo "!!! FAILED: $ds, continuing..."
done

echo "=== All done ==="
