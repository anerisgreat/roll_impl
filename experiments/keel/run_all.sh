#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

experiments=(
    experiment-glass0.py
    experiment-glass1.py
    experiment-glass2.py
    experiment-glass6.py
    experiment-haberman.py
    experiment-iris0.py
    experiment-new-thyroid1.py
    experiment-pima.py
    experiment-vehicle2.py
    experiment-vowel0.py
    experiment-wisconsin.py
    experiment-yeast3.py
)

total=${#experiments[@]}
for i in "${!experiments[@]}"; do
    script="${experiments[$i]}"
    echo "=== [$(( i + 1 ))/$total] $script ==="
    python "$SCRIPT_DIR/$script" || echo "!!! FAILED: $script, continuing..."
done

echo "=== All done ==="
