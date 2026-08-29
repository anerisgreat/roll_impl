#!/usr/bin/env bash
# Run an experiment script on beacon-wsl (NixOS, CUDA GPU).
#
# Usage:
#   ./scripts/run_remote_beacon.sh <experiment-script> [--no-fetch] [--detach]
#
# Examples:
#   ./scripts/run_remote_beacon.sh experiments/large/experiment-food101n.py --detach
#
# Uses the YubiKey GPG key via SSH agent (same key as set in ~/.ssh/config).
# Runs inside `nix develop` so CUDA-enabled PyTorch and all deps are available.

set -euo pipefail

REMOTE_USER="aner"
REMOTE_HOST="192.168.1.200"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
REMOTE_DIR="~/roll-impl"
FETCH_RESULTS=1
DETACH=0
SCRIPT=""

for arg in "$@"; do
    case "$arg" in
        --no-fetch) FETCH_RESULTS=0 ;;
        --detach)   DETACH=1 ;;
        *)          SCRIPT="$arg" ;;
    esac
done

if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 <experiment-script> [--no-fetch] [--detach]"
    echo "  <experiment-script>  path relative to project root"
    echo "  --no-fetch           skip rsyncing results back after run"
    echo "  --detach             run in background on remote (survives disconnect)"
    exit 1
fi

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Rsync code to remote ───────────────────────────────────────────────────────
echo "==> Syncing code to $REMOTE:$REMOTE_DIR ..."
rsync -avz --delete \
    -e "ssh" \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='results/' \
    --exclude='results-final/' \
    --exclude='poison/' \
    --exclude='poison-debug/' \
    --exclude='noise/' \
    --exclude='.direnv/' \
    --exclude='datasets/' \
    --exclude='scripts/' \
    "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

# ── Determine runner ───────────────────────────────────────────────────────────
case "$SCRIPT" in
    *.sh) RUNNER="bash" ;;
    *)    RUNNER="python" ;;
esac

# ── Run experiment ─────────────────────────────────────────────────────────────
echo ""
echo "==> Running: $SCRIPT"
echo "──────────────────────────────────────────────────────────────────────"

if [ "$DETACH" -eq 1 ]; then
    LOG="$REMOTE_DIR/run.log"
    ssh "$REMOTE" \
      "cd $REMOTE_DIR && nohup env LD_LIBRARY_PATH=/usr/lib/wsl/lib nix develop --command $RUNNER '$SCRIPT' > $LOG 2>&1 &"
    echo "==> Detached. Experiment running on beacon-wsl; output → $LOG"
    echo "    Follow: ssh $REMOTE 'tail -f $LOG'"
    echo "    Fetch when done: ./scripts/fetch_results_beacon.sh"
else
    ssh "$REMOTE" \
      "cd $REMOTE_DIR && env LD_LIBRARY_PATH=/usr/lib/wsl/lib nix develop --command $RUNNER '$SCRIPT'"

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "==> Experiment finished."

    if [ "$FETCH_RESULTS" -eq 1 ]; then
        echo "==> Fetching results..."
        rsync -avz --ignore-existing \
            -e "ssh" \
            "$REMOTE:$REMOTE_DIR/results/" "$LOCAL_DIR/results/"
        rsync -avz --ignore-existing \
            -e "ssh" \
            "$REMOTE:$REMOTE_DIR/results-final/" "$LOCAL_DIR/results-final/" 2>/dev/null || true
        echo "==> Results synced to $LOCAL_DIR/results/"
    else
        echo "(skipping result fetch — run ./scripts/fetch_results_beacon.sh when ready)"
    fi
fi
