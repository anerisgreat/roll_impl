#!/usr/bin/env bash
# Run an experiment script on the remote Mac.
#
# Usage:
#   ./scripts/run_remote.sh <experiment-script> [--no-fetch] [--detach]
#
# Examples:
#   ./scripts/run_remote.sh experiments/keel/experiment-pima.py
#   ./scripts/run_remote.sh experiments/other/experiment-gaussian.py --no-fetch
#   ./scripts/run_remote.sh experiments/other/experiment-cifar10-imbalanced.py --detach
#   ./scripts/run_remote.sh experiments/keel/run_all.sh
#
# --detach: fire-and-forget. The experiment runs under nohup on the remote and
#           survives SSH disconnects and Mac sleep. Output goes to ~/roll-impl/run.log.
#           Use ./scripts/tail_remote.sh to follow progress, fetch_results.sh when done.

set -euo pipefail

REMOTE_USER="chenzakobar"
REMOTE_HOST="192.168.1.190"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
SSH_KEY="$HOME/.ssh/roll_remote"
SSH_OPTS="-i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
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
    echo "  --detach             run in background on remote (survives sleep/disconnect)"
    exit 1
fi

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Rsync code to remote ───────────────────────────────────────────────────────
echo "==> Syncing code to $REMOTE:$REMOTE_DIR ..."
rsync -avz --delete \
    -e "ssh $SSH_OPTS" \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='results/' \
    --exclude='poison/' \
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
    ssh $SSH_OPTS "$REMOTE" \
      "source ~/roll-venv/bin/activate && { source ~/roll-env.sh 2>/dev/null || true; } && cd $REMOTE_DIR && nohup $RUNNER '$SCRIPT' > $LOG 2>&1 &"
    echo "==> Detached. Experiment running on remote; output → $LOG"
    echo "    Follow: ./scripts/tail_remote.sh"
    echo "    Fetch when done: ./scripts/fetch_results.sh"
else
    ssh $SSH_OPTS "$REMOTE" \
      "source ~/roll-venv/bin/activate && { source ~/roll-env.sh 2>/dev/null || true; } && cd $REMOTE_DIR && $RUNNER '$SCRIPT'"

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "==> Experiment finished."

    if [ "$FETCH_RESULTS" -eq 1 ]; then
        echo "==> Fetching results..."
        rsync -avz --ignore-existing \
            -e "ssh $SSH_OPTS" \
            "$REMOTE:$REMOTE_DIR/results/" "$LOCAL_DIR/results/"
        rsync -avz --ignore-existing \
            -e "ssh $SSH_OPTS" \
            "$REMOTE:$REMOTE_DIR/poison/" "$LOCAL_DIR/poison/" 2>/dev/null || true
        echo "==> Results synced to $LOCAL_DIR/results/"
    else
        echo "(skipping result fetch — run ./scripts/fetch_results.sh when ready)"
    fi
fi
