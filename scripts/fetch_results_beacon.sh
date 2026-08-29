#!/usr/bin/env bash
# Fetch results from beacon-wsl after a remote run.

set -euo pipefail

REMOTE_USER="aner"
REMOTE_HOST="192.168.1.200"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
REMOTE_DIR="~/roll-impl"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Fetching results from $REMOTE..."

rsync -avz --ignore-existing \
    -e "ssh" \
    "$REMOTE:$REMOTE_DIR/results/" "$LOCAL_DIR/results/"

rsync -avz --ignore-existing \
    -e "ssh" \
    "$REMOTE:$REMOTE_DIR/results-final/" "$LOCAL_DIR/results-final/" 2>/dev/null || true

echo "==> Done. Results in: $LOCAL_DIR/results/"
