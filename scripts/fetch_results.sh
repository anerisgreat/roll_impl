#!/usr/bin/env bash
# Fetch results and poison directories from the remote machine.
# Use this after a manual run, or to pull results without re-running.
#
# Usage:
#   ./scripts/fetch_results.sh

set -euo pipefail

REMOTE_USER="chenzakobar"
REMOTE_HOST="192.168.1.190"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
SSH_KEY="$HOME/.ssh/roll_remote"
SSH_OPTS="-i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
REMOTE_DIR="~/roll-impl"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Fetching results from $REMOTE..."

rsync -avz --ignore-existing \
    -e "ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/results/" "$LOCAL_DIR/results/"

rsync -avz --ignore-existing \
    -e "ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/results-final/" "$LOCAL_DIR/results-final/" 2>/dev/null || true

rsync -avz --ignore-existing \
    -e "ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/poison/" "$LOCAL_DIR/poison/" 2>/dev/null || true

echo "==> Done. Results in: $LOCAL_DIR/results/ and $LOCAL_DIR/results-final/"
