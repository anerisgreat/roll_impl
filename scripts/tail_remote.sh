#!/usr/bin/env bash
# Tail the log from a --detach run on the remote Mac.
# Ctrl-C to stop tailing (experiment keeps running).

set -euo pipefail

REMOTE_USER="chenzakobar"
REMOTE_HOST="192.168.1.190"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
SSH_KEY="$HOME/.ssh/roll_remote"
SSH_OPTS="-i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
REMOTE_DIR="~/roll-impl"
LOG="$REMOTE_DIR/run.log"

echo "==> Tailing $REMOTE:$LOG (Ctrl-C to stop) ..."
ssh $SSH_OPTS "$REMOTE" "tail -f $LOG"
