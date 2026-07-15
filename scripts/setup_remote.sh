#!/usr/bin/env bash
# One-time setup for the remote Mac experiment runner.
# Run this once before using run_remote.sh.
#
# Prerequisites (local):
#   sshpass installed: sudo apt install sshpass
#   OR run ssh-copy-id manually first, then re-run with --skip-key-copy
#
# Usage:
#   ./scripts/setup_remote.sh [--skip-key-copy]

set -euo pipefail

REMOTE_USER="chenzakobar"
REMOTE_HOST="192.168.1.190"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
REMOTE_PASSWORD="Ella9435"
SKIP_KEY_COPY=0

for arg in "$@"; do
    case "$arg" in
        --skip-key-copy) SKIP_KEY_COPY=1 ;;
    esac
done

# ── SSH key setup ──────────────────────────────────────────────────────────────
if [ "$SKIP_KEY_COPY" -eq 0 ]; then
    echo "==> Copying SSH public key to remote (requires sshpass)..."
    if ! command -v sshpass &>/dev/null; then
        echo "sshpass not found. Install it with: sudo apt install sshpass"
        echo "Or copy your key manually: ssh-copy-id $REMOTE"
        echo "Then re-run with: ./scripts/setup_remote.sh --skip-key-copy"
        exit 1
    fi
    # Ensure local key exists
    if [ ! -f ~/.ssh/id_rsa.pub ] && [ ! -f ~/.ssh/id_ed25519.pub ]; then
        echo "No SSH public key found. Generating one..."
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
    fi
    PUB_KEY=$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub)
    sshpass -p "$REMOTE_PASSWORD" ssh -o StrictHostKeyChecking=no "$REMOTE" \
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$PUB_KEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    echo "    SSH key copied."
fi

# ── Remote bootstrap ───────────────────────────────────────────────────────────
echo "==> Bootstrapping remote machine..."
ssh -i ~/.ssh/roll_remote -o IdentitiesOnly=yes -o StrictHostKeyChecking=no "$REMOTE" bash << 'REMOTE_SCRIPT'
set -euo pipefail

# Find python3
PYTHON=$(command -v python3.11 || command -v python3.10 || command -v python3.9 || command -v python3 || true)
if [ -z "$PYTHON" ]; then
    echo "ERROR: python3 not found on remote. Install Python 3 (e.g. via Homebrew: brew install python3)"
    exit 1
fi
PYTHON_VER=$("$PYTHON" --version 2>&1)
echo "  Using Python: $PYTHON ($PYTHON_VER)"

# Create venv
echo "==> Creating venv at ~/roll-venv..."
"$PYTHON" -m venv ~/roll-venv
source ~/roll-venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
echo "==> Installing Python packages..."
pip install \
    numpy scipy torch torchvision plotly scikit-learn matplotlib \
    pandas KDEpy "libauc==2.0.1" \
    --quiet

# adult-dataset pins numpy<2 in its metadata which has no py3.13 wheel;
# install without deps so it inherits whatever numpy we just installed.
pip install adult-dataset --no-deps --quiet

# keel_ds is GitHub-only; use zip URL to avoid needing git/Xcode CLT
echo "==> Installing keel_ds from GitHub..."
pip install "https://github.com/maicondallg/KeelDS/archive/918f000.zip" --quiet

echo "==> Creating dataset directory structure..."
mkdir -p ~/roll-datasets/bank-marketing
mkdir -p ~/roll-datasets/higgs

# Create env.sh for dataset paths
cat > ~/roll-env.sh << 'ENVFILE'
# Dataset env vars for roll experiments (replaces Nix shellHook)
# KEEL datasets: no env vars needed — keel_ds downloads them automatically.
# Forest Cover: auto-resolved via ~/.data/forestcover/cover.mat (expanduser)
# CIFAR-10: auto-downloaded by torchvision to ~/.data/cifar10

export uci_bank_additional_dir="$HOME/roll-datasets/bank-marketing"
export uci_higgs_dir="$HOME/roll-datasets/higgs"
export credit_card_fraud_dir="$HOME/.data/creditcard"
export home_credit_dir="$HOME/.data/homecredit"
ENVFILE

echo ""
echo "✓ Setup complete."
echo ""
echo "──────────────────────────────────────────────────────────────────────"
echo "Dataset notes:"
echo ""
echo "  KEEL datasets     — downloaded automatically by keel_ds on first run"
echo "  CIFAR-10          — downloaded automatically by torchvision on first run"
echo "  Forest Cover      — place cover.mat at: ~/.data/forestcover/cover.mat"
echo "                      (scipy mat file, ~75 MB — download from UCI or use scp)"
echo "  Bank Marketing    — download from: https://archive.ics.uci.edu/dataset/222/bank+marketing"
echo "                      extract CSVs into: ~/roll-datasets/bank-marketing/"
echo "  HIGGS             — download from: https://archive.ics.uci.edu/dataset/280/higgs"
echo "                      place HIGGS.csv.gz into: ~/roll-datasets/higgs/"
echo "  Credit Card Fraud — Kaggle download → ~/.data/creditcard/creditcard.csv"
echo "  Home Credit       — Kaggle download → ~/.data/homecredit/application_train.csv"
echo "──────────────────────────────────────────────────────────────────────"
REMOTE_SCRIPT

echo ""
echo "Remote setup complete. You can now run experiments with:"
echo "  ./scripts/run_remote.sh experiments/keel/experiment-pima.py"
