import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from _base import run_keel_experiment

if __name__ == '__main__':
    run_keel_experiment('cleveland-0_vs_4', fprs = (0.1, 0.05), n_hidden_layers = 0, n_episodes = 1, batch_size = 64)

