import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from _base import run_keel_experiment

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--noise-rate', type=float, default=0.05)
    parser.add_argument('--n-episodes', type=int, default=15)
    parser.add_argument('--results-dir', type=str, default='noise')
    parser.add_argument('--no-mp', action='store_true', help='disable multiprocessing')
    parser.add_argument('--cpu', action='store_true', help='force CPU even if MPS/CUDA available')
    args = parser.parse_args()
    run_keel_experiment(args.dataset, noise_rate=args.noise_rate,
                        results_dir=args.results_dir, n_episodes=args.n_episodes,
                        is_mp=not args.no_mp, force_cpu=args.cpu)
