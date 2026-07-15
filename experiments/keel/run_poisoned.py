import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from _base import run_keel_experiment

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('n_duplicates', type=int)
    parser.add_argument('--no-mp', action='store_true', help='disable multiprocessing')
    args = parser.parse_args()
    run_keel_experiment(args.dataset, n_poison_duplicates=args.n_duplicates,
                        results_dir='poison', n_episodes=15, is_mp=not args.no_mp)
