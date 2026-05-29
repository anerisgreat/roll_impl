import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from _base import run_keel_experiment

if __name__ == '__main__':
    run_keel_experiment('glass2')
