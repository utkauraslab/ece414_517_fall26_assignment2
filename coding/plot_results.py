"""Plot one exploration configuration across one or more seeds."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csvs', nargs='+')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    curves = []
    for filename in args.csvs:
        data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None, encoding='utf-8')
        scores = np.atleast_1d(data['score'])
        if len(scores) % 100 or len(scores) < 100:
            raise ValueError('Use training runs with a positive multiple of 100 episodes.')
        curves.append(scores.reshape(-1, 100).mean(axis=1))
    if len({len(c) for c in curves}) != 1:
        raise ValueError('All runs must have equal length.')
    curves = np.asarray(curves)
    x = 100*np.arange(1, curves.shape[1]+1)
    mean = curves.mean(axis=0)
    plt.plot(x, mean, label=f'Mean across {len(curves)} seed(s)')
    if len(curves) > 1:
        sd = curves.std(axis=0, ddof=1)
        plt.fill_between(x, mean-sd, mean+sd, alpha=.2, label='Across-seed sample SD')
    plt.xlabel('Training episode (end of 100-episode block)')
    plt.ylabel('Mean episode length')
    plt.legend()
    plt.grid(alpha=.2)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)


if __name__ == '__main__':
    main()
