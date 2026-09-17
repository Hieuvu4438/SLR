"""Route experiment stdout/stderr to a durable file, independent of tool pipes."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import runpy
import sys

from .common import ART


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(42, 1337, 2026), required=True)
    parser.add_argument('--condition', choices=('on', 'off'), required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    suffix = f'-attempt{args.attempt}' if args.attempt > 1 else ''
    log = ART/f'AS-C40-s{args.seed}-{args.condition}{suffix}-console.log'
    # Exclusive creation: preserve every failed/successful attempt's console.
    with log.open('x', buffering=1) as handle, redirect_stdout(handle), redirect_stderr(handle):
        sys.argv = ['augmentation_training_probe', '--seed', str(args.seed), '--condition', args.condition,
                    '--attempt', str(args.attempt)]
        runpy.run_module('methods.information_probe.augmentation_training_probe', run_name='__main__')


if __name__ == '__main__':
    main()
