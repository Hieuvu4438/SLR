"""Launch a bounded local job with persistent log and process provenance."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from inventory import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 1 <= args.seconds <= 9000:
        raise ValueError('Explicit command and bounded timeout required')
    folder = ROOT / 'artifacts/slret_goal/jobs' / args.name
    folder.mkdir(parents=True, exist_ok=False)
    full = ['timeout', '--signal=TERM', '--kill-after=10s', str(args.seconds) + 's', *command]
    with (folder / 'console.log').open('xb') as log:
        process = subprocess.Popen(full, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    report = dict(name=args.name, timeout_pid=process.pid, command=full,
                  start_unix=time.time(), hard_timeout_seconds=args.seconds,
                  log=str(folder / 'console.log'))
    (folder / 'launch.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
