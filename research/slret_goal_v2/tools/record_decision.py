"""Append a DEV-driven decision without overwriting the original run report."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--decision',choices=['promote','refine','defer','drop','smoke_pass','repair'],required=True)
    parser.add_argument('--reason',required=True)
    args=parser.parse_args()
    source=ROOT/'artifacts/slret_goal_v2'/args.run_id/'run.json'
    payload=source.read_bytes();report=json.loads(payload)
    if report['status'] not in ('completed','failed'):
        raise ValueError('Do not adjudicate an unfinished run')
    row=dict(event='decision',run_id=args.run_id,candidate=report['candidate'],
             decision=args.decision,reason=args.reason,time_unix=time.time(),
             report_sha256=hashlib.sha256(payload).hexdigest(),
             selection=report.get('selection'),test_used=False)
    with (ROOT/'research/slret_goal_v2/EXPERIMENTS.jsonl').open('a') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row))


if __name__=='__main__':
    main()
