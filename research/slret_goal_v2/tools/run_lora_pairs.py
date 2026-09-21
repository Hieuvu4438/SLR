"""Bounded sequential C04/fusion-control replication; no TEST or seed selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/slret_goal/tools'))
from extraction_resume import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--seconds-per-job',type=int,default=900)
    args=parser.parse_args()
    assert 1 <= args.seconds_per_job <= 900
    out=ROOT/'artifacts/slret_goal_v2'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    jobs=[(seed,policy,f'seds-{policy}-seed{seed}-001')
          for seed in (1337,2026) for policy in ('fusion','lora')]
    for _,_,run in jobs:
        assert not (out.parent/run).exists(),f'Refuse duplicate {run}'
    runner=ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'
    sources=[runner,*sorted((ROOT/'methods/seds_adaptation').glob('*.py'))]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    start=time.time()
    report=dict(run_id=args.run_id,status='running',pid=os.getpid(),command=sys.argv,
                jobs=[dict(seed=s,policy=p,run_id=r) for s,p,r in jobs],completed=[],
                source_sha256=hashes,test_used=False,start_unix=start)
    ledger=ROOT/'research/slret_goal_v2/EXPERIMENTS.jsonl'

    def record(terminal=False):
        report['wall_seconds']=time.time()-start
        atomic_json(out/'run.json',report)
        if terminal:
            with ledger.open('a') as f:
                f.write(json.dumps(report)+'\n')
    record(True)
    try:
        for seed,policy,run in jobs:
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),'Live source drift'
            command=['timeout','--signal=TERM','--kill-after=10s',f'{args.seconds_per_job}s',
                '/home/haipd/miniconda3/envs/seds/bin/python','-m','torch.distributed.run',
                '--standalone','--nproc_per_node=1',str(runner),'--run-id',run,
                '--aux-weight','1','--policy',policy,'--sign-lr','1e-5','--lora-lr','1e-4',
                '--epochs','3','--seed',str(seed)]
            with (out/f'{run}.log').open('xb') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                report['current']=dict(run_id=run,child_pid=child.pid,command=command,start_unix=time.time())
                record()
                returncode=child.wait()
            child_report=out.parent/run/'run.json'
            result=json.loads(child_report.read_text()) if child_report.exists() else {}
            report['completed'].append(dict(run_id=run,returncode=returncode,status=result.get('status'),
                wall_seconds=result.get('wall_seconds'),selection=result.get('selection')))
            record()
            if returncode != 0 or result.get('status') != 'completed':
                raise RuntimeError(f'{run} failed; stop queue, no automatic retry')
        report.update(status='completed',exit_status=0,current=None)
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        record(True)


if __name__=='__main__':
    main()
