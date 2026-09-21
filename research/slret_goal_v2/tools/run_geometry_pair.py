"""Registered C09 smoke→XY→XYZ, bounded, identical recipes, no auto retry."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/slret_goal/tools'))
from extraction_resume import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-fusion',action='store_true')
    args=parser.parse_args()
    suffix='frozen-' if args.freeze_fusion else ''
    out=ROOT/f'artifacts/slret_goal_v2/c09-geometry-{suffix}pair-001'
    prefix='seds-geometry-'+suffix
    jobs=[(prefix+'smoke-001','xyz',240,True),
          (prefix+'xy-001','xy',900,False),
          (prefix+'xyz-001','xyz',900,False)]
    for name,_,_,_ in jobs:
        if (out.parent/name).exists():raise FileExistsError(name)
    out.mkdir(exist_ok=False)
    started=time.time()
    sources=[Path(__file__),ROOT/'research/slret_goal_v2/tools/train_seds_extended.py',
        ROOT/'research/slret_goal/tools/seds_runtime.py',ROOT/'third_party/SEDS/main_task_retrieval.py',
        *sorted((ROOT/'methods/seds_adaptation').glob('*.py'))]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report=dict(status='running',run_id=out.name,pid=os.getpid(),start_unix=started,
                source_sha256=hashes,completed=[],test_used=False,freeze_fusion=args.freeze_fusion)
    child=None
    def record():
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
    def terminate(signum,frame):
        if child is not None and child.poll() is None:child.terminate()
        raise TimeoutError('Parent received SIGTERM')
    signal.signal(signal.SIGTERM,terminate)
    record()
    try:
        for name,mode,seconds,smoke in jobs:
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
                raise ValueError('Source drift in matched pair')
            command=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',
                '/home/haipd/miniconda3/envs/seds/bin/python','-m','torch.distributed.run',
                '--standalone','--nproc_per_node=1',str(ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
                '--run-id',name,'--geometry',mode,'--policy','fusion','--aux-weight','1',
                '--batch-size','32','--seed','42','--lr','1e-5','--sign-lr','1e-4','--epochs','8',
                '--early-stop-drop-pp','2']
            if smoke:command.append('--smoke')
            if args.freeze_fusion:command.append('--geometry-freeze-fusion')
            with (out/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                report['current']=dict(run_id=name,pid=child.pid,command=command);record()
                code=child.wait()
            path=out.parent/name/'run.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            report['completed'].append(dict(run_id=name,returncode=code,status=result.get('status'),
                wall_seconds=result.get('wall_seconds'),steps=result.get('steps'),selection=result.get('selection')))
            record()
            if code or result.get('status')!='completed':raise RuntimeError(f'{name} failed; no retry')
            if not (result.get('geometry_output_updated') and result.get('adaptation_roundtrip_passed')
                    and result.get('frozen_buffers_unchanged')):
                raise ValueError('Geometry update/checkpoint/frozen-buffer check failed')
            groups=result['actual_optimizer_groups']
            expected_lrs={'geometry':1e-4} if args.freeze_fusion else {'geometry':1e-4,'fusion':1e-5}
            if {g['adaptation_group']:g['lr'] for g in groups}!=expected_lrs:
                raise ValueError('Learning-rate mismatch')
            if args.freeze_fusion and not (result.get('only_geometry_updated') and result.get('fusion_state_unchanged_final')):
                raise ValueError('Frozen-fusion invariant failed')
            if smoke and result['steps']!=2:raise ValueError('Expected two smoke updates')
        xy=json.loads((out.parent/(prefix+'xy-001')/'run.json').read_text())
        xyz=json.loads((out.parent/(prefix+'xyz-001')/'run.json').read_text())
        if xy['batch_order_sha256']!=xyz['batch_order_sha256'] or xy['geometry_data']!=xyz['geometry_data']:
            raise ValueError('Pair data/order mismatch')
        report.update(status='completed',exit_status=0,
            selected_xyz_minus_xy=xyz['selection']['mean_R1']-xy['selection']['mean_R1'],
            decision='await_user_result_review; no automatic promotion or next experiment')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc());raise
    finally:record()


if __name__=='__main__':main()
