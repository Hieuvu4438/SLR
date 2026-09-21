"""Registered C11 composition smoke -> pilot. No automatic follow-on trial."""
import hashlib
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
    out=ROOT/'artifacts/slret_goal_v2/c11-gcn-lora-pilot-001'
    jobs=[('seds-gcn-lora-smoke-001',240,True),('seds-gcn-lora-001',960,False)]
    for name,_,_ in jobs:
        if (out.parent/name).exists(): raise FileExistsError(name)
    out.mkdir(exist_ok=False)
    sources=[Path(__file__),ROOT/'research/slret_goal_v2/tools/train_seds_extended.py',
        *sorted((ROOT/'methods/seds_adaptation').glob('*.py')),
        ROOT/'research/slret_goal/tools/seds_runtime.py',ROOT/'third_party/SEDS/main_task_retrieval.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    started=time.time(); child=None
    report=dict(run_id=out.name,status='running',pid=os.getpid(),completed=[],test_used=False,source_sha256=hashes)
    def record():
        report['wall_seconds']=time.time()-started; atomic_json(out/'run.json',report)
    def terminate(signum,frame):
        if child is not None and child.poll() is None: child.terminate()
        raise TimeoutError('Parent received SIGTERM')
    signal.signal(signal.SIGTERM,terminate); record()
    try:
        for name,seconds,smoke in jobs:
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
                raise ValueError('Source drift')
            command=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',
                '/home/haipd/miniconda3/envs/seds/bin/python','-m','torch.distributed.run',
                '--standalone','--nproc_per_node=1',str(ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
                '--run-id',name,'--gcn-lora','--policy','lora','--aux-weight','1','--batch-size','32',
                '--seed','42','--lr','1e-5','--sign-lr','1e-6','--lora-lr','1e-4',
                '--epochs','1','--early-stop-drop-pp','2']
            if smoke: command.append('--smoke')
            with (out/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                report['current']=dict(run_id=name,pid=child.pid,command=command); record()
                code=child.wait()
            path=out.parent/name/'run.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            report['completed'].append(dict(run_id=name,returncode=code,status=result.get('status'),
                wall_seconds=result.get('wall_seconds'),steps=result.get('steps'),selection=result.get('selection')))
            record()
            if code or result.get('status')!='completed': raise RuntimeError(f'{name} failed; no automatic retry')
            if not result.get('gcn_lora_update_checks_passed') or not result.get('lora_updated'):
                raise ValueError('GCN/LoRA/frozen-state checks failed')
            if {g['adaptation_group']:g['lr'] for g in result['actual_optimizer_groups']} != dict(encoder=1e-6,fusion=1e-5,lora=1e-4):
                raise ValueError('LR mismatch')
            if not smoke:
                control=json.loads((out.parent/'seds-masked-control-002/run.json').read_text())
                if result['batch_order_sha256']!=control['batch_order_sha256']:
                    raise ValueError('Matched-control batch order mismatch')
        report.update(status='completed',exit_status=0,decision='await_user_review; no further automatic experiments')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc()); raise
    finally: record()


if __name__=='__main__': main()
