"""One registered SignRep transfer/control contrast; no retries or next methods."""
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/slret_goal/tools'))
from extraction_resume import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offload-retry',action='store_true')
    cli = parser.parse_args()
    base = ROOT/'artifacts/slret_goal_v2'
    out = base/'c16-signrep-transfer-pair-002'
    jobs = [('seds-signrep-transfer-smoke-002','transfer',True,120),
            ('seds-signrep-transfer-001','transfer',False,800),
            ('seds-signrep-control-001','control',False,800)]
    if cli.offload_retry:
        out = base/'c16-signrep-offload-retry-003'
        jobs = [('seds-signrep-offload-smoke-003','transfer',True,150),
                ('seds-signrep-transfer-offload-001','transfer',False,1400)]
    for name,_,_,_ in jobs:
        if (base/name).exists():
            raise FileExistsError(name)
    out.mkdir(exist_ok=False)
    sources = [Path(__file__),ROOT/'research/slret_goal_v2/tools/train_seds_extended.py',
               ROOT/'research/slret_goal/tools/seds_runtime.py',
               *sorted((ROOT/'methods/seds_adaptation').glob('*.py'))]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    started = time.time()
    report = dict(run_id=out.name,status='running',pid=os.getpid(),completed=[],
                  source_sha256=hashes,reservation_seconds=1600 if cli.offload_retry else 1800,
                  test_used=False,offload_retry=cli.offload_retry)
    child = None

    def record():
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json',report)

    def terminate(signum,frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise TimeoutError('Bounded parent terminated')

    signal.signal(signal.SIGTERM,terminate)
    record()
    try:
        for name,arm,smoke,cap in jobs:
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
                raise ValueError('Source changed after registration')
            command = ['timeout','--signal=TERM','--kill-after=10s',f'{cap}s',
                '/home/haipd/miniconda3/envs/seds/bin/python','-m','torch.distributed.run',
                '--standalone','--nproc_per_node=1',str(ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
                '--run-id',name,'--signrep',arm,'--masked-pose','control','--policy','fusion',
                '--aux-weight','1','--batch-size','32','--seed','42','--epochs','10',
                '--lr','1e-5','--sign-lr','1e-6','--early-stop-drop-pp','2']
            if smoke:
                command.append('--smoke')
            if cli.offload_retry:
                command.append('--activation-offload')
            with (out/(name+'.log')).open('xb') as log:
                child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                report['current'] = dict(run_id=name,pid=child.pid,command=command)
                record()
                code = child.wait()
            path = base/name/'run.json'
            result = json.loads(path.read_text()) if path.exists() else {}
            report['completed'].append(dict(run_id=name,returncode=code,status=result.get('status'),
                wall_seconds=result.get('wall_seconds'),steps=result.get('steps'),selection=result.get('selection')))
            record()
            if code or result.get('status') != 'completed':
                raise RuntimeError(f'{name} failed; no automatic retry')
            if not result.get('masked_update_checks_passed') or not result.get('delta_roundtrip_passed'):
                raise ValueError('Native update/buffer/checkpoint gates failed')
            if arm == 'transfer' and not (result.get('signrep_gradient_passed') and result.get('signrep_head_updated')):
                raise ValueError('Transfer gradient or learning inactive')
            expected_step0 = ('training-memory smoke; no DEV evaluation' if cli.offload_retry and smoke
                else 'fresh fullDEV initialization; exact same-batch hook proof; historical recall matched')
            if result['step0'] != expected_step0:
                raise ValueError('Missing fresh initialization / hook proof')
            if cli.offload_retry and smoke:
                if result['peak_cuda_bytes'] > 16*1024**3:
                    raise ValueError('Offload smoke exceeds registered16GiB memory envelope')
        if cli.offload_retry:
            report.update(status='completed',exit_status=0,
                decision='collect C16 pilot; matched offload control not yet run; no promotion')
            return
        transfer = json.loads((base/jobs[1][0]/'run.json').read_text())
        control = json.loads((base/jobs[2][0]/'run.json').read_text())
        for key in ('batch_order_sha256','checkpoint_sha256_inherited','signrep_data'):
            if transfer[key] != control[key]:
                raise ValueError(f'Unmatched paired recipe: {key}')
        for stream in ('fusion','pose','rgb'):
            for direction in ('T2V','V2T'):
                for metric in ('R1','R5','R10'):
                    assert transfer['evaluations']['0'][stream][direction][metric] == control['evaluations']['0'][stream][direction][metric]
        report.update(status='completed',exit_status=0,
            selected_delta_control=transfer['selection']['mean_R1']-control['selection']['mean_R1'],
            decision='collect paired DEV curves; exploratory TRAIN512, no promotion from this alone')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        record()


if __name__ == '__main__':
    main()
