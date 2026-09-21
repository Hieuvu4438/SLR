"""Bounded public-checkpoint download -> smoke -> admitted extraction, no train."""
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
sys.path.insert(0,str(ROOT))
from research.slret_goal_v2.tools.extract_signrep import atomic_json, make_plan, sha


def main():
    start=time.time();base=ROOT/'artifacts/slret_goal_v2'
    out=base/'c16-signrep-assets-001';out.mkdir(exist_ok=False)
    source=base/'external/SignRep';ckpt=source/'ckpt.pt'
    expected='f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b'
    files=[Path(__file__),ROOT/'research/slret_goal_v2/tools/extract_signrep.py',
           ROOT/'methods/seds_adaptation/geometry_cache.py']
    frozen={str(p):sha(p) for p in files}
    report=dict(run_id=out.name,status='running',pid=os.getpid(),start_unix=start,
        reservation_seconds=3600,test_used=False,retrieval_training=False,completed=[],
        source_sha256=frozen,checkpoint_url='https://github.com/ryanwongsa/SignRep/releases/download/v0.0.1/ckpt.pt',
        expected_checkpoint_sha256=expected,expected_checkpoint_bytes=222676746)
    child=None
    def record():
        report['wall_seconds']=time.time()-start;atomic_json(out/'run.json',report)
    def terminate(sig,frame):
        if child is not None and child.poll() is None:child.terminate()
        raise TimeoutError('Parent hard bound; completed caches retained')
    signal.signal(signal.SIGTERM,terminate)
    def run(name,command,cap):
        nonlocal child
        if any(sha(p)!=h for p,h in frozen.items()):raise ValueError('Live source changed')
        tick=time.time()
        with (out/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=10s',f'{cap}s',*command],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            report['current']=dict(name=name,pid=child.pid,command=command,cap=cap);record()
            code=child.wait()
        entry=dict(name=name,returncode=code,wall_seconds=time.time()-tick)
        report['completed'].append(entry);record()
        if code:raise RuntimeError(f'{name} failed, no automatic retry')
        return entry
    record()
    try:
        if not ckpt.is_file() or ckpt.stat().st_size!=222676746 or sha(ckpt)!=expected:
            run('download',['/home/haipd/.local/bin/aria2c','--continue=true','--allow-overwrite=false',
                '--auto-file-renaming=false','--file-allocation=none','--split=8','--max-connection-per-server=8',
                '--min-split-size=4M','--max-tries=2','--retry-wait=5','--timeout=30',
                '--summary-interval=30','--console-log-level=warn','--download-result=full',
                '--dir='+str(source),'--out=ckpt.pt',report['checkpoint_url']],1500)
        if ckpt.stat().st_size!=222676746 or sha(ckpt)!=expected:raise ValueError('Official checkpoint checksum mismatch')
        report['checkpoint_verified']=True;record()
        py='/home/haipd/miniconda3/envs/seds/bin/python'
        script=str(ROOT/'research/slret_goal_v2/tools/extract_signrep.py')
        smoke='signrep-native-window-smoke-001'
        run(smoke,[py,script,'--run-id',smoke,'--smoke'],180)
        result=json.loads((base/smoke/'run.json').read_text())
        if result['status']!='completed' or not result['strict_checkpoint_loaded']:
            raise ValueError('SignRep smoke failed')
        total=make_plan()['total_clips']
        # Conservative end-to-end smoke wall includes model load; no timing-only
        # GPU reruns and no automatic full-TRAIN expansion.
        estimate=result['wall_seconds']/result['completed_clips']*total*1.25
        remaining=int(3540-(time.time()-start))
        report.update(extraction_estimate_seconds=estimate,remaining_seconds=remaining)
        if estimate>remaining:
            report.update(status='prepared_not_admitted',decision='checkpoint/smoke ready; extraction exceeds current bounded allocation')
            return
        name='signrep-native-window-pilot-001'
        run(name,[py,script,'--run-id',name],remaining)
        result=json.loads((base/name/'run.json').read_text())
        if result['status']!='completed':raise ValueError('Incomplete SignRep extraction')
        report.update(status='completed',decision='await user return; no retrieval training launched')
    except Exception:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:record()


if __name__=='__main__':main()
