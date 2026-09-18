"""Fixed sequential replication queue; fail closed, never retry an existing run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from extraction_resume import atomic_json
from inventory import ROOT,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started=time.time()
    report=dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,
        test_loaded=False,steps=[],active_child=None,script_sha256=sha(__file__),
        protocol_sha256=sha(ROOT/'research/slret_goal/CICO_NUMERICAL_REPLICATION_PROTOCOL.md'),
        fixed_pairs=[['ph',1337],['ph',2026],['csl',42],['csl',1337],['csl',2026]])
    paths=[ROOT/'research/slret_goal/tools'/n for n in
           ['cico_numeric_continue.py','cico_fp32_moments.py','cico_numeric_audit.py']]
    report['source_sha256']={str(p):sha(p) for p in paths}
    def record():
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
    record()
    def run(name,script,args,limit):
        assert all(sha(p)==v for p,v in report['source_sha256'].items()),'Live source changed'
        assert time.time()-started+limit+15<7000,'Insufficient remaining registered budget'
        assert not (out.parent/name).exists(),('Run already exists; do not restart',name)
        cmd=['timeout','--signal=TERM','--kill-after=10s',f'{limit}s',sys.executable,
             str(ROOT/'research/slret_goal/tools'/script),'--run-id',name,*args]
        with (out/f'{name}.log').open('xb') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            report['active_child']=dict(run_id=name,timeout_pid=process.pid,command=cmd,start_unix=time.time())
            record()
            print(json.dumps(dict(event='start',**report['active_child'])),flush=True)
            code=process.wait()
        childpath=out.parent/name/'run.json'
        child=json.loads(childpath.read_text()) if childpath.exists() else {}
        report['steps'].append(dict(run_id=name,returncode=code,status=child.get('status'),
                                  wall_seconds=child.get('wall_seconds'),report_sha256=sha(childpath) if childpath.exists() else None))
        report['active_child']=None
        record()
        assert code==0 and child.get('status')=='completed',('Child failed; no auto retry',name,child.get('error'))
        print(json.dumps(dict(event='completed',run_id=name,wall_seconds=child['wall_seconds'])),flush=True)
        return child
    try:
        native='cico-repl-csl-smoke-native-001'
        run(native,'cico_numeric_continue.py',['--dataset','csl','--seed','42','--arm','native','--smoke'],300)
        smoke=run('cico-repl-csl-smoke-fp32-001','cico_numeric_continue.py',
                  ['--dataset','csl','--seed','42','--arm','fp32','--smoke','--reference',native],300)
        assert smoke['peak_cuda_bytes']<47*1024**3
        assert smoke['wall_seconds']<60,'CSL smoke throughput requires revised admission'
        for dataset,seed in report['fixed_pairs']:
            stem=f'cico-repl-{dataset}-s{seed}'
            native=stem+'-native-001'
            fixed=stem+'-fp32-001'
            run(native,'cico_numeric_continue.py',['--dataset',dataset,'--seed',str(seed),'--arm','native'],900)
            run(fixed,'cico_numeric_continue.py',['--dataset',dataset,'--seed',str(seed),'--arm','fp32','--reference',native],900)
            audit=run(stem+'-audit-001','cico_numeric_audit.py',['--native',native,'--fp32',fixed],180)
            print(json.dumps(dict(event='paired_result',dataset=dataset,seed=seed,
                endpoint_delta=audit['endpoint_delta'],accuracy_gate=audit['accuracy_gate'],
                mechanism_replication_gate=audit['mechanism_replication_gate'])),flush=True)
        report.update(status='completed',exit_status=0,decision='Fixed replication queue complete; synthesis and independent confirmation pending.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        record()
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps({k:report.get(k) for k in ['run_id','status','wall_seconds','error']}),flush=True)


if __name__=='__main__':main()
