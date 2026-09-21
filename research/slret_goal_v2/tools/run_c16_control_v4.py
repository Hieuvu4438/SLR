"""One bounded control; local resource watchdog/status, no model polling."""
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
from inventory import sha
from job_resources import ProcessTree

RUN = 'seds-signrep-control-offload-001'
JOB = 'v4-c16-control-001'
BASE = ROOT/'artifacts/slret_goal_v2'
OUT = ROOT/'artifacts/slret_goal/jobs'/JOB
MODE = 'control'
LOSS = 'pointwise'
WEIGHT = .1
REFERENCE = 'seds-signrep-transfer-offload-001'
ARTIFACT_CAP = 36
INPUT_FLAGS = None  # Optional native-subset interface; historical C16 defaults unchanged.
EXTRA_SOURCES = []
COMMAND_OVERRIDE = None
TOTAL_UNITS = 160
HARD_SECONDS = 1800
OUTER_SECONDS = 1850
ESTIMATED_SECONDS = 1200
GPU_QUERY_TIMEOUT_SECONDS = 5
CHECKPOINT_LABEL = 'third_party/SEDS/ckpt/ph_best_model.bin'


def validate(result):
    if result.get('status') != 'completed' or not (BASE/RUN/'last.pt').is_file():
        raise ValueError('Missing successful report or last checkpoint')
    anchor = json.loads((BASE/REFERENCE/'run.json').read_text())
    for key, expected in (('signrep',MODE),('signrep_loss',LOSS),('signrep_weight',WEIGHT)):
        default = {'signrep_loss':'pointwise','signrep_weight':.1}.get(key)
        if result['config'].get(key,default) != expected:
            raise ValueError('Registered intervention mismatch: '+key)
    for key in ('batch_order_sha256','checkpoint_sha256_inherited','signrep_data'):
        if result[key] != anchor[key]:
            raise ValueError('Control mismatch: '+key)
    for key,value in anchor['config'].items():
        if key not in ('run_id','signrep','signrep_loss','signrep_weight','artifact_cap_gib') and result['config'][key] != value:
            raise ValueError('Recipe mismatch: '+key)
    for stream in ('fusion','pose','rgb'):
        for direction in ('T2V','V2T'):
            for metric in ('R1','R5','R10'):
                if abs(result['evaluations']['0'][stream][direction][metric] -
                       anchor['evaluations']['0'][stream][direction][metric]) > 1e-10:
                    raise ValueError('Initial recall mismatch')
    for key in ('masked_update_checks_passed','delta_roundtrip_passed'):
        if not result.get(key):
            raise ValueError('Missing gate: '+key)
    if MODE == 'transfer' and not all(result.get(k) for k in ('signrep_gradient_passed','signrep_head_updated')):
        raise ValueError('Missing transfer gradient/update gate')
    best = Path(result['selection']['checkpoint'])
    if not best.is_file():
        raise ValueError('Selected checkpoint absent')
    if result['selection'].get('checkpoint_sha256') and sha(best)!=result['selection']['checkpoint_sha256']:
        raise ValueError('Selected checksum mismatch')
    delta = result['selection']['mean_R1']-anchor['selection']['mean_R1']
    return -delta if MODE == 'control' else delta


def main():
    if (BASE/RUN).exists():
        raise FileExistsError(RUN)
    start = time.time()
    input_flags = INPUT_FLAGS if INPUT_FLAGS is not None else [
        '--signrep',MODE,'--signrep-loss',LOSS,'--signrep-weight',str(WEIGHT)]
    command = [sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=1',
        str(ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),'--run-id',RUN,
        *input_flags,
        '--artifact-cap-gib',str(ARTIFACT_CAP),
        '--masked-pose','control','--policy','fusion','--aux-weight','1',
        '--batch-size','32','--seed','42','--epochs','10','--lr','1e-5','--sign-lr','1e-6',
        '--early-stop-drop-pp','2','--activation-offload','--allocator-cap-gib','16']
    if COMMAND_OVERRIDE is not None:
        command = list(COMMAND_OVERRIDE)
    state = dict(run_id=JOB,training_run=RUN,stage='initialization',status='STARTING',
        start_time=start,update_time=start,end_time=None,pid=os.getpid(),worker_pid=None,
        completed=0,total=TOTAL_UNITS,exit_code=None,output_path=str(BASE/RUN),
        hard_timeout_seconds=HARD_SECONDS,vram_limit_bytes=20_000_000_000,
        watchdog_trip_bytes=19_000_000_000,allocator_cap_bytes=16*1024**3,
        peak_observed_process_gpu_bytes=0,
        gpu_query_timeout_seconds=GPU_QUERY_TIMEOUT_SECONDS,
        last_gpu_sample_time=None,max_gpu_query_seconds=0)
    child = None
    tree = None
    def record():
        state['update_time']=time.time()
        state['wall_seconds']=time.time()-start
        atomic_json(OUT/'status.json',state)
    def stop_child():
        if tree is not None and tree.members():
            tree.send_signal(signal.SIGTERM)
            try: child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                tree.send_signal(signal.SIGKILL)
                child.wait()
            tree.send_signal(signal.SIGKILL)  # Any remaining known worker, never unrelated PIDs.
    def interrupted(sig,frame):
        state.update(status='CANCELLED',reason=f'signal{sig}')
        raise InterruptedError(state['reason'])
    signal.signal(signal.SIGTERM,interrupted)
    record()
    try:
        sources=[Path(__file__),Path(sys.argv[0]).resolve(),*EXTRA_SOURCES,
            ROOT/'research/slret_goal_v2/tools/job_resources.py',ROOT/'research/slret_goal_v2/tools/train_seds_extended.py',
            ROOT/'research/slret_goal/tools/seds_runtime.py',
            *sorted((ROOT/'methods/seds_adaptation').glob('*.py'))]
        atomic_json(OUT/'launch.json',dict(state,command=command,cwd=str(ROOT),env_name='seds',
            code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
            checkpoint=CHECKPOINT_LABEL,estimated_seconds=ESTIMATED_SECONDS))
        child=subprocess.Popen(command,cwd=ROOT,start_new_session=True,
            stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=subprocess.STDOUT)
        tree=ProcessTree(child.pid)
        state.update(status='RUNNING',worker_pid=child.pid)
        record()
        last_heartbeat=0
        while child.poll() is None:
            query_start=time.monotonic()
            used=tree.gpu_bytes(timeout=GPU_QUERY_TIMEOUT_SECONDS)
            state['last_gpu_sample_time']=time.time()
            state['max_gpu_query_seconds']=max(
                time.monotonic()-query_start,state['max_gpu_query_seconds'])
            state['owned_pids']=sorted(tree.members())
            state['peak_observed_process_gpu_bytes']=max(used,state['peak_observed_process_gpu_bytes'])
            if used>state['watchdog_trip_bytes']:
                state.update(status='FAILED',reason='GPU memory safety trip below20GB')
                stop_child();break
            if time.time()-start>=HARD_SECONDS:
                state.update(status='TIMED_OUT',reason=f'{HARD_SECONDS}s allocation reached')
                stop_child();break
            if time.time()-last_heartbeat>=30:
                path=BASE/RUN/'run.json'
                if path.exists():
                    current=json.loads(path.read_text())
                    state.update(completed=current.get('steps',0),stage='training_or_evaluation',
                        trainer_pid=current.get('pid'))
                record(); print(json.dumps(dict(event='heartbeat',**state)),flush=True)
                last_heartbeat=time.time()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
        state['exit_code']=child.wait()
        path=BASE/RUN/'run.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        state['completed']=result.get('steps',0)
        if state['status']=='RUNNING':
            if state['exit_code']!=0:
                raise RuntimeError(result.get('error','Trainer exited unsuccessfully'))
            if state['peak_observed_process_gpu_bytes'] <= 0:
                raise RuntimeError('GPU telemetry unavailable; cannot validate resource monitoring')
            delta=validate(result)
            state.update(status='COMPLETED',stage='finished',selection=result['selection'],
                reference_run=REFERENCE,evaluations=result['evaluations'],
                peak_cuda_bytes=result.get('peak_cuda_bytes'),
                decision='await user; no automatic refinement')
            state['transfer_minus_control_pp' if MODE=='control' else 'candidate_minus_control_pp']=delta
    except Exception:
        if state['status'] not in ('CANCELLED','TIMED_OUT'):
            state['status']='FAILED'
        state['error']=traceback.format_exc()
    finally:
        stop_child()
        if child is not None: state['exit_code']=child.returncode
        state['end_time']=time.time();record()
        atomic_json(OUT/'summary.json',state)
        print(json.dumps(state),flush=True)
    if state['status']!='COMPLETED':
        raise SystemExit(1)


def launch(entry=None):
    if (BASE/RUN).exists():
        raise FileExistsError(RUN)
    OUT.mkdir(parents=True,exist_ok=False)
    command = ['timeout','--kill-after=10s',f'{OUTER_SECONDS}s',sys.executable,'-u',str(Path(entry or __file__).resolve())]
    with (OUT/'run.log').open('x') as log:
        child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    metadata = dict(pid=child.pid,start_time=time.time(),command=command,
        status_path=str(OUT/'status.json'),log_path=str(OUT/'run.log'))
    atomic_json(OUT/'launcher.json',metadata)
    print(json.dumps(metadata))


if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        launch()
    elif sys.argv[1:]:
        raise SystemExit('Only --launch is accepted')
    else:
        main()
