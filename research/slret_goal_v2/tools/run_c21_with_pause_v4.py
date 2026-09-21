"""Pause the registered UniFormerV2 extraction, run C21, always resume it."""
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

TARGET_PID = 3493585
TARGET_PGID = 3493571
EXPECTED = ('/home/haipd/miniconda3/envs/seds/bin/python','-u',
            'scripts/extract_ph_uniformerv2.py','--splits','test','dev','train',
            '--stride','1','--batch-size','8','--device','cuda:0')
JOB = 'v4-c21-geometry-canonical-motion-003'
OUT = ROOT/'artifacts/slret_goal/jobs'/JOB
REPORT = OUT/'pause_contract.json'
CANDIDATE_ENTRY = ROOT/'research/slret_goal_v2/tools/run_c21_canonical_motion_v4.py'
WRAPPER_ENTRY = Path(__file__).resolve()


def proc_stat(pid):
    fields = (Path('/proc')/str(pid)/'stat').read_text().split()
    return dict(state=fields[2],start_ticks=int(fields[21]),pgid=int(fields[4]))


def cmdline(pid):
    return tuple(x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x)


def group_members(pgid):
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if os.getpgid(int(path.name)) == pgid:
                members.append(int(path.name))
        except (ProcessLookupError,PermissionError):
            pass
    return sorted(members)


def wait_state(stopped,timeout=5):
    deadline=time.time()+timeout
    while time.time()<deadline:
        try: state=proc_stat(TARGET_PID)['state']
        except FileNotFoundError:return False,None
        if (state in ('T','t')) == stopped:
            return True,state
        time.sleep(.1)
    return False,proc_stat(TARGET_PID)['state']


def validate_target(expected_start=None):
    path=Path('/proc')/str(TARGET_PID)
    if not path.is_dir() or path.stat().st_uid != os.getuid():
        raise RuntimeError('Pause target absent or not owned by current user')
    stat=proc_stat(TARGET_PID)
    if stat['pgid'] != TARGET_PGID or cmdline(TARGET_PID) != EXPECTED:
        raise RuntimeError('Pause target identity changed')
    if expected_start is not None and stat['start_ticks'] != expected_start:
        raise RuntimeError('Pause target PID was reused')
    return stat


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    stat=validate_target()
    if stat['state'] in ('T','t'):
        raise RuntimeError('Target was already stopped before C21 wrapper')
    OUT.mkdir(parents=True,exist_ok=False)
    report=dict(status='initializing',target_pid=TARGET_PID,target_pgid=TARGET_PGID,
        target_start_ticks=stat['start_ticks'],target_command=list(EXPECTED),
        group_members_before=group_members(TARGET_PGID),start_unix=time.time(),
        c21_command=['/home/haipd/miniconda3/envs/seds/bin/python','-u',
                     str(CANDIDATE_ENTRY)],
        resumed=False,resume_verified=False,test_used=False)
    atomic_json(REPORT,report)
    paused=False
    child=None
    def interrupted(sig,frame):
        raise InterruptedError(f'pause wrapper received signal{sig}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        # Stop only the GPU worker. Stopping the foreground shell's complete
        # process group lets its controlling shell observe a stopped job and can
        # cause an unsolicited SIGCONT.
        os.kill(TARGET_PID,signal.SIGSTOP)
        paused=True
        ok,state=wait_state(True)
        if not ok:
            raise RuntimeError(f'Could not verify stopped target; state={state}')
        report.update(status='target_paused',pause_unix=time.time(),pause_state=state,
                      group_members_paused=group_members(TARGET_PGID))
        atomic_json(REPORT,report)
        with (OUT/'pause_wrapper.log').open('xb') as log:
            child=subprocess.Popen(report['c21_command'],cwd=ROOT,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report.update(status='c21_running',c21_supervisor_pid=child.pid)
            atomic_json(REPORT,report)
            code=child.wait()
        report.update(c21_returncode=code,status='c21_finished')
        if code:
            raise RuntimeError('C21 supervisor returned nonzero')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        # The C21 supervisor owns and cleans its complete descendant tree on
        # SIGTERM.  Finish that cleanup before releasing the paused extractor.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            try:
                os.kill(child.pid,signal.SIGTERM)
                child.wait(timeout=15)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                report['c21_cleanup_error']='C21 supervisor ignored SIGTERM; forcing exit'
                try: os.kill(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait()
        if paused:
            try:
                validate_target(stat['start_ticks'])
                os.kill(TARGET_PID,signal.SIGCONT)
                ok,state=wait_state(False)
                report.update(resumed=True,resume_verified=ok,resume_state=state,
                              resume_unix=time.time(),group_members_resumed=group_members(TARGET_PGID))
                if not ok:
                    report['resume_error']=f'SIGCONT sent but target state remained {state}'
            except BaseException:
                report['resume_error']=traceback.format_exc()
        report['end_unix']=time.time()
        atomic_json(REPORT,report)


def launch():
    if OUT.exists():
        raise FileExistsError(OUT)
    command=[sys.executable,'-u',str(WRAPPER_ENTRY)]
    log=ROOT/'artifacts/slret_goal/jobs'/f'{JOB}-pause-launch.log'
    with log.open('xb') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,command=command,log=str(log),pause_contract=str(REPORT))))


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        launch()
    elif sys.argv[1:]:
        raise SystemExit('Only --launch is accepted')
    else:
        main()
