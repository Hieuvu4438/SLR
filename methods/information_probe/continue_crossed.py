"""Monitor the exact existing extractor, then run the registered matrix once.

Never restarts extraction or retries a failed training command. PID identity,
terminal status and cache validation gate the dependent training job.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT,dump,sha

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


def identity(pid):
    path=Path(f'/proc/{pid}')
    if not path.exists():
        return None
    try:
        fields=(path/'stat').read_text().split()
        if fields[2]=='Z':
            return None
        return {'start_ticks':fields[21],
                'command':(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--extraction-pid',type=int,required=True)
    args=p.parse_args()
    expid='AS-C02-CAMPAIGN'
    destination=OUT/f'{expid}_run.json'
    if destination.exists():
        raise FileExistsError('Campaign already registered; inspect rather than restart')
    observed=identity(args.extraction_pid)
    if not observed or '-m methods.information_probe.cache_raw' not in observed['command']:
        raise RuntimeError('The specified extraction process is not currently live')
    state={'experiment_id':expid,'status':'waiting_for_extraction','pid':os.getpid(),
           'extraction_pid':args.extraction_pid,'extraction_identity':observed,
           'start_unix':time.time(),'code_sha256':sha(__file__),
           'commands':[
               [sys.executable,'-m','methods.information_probe.train_raw_readout'],
               [sys.executable,'-m','methods.information_probe.summarize_raw']],
           'retry_policy':'none; failure stops dependent campaign'}
    dump(destination,state)
    while True:
        now=identity(args.extraction_pid)
        if now is None:
            break
        if now!=observed:
            raise RuntimeError('Extraction PID identity changed; refusing to follow reused PID')
        if time.time()-state['start_unix']>7200:
            state['status']='monitor_deadline_reached_no_training'
            dump(destination,state)
            return
        time.sleep(10)
    extraction=json.loads((OUT/'AS-C02-RAW-CACHE-B32_run.json').read_text())
    if extraction['pid']!=args.extraction_pid or extraction['status']!='completed':
        state['status']='extraction_not_successful_no_training'
        state['extraction_terminal_record']=extraction
        dump(destination,state)
        return
    state['extraction_record_sha256']=sha(OUT/'AS-C02-RAW-CACHE-B32_run.json')
    # The registered training loader independently verifies full IDs, hashes,
    # completeness, pretrained checkpoint and every per-video parity result.
    for number,command in enumerate(state['commands']):
        log=OUT/f'{expid}_stage{number}.log'
        with log.open('x') as handle:
            child=subprocess.Popen(['timeout','1800',*command],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            state.update(status='training' if number==0 else 'summarizing',
                         child_pid=child.pid,child_command=command,log=str(log),stage_start_unix=time.time())
            dump(destination,state)
            print(json.dumps({'status':state['status'],'child_pid':child.pid,'log':str(log)}),flush=True)
            while child.poll() is None:
                time.sleep(10)
            state['last_exit_code']=child.returncode
        if child.returncode!=0:
            state['status']='failed_no_retry'
            dump(destination,state)
            return
    state['status']='completed'
    state['wall_seconds']=time.time()-state['start_unix']
    dump(destination,state)


if __name__=='__main__':
    main()
