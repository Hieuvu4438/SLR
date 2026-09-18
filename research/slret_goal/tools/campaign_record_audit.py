"""Audit run/ledger coverage without altering historical or active reports."""
import argparse
import collections
import json
from pathlib import Path
import sys
import time

from extraction_resume import atomic_json
from inventory import ROOT,sha


def compare_record(report,ledger):
    if ledger is None:
        return dict(status_match=False,ledger_missing=True,field_differences=[],extra_fields=[])
    return dict(status_match=report.get('status')==ledger.get('status'),ledger_missing=False,
        field_differences=[key for key in sorted(set(report)&set(ledger)) if report[key]!=ledger[key]],
        extra_fields=sorted(set(report)^set(ledger)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    cli=parser.parse_args();started=time.time()
    root=ROOT/'artifacts/slret_goal'
    ledgerpath=ROOT/'research/slret_goal/experiments.jsonl'
    ledger_hash=sha(ledgerpath)
    entries=[json.loads(line) for line in ledgerpath.read_text().splitlines() if line.strip()]
    latest={row['run_id']:row for row in entries if 'run_id' in row}
    findings=[]
    for path in sorted(root.glob('*/run.json')):
        record=json.loads(path.read_text())
        item=dict(run_id=path.parent.name,report_path=str(path),report_sha256=sha(path),
            status=record.get('status'),command_present=bool(record.get('command')),
            comparison=compare_record(record,latest.get(path.parent.name)))
        if record.get('status')=='running':
            pid=record.get('pid')
            proc=Path(f'/proc/{pid}/cmdline') if isinstance(pid,int) else None
            try:
                cmdline=proc.read_bytes().replace(b'\x00',b' ').decode() if proc else ''
            except FileNotFoundError:
                cmdline=''
            item['live_process_verified']=bool(cmdline and path.parent.name in cmdline)
            item['pid']=pid
            item['command_line']=cmdline
        findings.append(item)
    if sha(ledgerpath)!=ledger_hash:raise RuntimeError('Ledger changed during snapshot; inspect before retry')
    terminal=[x for x in findings if x['status']!='running']
    mismatches=[x['run_id'] for x in terminal if not x['comparison']['status_match']]
    warnings=[x['run_id'] for x in terminal if x['comparison']['field_differences'] or x['comparison']['extra_fields']]
    out=root/cli.run_id;out.mkdir(exist_ok=False)
    report=dict(run_id=cli.run_id,status='completed',exit_status=0,command=sys.argv,
        script_sha256=sha(__file__),gpu_used=False,test_scores_accessed=False,
        ledger_before_sha256=ledger_hash,ledger_entry_count=len(entries),
        run_count=len(findings),status_counts=dict(collections.Counter(x['status'] for x in findings)),
        terminal_status_mismatches=mismatches,terminal_schema_or_content_warnings=warnings,
        missing_command_records=[x['run_id'] for x in findings if not x['command_present']],
        running_records=[x for x in findings if x['status']=='running'],findings=findings,
        wall_seconds=time.time()-started,
        decision='Coverage audit only: historical summary schemas retained; not scientific validation or per-checkpoint hash audit.')
    atomic_json(out/'run.json',report)
    with ledgerpath.open('a') as f:f.write(json.dumps(report)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['findings','running_records']},indent=2))
    print(json.dumps(report['running_records'],indent=2))


if __name__=='__main__':main()
