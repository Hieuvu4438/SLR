"""Read-only CPU moment-state census of provenance-locked local baselines."""
import argparse
import json
import sys
import time
import traceback

import torch

from extraction_resume import atomic_json
from inventory import ROOT,sha


def count_moments(states):
    counts={}
    for state in states.values():
        names=('next_m','next_v') if 'next_m' in state else ('exp_avg','exp_avg_sq')
        m,v=(state[n] for n in names)
        if m.shape!=v.shape or m.dtype!=v.dtype:
            raise ValueError('Moment dtype/shape mismatch')
        row=counts.setdefault(str(m.dtype),dict(tensors=0,elements=0,nonzero_m=0,
            nonzero_m_zero_v=0,nonfinite_m=0,nonfinite_v=0))
        row['tensors']+=1
        row['elements']+=m.numel()
        row['nonzero_m']+=int((m!=0).sum())
        row['nonzero_m_zero_v']+=int(((m!=0)&(v==0)).sum())
        row['nonfinite_m']+=int((~torch.isfinite(m)).sum())
        row['nonfinite_v']+=int((~torch.isfinite(v)).sum())
    for row in counts.values():
        row['fraction_all']=row['nonzero_m_zero_v']/row['elements']
        row['fraction_nonzero_first']=row['nonzero_m_zero_v']/row['nonzero_m'] if row['nonzero_m'] else None
    return counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    start=time.time()
    report=dict(run_id=cli.run_id,status='running',gpu_used=False,test_loaded=False,
                optimizer_updates=0,script_sha256=sha(__file__),command=sys.argv,checkpoints={})
    try:
        torch.set_num_threads(4)
        sources=[]
        for dataset in ['ph','csl']:
            root=ROOT/f'runs/{dataset}_base_b512_s42'
            selector=json.loads((root/'selection.json').read_text())
            sources.append((f'cico_{dataset}',root/selector['checkpoint'],selector['checkpoint_sha256'],
                            'optimizer','local_AdamW',root/'selection.json'))
        up=ROOT/'runs/method1/ph/base/seed42/best_dev.pt'
        upprovenance=ROOT/'artifacts/slret_goal/upret-load-check-002/run.json'
        upreport=json.loads(upprovenance.read_text())
        assert upreport['status']=='completed'
        sources.append(('upret_ph_partial',up,upreport['checkpoint_sha256'],
                        'optimizer_state_dict','BertAdam_FP32_wrapper',upprovenance))
        # Existing paired audit supplies recorded terminal-checkpoint hashes.
        auditpath=ROOT/'artifacts/slret_goal/seds-pair-audit-001/run.json'
        audit=json.loads(auditpath.read_text())
        assert audit['status']=='completed'
        final=audit['checkpoint_checks']['222']
        for name,path,expected in zip(['seds_native','seds_fp32'],final['paths'],final['sha256']):
            sources.append((name,ROOT/path,expected,'optimizer','BertAdam',auditpath))
        for name,path,expected,key,kind,provenance in sources:
            actual=sha(path)
            assert actual==expected,('Checkpoint identity mismatch',name)
            data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            counts=count_moments(data[key]['state'])
            report['checkpoints'][name]=dict(path=str(path),sha256=actual,optimizer=kind,counts=counts,
                provenance_path=str(provenance),provenance_exists=provenance.exists(),
                provenance_sha256=sha(provenance) if provenance.exists() else None,
                stored_step=data.get('completed_steps',data.get('global_step',data.get('step'))),
                fp16_exposure_gt_one_percent=counts.get('torch.float16',{}).get('fraction_all',0)>.01)
            del data
        report['cross_model_dataset_exposure_gate']=all(report['checkpoints'][n]['fp16_exposure_gt_one_percent']
                                                       for n in ['seds_native','cico_ph','cico_csl'])
        report.update(status='completed',exit_status=0,
            decision='Checkpoint exposure only; different stages/optimizers, causal retrieval impact UNKNOWN outside SEDS.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds']=time.time()-start
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__': main()
