"""CPU checkpoint audit of native Adam moment representability; no intervention."""
import argparse
import json
import time
import torch

from inventory import ROOT, sha
from extraction_resume import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/args.run_id
    out.mkdir(exist_ok=False)
    start=time.time()
    source=ROOT/'artifacts/slret_goal/seds-continuation-control-001/checkpoint_step111.pt'
    source_hash=sha(source)
    checkpoint=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    torch.set_num_threads(4)
    counts={}
    for state in checkpoint['optimizer']['state'].values():
        m,v=state['next_m'],state['next_v']
        key=str(m.dtype)
        row=counts.setdefault(key,dict(tensors=0,elements=0,nonzero_m_zero_v=0,nonzero_m=0,
                                       nonfinite_m=0,nonfinite_v=0))
        row['tensors']+=1
        row['elements']+=m.numel()
        row['nonzero_m_zero_v']+=int(((m!=0)&(v==0)).sum())
        row['nonzero_m']+=int((m!=0).sum())
        row['nonfinite_m']+=int((~torch.isfinite(m)).sum())
        row['nonfinite_v']+=int((~torch.isfinite(v)).sum())
    for row in counts.values():
        row['nonzero_m_zero_v_fraction']=row['nonzero_m_zero_v']/row['elements']
    exposed=counts.get('torch.float16',{}).get('nonzero_m_zero_v_fraction',0)>.01
    report=dict(run_id=args.run_id,status='completed',exit_status=0,source=str(source),
                source_sha256=source_hash,script_sha256=sha(__file__),counts=counts,
                test_loaded=False,gpu_used=False,optimizer_updates=0,
                exposure_gate=exposed,wall_seconds=time.time()-start,
                decision='Consider matched numerical baseline correction; recall causality UNKNOWN.' if exposed else
                'Exposure gate failed; do not pursue zero-second-moment explanation.')
    atomic_json(out/'run.json',report)
    with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
        f.write(json.dumps(report)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
