"""Descriptive all-seed DEV comparison; not independent confirmation or a CI."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/slret_goal/tools'))
from extraction_resume import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    directory=ROOT/'artifacts/slret_goal_v2'
    reports={}
    for seed in (42,1337,2026):
        for policy in ('fusion','lora'):
            run=('seds-lora-control-fusion-001' if policy=='fusion' else 'seds-lora-001') if seed==42 else f'seds-{policy}-seed{seed}-001'
            path=directory/run/'run.json';data=path.read_bytes();report=json.loads(data)
            if report['status']!='completed' or report['steps']!=666:
                raise ValueError(f'Incomplete {run}')
            reports[seed,policy]=(report,hashlib.sha256(data).hexdigest())
    rows=[];pairs=[]
    for seed in (42,1337,2026):
        control=reports[seed,'fusion'][0];method=reports[seed,'lora'][0]
        for key in ('batch_order_sha256','checkpoint_sha256_inherited','inherited_asset_digests'):
            if control[key]!=method[key]:raise ValueError(f'Pair mismatch {seed}/{key}')
        for key in ('batch_size','seed','lr','sign_lr','warmup_proportion','rgb_pose_match_loss'):
            if control['native_config'][key]!=method['native_config'][key]:raise ValueError(f'Recipe mismatch {seed}/{key}')
        for policy in ('fusion','lora'):
            report=reports[seed,policy][0]
            selected=report['selection'];selection_step=str(selected['step'])
            for phase,step in [('initial','0'),('selected',selection_step),('endpoint','666')]:
                metrics=report['evaluations'][step]['fusion']
                for direction in ('T2V','V2T'):
                    values=metrics[direction]
                    rows.append(dict(seed=seed,policy=policy,phase=phase,step=int(step),direction=direction,
                                     **{k:values[k] for k in ('R1','R5','R10','MedianR','MeanR')}))
        pairs.append(dict(seed=seed,selected_delta=method['selection']['mean_R1']-control['selection']['mean_R1'],
            method_selected_mean=method['selection']['mean_R1'],control_selected_mean=control['selection']['mean_R1'],
            method_step=method['selection']['step'],control_step=control['selection']['step'],
            method_delta_reference=method['selection']['mean_R1']-method['reference_mean_R1'],
            method_delta_inherited_incumbent=method['selection']['mean_R1']-method['incumbent_mean_R1']))
    output=directory/args.run_id;output.mkdir(parents=True,exist_ok=False)
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    deltas=[p['selected_delta'] for p in pairs]
    summary=dict(run_id=args.run_id,status='completed',pairs=pairs,
        mean_selected_delta=statistics.mean(deltas),sample_sd_selected_delta=statistics.stdev(deltas),
        positive_seeds=sum(d>1e-8 for d in deltas),
        tied_seeds=sum(abs(d)<=1e-8 for d in deltas),
        tie_tolerance_pp=1e-8,
        source_sha256={f'{seed}/{policy}':value[1] for (seed,policy),value in reports.items()},
        metrics_path=str(output/'metrics.csv'),
        scope='Three continuation seeds on one initialization; historically exposed selection DEV. No CI, TEST or SOTA claim.')
    atomic_json(output/'run.json',summary)
    with (ROOT/'research/slret_goal_v2/EXPERIMENTS.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
