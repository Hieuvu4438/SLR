"""Descriptive aggregation of all six audited pairs; no significance inference."""
import argparse
import csv
import json
import statistics
import sys
import time

from extraction_resume import atomic_json
from inventory import ROOT,sha


def summarize(rows):
    if len(rows)!=3 or {r['seed'] for r in rows}!={42,1337,2026}:
        raise ValueError('Exactly three registered seeds required; no survivor-only summary')
    deltas=[r['endpoint_mean_delta'] for r in rows]
    directions={d:statistics.mean(r['endpoint_delta'][d] for r in rows) for d in ['T2V','V2T']}
    result=dict(n_seeds=3,endpoint_delta_mean=statistics.mean(deltas),
                endpoint_delta_sample_std=statistics.stdev(deltas),direction_delta_means=directions,
                positive_seed_count=sum(d>0 for d in deltas),
                selected_delta_mean=statistics.mean(r['selected_mean_delta'] for r in rows),
                accuracy_gate_seed_count=sum(r['accuracy_gate'] for r in rows))
    result['replication_gate']=result['endpoint_delta_mean']>=1 and result['positive_seed_count']>=2 and all(v>=0 for v in directions.values())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli=parser.parse_args()
    started=time.time()
    root=ROOT/'artifacts/slret_goal'
    driver=json.loads((root/'cico-numeric-replication-001/run.json').read_text())
    if driver['status']!='completed':raise ValueError('Replication queue incomplete; retain all failures, do not summarize survivors')
    out=root/cli.run_id
    out.mkdir(exist_ok=False)
    report=dict(run_id=cli.run_id,status='running',test_loaded=False,gpu_used=False,
                command=sys.argv,script_sha256=sha(__file__),pairs=[],datasets={},
                inference='Descriptive training-seed variation on historically exposed DEV; no confidence/significance/independence claim.')
    metric_rows=[]
    for dataset in ['ph','csl']:
        group=[]
        for seed in [42,1337,2026]:
            auditname='cico-numeric-audit-001' if dataset=='ph' and seed==42 else f'cico-repl-{dataset}-s{seed}-audit-001'
            path=root/auditname/'run.json'
            audit=json.loads(path.read_text())
            assert audit['status']=='completed' and audit['exit_status']==0
            entries=[]
            for source,expected in audit['source_reports'].items():
                assert sha(source)==expected
                run=json.loads(__import__('pathlib').Path(source).read_text())
                assert run['seed']==seed
                assert run.get('dataset','ph')==dataset
                assert sha(run['final_checkpoint'])==run['final_checkpoint_sha256']
                entries.append(run)
                for step,metric in run['evaluations'].items():
                    for direction in ['T2V','V2T']:
                        metric_rows.append(dict(dataset=dataset,seed=seed,arm=run['arm'],step=int(step),
                                                direction=direction,**metric[direction]))
            assert {r['arm'] for r in entries}=={'native','fp32'}
            record=dict(dataset=dataset,seed=seed,audit_sha256=sha(path),audit_run_id=auditname,
                        discovery_seed=dataset=='ph' and seed==42,
                        **{k:audit[k] for k in ['endpoint_delta','endpoint_mean_delta','selected_mean_delta','accuracy_gate','selections']})
            record['arms']={r['arm']:dict(endpoint=r['evaluations'][str(r['optimizer_updates'])],
                                         initial=r['evaluations']['0'],selection=r['selection']) for r in entries}
            report['pairs'].append(record)
            group.append(record)
        report['datasets'][dataset]=summarize(group)
    with (out/'all_epoch_metrics.csv').open('x',newline='') as f:
        fields=['dataset','seed','arm','step','direction','R1','R5','R10','MedianR','MeanR']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    report['statistical_fallacy_scan']={
        '1_simpson':'Datasets kept separate; all direction and seed effects retained; no pooled benchmark score.',
        '2_ecological':'Inference limited to tested seed/model/dataset regimes, not individual signer or language meaning.',
        '3_selection':'Historical selected checkpoints and DEV exposure remain selection limitations, not random population samples.',
        '4_collider':'No conditioning on post-treatment success; fixed six pairs required before summary.',
        '5_base_rate':'Official full galleries retained; not a diagnostic sensitivity/specificity claim.',
        '6_regression_to_mean':'Matched continuation controls and initialization are both reported; no uncontrolled pre/post claim.',
        '7_survivorship':'All six registered pairs and queue completion required; missing/failed pair blocks full summary.',
        '8_look_elsewhere':'PH42 is discovery; all other fixed seeds and both datasets included; no p-values or significance claims.',
        '9_forking_paths':'Protocol fixed before replication; B32/B512 pilot decisions documented separately; no hyperparameter rescue.',
        '10_causality':'Only paired numerical intervention effects in tested runs; no inference about author training or semantic correctness.',
        '11_reverse_causality':'Precision assignment precedes updates; source identities and paired exposure audited.'}
    report.update(status='completed',exit_status=0,wall_seconds=time.time()-started,
                  metric_rows=len(metric_rows),csv_sha256=sha(out/'all_epoch_metrics.csv'),
                  decision='Replication description only; independent confirmation and contribution/novelty audit still required.')
    atomic_json(out/'run.json',report)
    with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    print(json.dumps(dict(status=report['status'],datasets=report['datasets'],metric_rows=len(metric_rows)),indent=2))


if __name__=='__main__':main()
