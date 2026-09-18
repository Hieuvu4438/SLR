"""Registered paired cluster bootstrap of locked PH TEST endpoints."""
import argparse
import csv
import json
import statistics
import sys
import time

import numpy as np

from extraction_resume import atomic_json
from inventory import ROOT,sha
from cico_locked_ph_test import verify_lock


def contributions(perquery,ids,direction):
    if [r['query_id'] for r in perquery]!=ids:raise ValueError('Query ordering mismatch')
    values=[]
    for row in perquery:
        ranks=row['official_tie_ranks'] if direction=='T2V' else [row['rank']]
        if not ranks:raise ValueError('Empty official rank denominator')
        values.append([sum(r<1 for r in ranks),len(ranks)])
    return np.asarray(values,dtype=np.int64)


def cluster_interval(counts,groups,*,draws=10000,seed=20260918):
    """counts=[3seeds,2arms,2directions,Nqueries,success/denominator]."""
    if counts.shape[:3]!=(3,2,2) or counts.shape[3:]!=(len(groups),2):
        raise ValueError('All pairedseed/arm/direction counts required')
    labels=list(dict.fromkeys(groups))
    grouped=np.stack([counts[:,:,:,np.asarray(groups)==g,:].sum(axis=3) for g in labels],axis=3)
    rng=np.random.default_rng(seed)
    deltas=[]
    for start in range(0,draws,250):
        multiplicity=rng.multinomial(len(labels),np.full(len(labels),1/len(labels)),size=min(250,draws-start))
        aggregate=np.einsum('bg,sadgk->bsadk',multiplicity,grouped)
        rates=aggregate[...,0]/aggregate[...,1]*100
        deltas.extend((rates[:,:,1,:]-rates[:,:,0,:]).mean(axis=(1,2)).tolist())
    return dict(draws=draws,analysis_seed=seed,cluster_count=len(labels),
                ci95=np.quantile(deltas,[.025,.975]).tolist(),
                bootstrap_mean=float(np.mean(deltas)),
                cluster_sizes=[groups.count(g) for g in labels])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli=parser.parse_args()
    started=time.time()
    root=ROOT/'artifacts/slret_goal'
    source=root/'cico-ph-locked-test-001'
    run=json.loads((source/'run.json').read_text())
    if run['status']!='completed' or len(run['results'])!=7:raise ValueError('All7 locked TEST models required')
    assert sha(run['lock_path'])==run['lock_sha256']
    lock=json.loads(__import__('pathlib').Path(run['lock_path']).read_text())
    verify_lock(lock)
    out=root/cli.run_id
    out.mkdir(exist_ok=False)
    sys.path.insert(0,str(ROOT/'shared'))
    from slr_common.data.manifest import load_manifest
    from slr_common.evaluation.cico_eval import evaluate_score_matrix
    records=load_manifest(lock['test_manifest'],expected_split='test')
    ids=[r.video_id for r in records]
    assert ids==[r.caption_id for r in records]
    assert len(set(ids))==642
    mapping={vid:[vid] for vid in ids}
    groups=[vid.rsplit('-',1)[0] for vid in ids]
    models={}
    counts={}
    table=[]
    for name,entry in run['results'].items():
        folder=source/name
        for filename,key in [('scores_video_x_text.npy','score_sha256'),('metrics.json','metrics_sha256'),('per_query.json','per_query_sha256')]:
            assert sha(folder/filename)==entry[key]
        scores=np.load(folder/'scores_video_x_text.npy')
        recomputed=evaluate_score_matrix(scores,video_ids=ids,text_ids=ids,video_to_text=mapping,text_to_video=mapping)
        saved=json.loads((folder/'metrics.json').read_text())
        perquery=json.loads((folder/'per_query.json').read_text())
        counts[name]=[]
        for d in ['T2V','V2T']:
            assert recomputed[d]==saved[d]
            expected=[{k:v for k,v in row.items() if k!='ranked_candidate_ids'} for row in recomputed['per_query'][d]]
            assert perquery[d]==expected
            value=contributions(perquery[d],ids,d)
            assert abs(value[:,0].sum()/value[:,1].sum()*100-saved[d]['R1'])<1e-10
            counts[name].append(value)
            table.append(dict(model=name,direction=d,**{k:v for k,v in saved[d].items() if k!='cols'}))
        counts[name]=np.stack(counts[name])
        models[name]=entry['metrics']
    paired=np.stack([np.stack([counts[f'{a}_s{s}'] for a in ['native','fp32']]) for s in [42,1337,2026]])
    interval=cluster_interval(paired,groups)
    effects=[]
    initial=sum(models['initial'][d]['R1'] for d in ['T2V','V2T'])/2
    for seed in [42,1337,2026]:
        a,b=models[f'native_s{seed}'],models[f'fp32_s{seed}']
        delta={d:b[d]['R1']-a[d]['R1'] for d in ['T2V','V2T']}
        effects.append(dict(seed=seed,direction_delta=delta,mean_delta=sum(delta.values())/2,
                            native_mean=sum(a[d]['R1'] for d in delta)/2,
                            fp32_mean=sum(b[d]['R1'] for d in delta)/2,
                            fp32_minus_initial=sum(b[d]['R1'] for d in delta)/2-initial))
    mean=statistics.mean(x['mean_delta'] for x in effects)
    direction_means={d:statistics.mean(x['direction_delta'][d] for x in effects) for d in ['T2V','V2T']}
    passed=mean>=1 and sum(x['mean_delta']>0 for x in effects)>=2 and all(v>=0 for v in direction_means.values()) and interval['ci95'][0]>0
    report=dict(run_id=cli.run_id,status='completed',exit_status=0,gpu_used=False,test_loaded=True,
                source_report_sha256=sha(source/'run.json'),lock_sha256=run['lock_sha256'],script_sha256=sha(__file__),
                models=models,effects=effects,initial_mean_R1=initial,
                paired_endpoint_mean_delta=mean,paired_endpoint_delta_sample_std=statistics.stdev(x['mean_delta'] for x in effects),
                direction_delta_means=direction_means,conditional_cluster_bootstrap=interval,
                mechanism_confirmation_gate=passed,selected_model_accuracy_delta=0.,
                limitations=['QueryclusterCI conditional on trainedmodels and fixedgallery; not trainingseedpopulation CI.',
                             'Filenameprefix clusters are inferredbroadcast proxies, not verified independent source/signers.',
                             'Campaignheldout TEST; historicalrelease/pretraining/testselection provenance incomplete.',
                             'KnownFP32 optimizerstate correction, not newmethod; noCSLTESTresult.',
                             'AllPHDEVselectors initialization; endpointmitigation isnot selectedaccuracygain orSOTA.'],
                decision='Registered PH TEST contrast only; final contribution/claim audit still required.',wall_seconds=time.time()-started)
    with (out/'model_metrics.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','direction','R1','R5','R10','MedianR','MeanR'])
        writer.writeheader();writer.writerows(table)
    atomic_json(out/'run.json',report)
    with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    print(json.dumps({k:report[k] for k in ['run_id','status','effects','paired_endpoint_mean_delta','paired_endpoint_delta_sample_std','direction_delta_means','mechanism_confirmation_gate','selected_model_accuracy_delta']},indent=2))
    print(json.dumps({k:interval[k] for k in ['cluster_count','draws','ci95']},indent=2))


if __name__=='__main__':main()
