"""Preregistered fixed-gallery CSL caption-cluster analysis, no model selection."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from cico_locked_csl_test import SEEDS, NAMES, verify_lock
from extraction_resume import atomic_json
from inventory import ROOT, sha


def cluster_counts(perquery, video_ids, text_ids, video_to_caption):
    """[direction,caption_group,success/denom]; grouped metric uses scalar ranks."""
    if [r['query_id'] for r in perquery['T2V']]!=text_ids:
        raise ValueError('Text order drift')
    if [r['query_id'] for r in perquery['V2T']]!=video_ids:
        raise ValueError('Video order drift')
    indexes={cid:i for i,cid in enumerate(text_ids)}
    counts=np.zeros((2,len(text_ids),2),dtype=np.int64)
    for d,direction in enumerate(['T2V','V2T']):
        for row in perquery[direction]:
            cid=row['query_id'] if d==0 else video_to_caption[row['query_id']]
            counts[d,indexes[cid]] += [int(row['rank']<1),1]
    if np.any(counts[...,1]==0):raise ValueError('Missing caption cluster contribution')
    return counts


def cluster_interval(counts,*,draws=10000,seed=20260919):
    """[3 seeds,2 arms,2 directions,caption_groups,success/denom]."""
    if counts.ndim!=5 or counts.shape[:3]!=(3,2,2) or counts.shape[-1]!=2:
        raise ValueError('All paired seeds/arms/directions required')
    if np.any(counts[...,1]<=0):raise ValueError('Empty cluster denominator')
    groups=counts.shape[3]
    rng=np.random.default_rng(seed)
    deltas=[]
    for start in range(0,draws,250):
        weights=rng.multinomial(groups,np.full(groups,1/groups),size=min(250,draws-start))
        summed=np.einsum('bg,sadgk->bsadk',weights,counts)
        rates=summed[...,0]/summed[...,1]*100
        deltas.extend((rates[:,:,1,:]-rates[:,:,0,:]).mean(axis=(1,2)).tolist())
    return dict(draws=draws,analysis_seed=seed,cluster_count=groups,
        ci95=np.quantile(deltas,[.025,.975]).tolist(),bootstrap_mean=float(np.mean(deltas)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    cli=parser.parse_args();started=time.time()
    root=ROOT/'artifacts/slret_goal';source=root/'cico-csl-locked-test-001'
    run=json.loads((source/'run.json').read_text())
    if run['status']!='completed' or set(run['results'])!=NAMES:
        raise ValueError('All ten locked models required')
    assert sha(run['lock_path'])==run['lock_sha256']
    lock=json.loads(Path(run['lock_path']).read_text());verify_lock(lock)
    out=root/cli.run_id;out.mkdir(exist_ok=False)
    sys.path.insert(0,str(ROOT/'shared'))
    from slr_common.data.manifest import load_manifest
    from slr_common.evaluation.cico_eval import evaluate_score_matrix
    from slr_common.utils import ordered_hash,sha256_json
    records=load_manifest(lock['test_manifest'],expected_split='test')
    video_ids=[r.video_id for r in records]
    text_ids=list(dict.fromkeys(r.caption_id for r in records))
    v2t={r.video_id:[r.caption_id] for r in records}
    t2v={cid:[r.video_id for r in records if r.caption_id==cid] for cid in text_ids}
    video_to_caption={r.video_id:r.caption_id for r in records}
    expected_hashes=dict(videos=ordered_hash(video_ids),texts=ordered_hash(text_ids),
        positive_mapping=sha256_json(dict(video_to_text=v2t,text_to_video=t2v)))
    counts={};models={};table=[]
    for name,entry in run['results'].items():
        folder=source/name
        for filename,key in [('scores_video_x_text.npy','score_sha256'),('metrics.json','metrics_sha256'),('per_query.json','per_query_sha256')]:
            assert sha(folder/filename)==entry[key]
        scores=np.load(folder/'scores_video_x_text.npy')
        recomputed=evaluate_score_matrix(scores,video_ids=video_ids,text_ids=text_ids,
            video_to_text=v2t,text_to_video=t2v)
        saved=json.loads((folder/'metrics.json').read_text())
        perquery=json.loads((folder/'per_query.json').read_text())
        assert saved['id_hashes']==expected_hashes
        assert saved['metric_kernel']==recomputed['metric_kernel']=='id_multi_positive_best_rank'
        counts[name]=cluster_counts(perquery,video_ids,text_ids,video_to_caption)
        for d,direction in enumerate(['T2V','V2T']):
            assert recomputed[direction]==saved[direction]
            expected=[{k:v for k,v in r.items() if k!='ranked_candidate_ids'} for r in recomputed['per_query'][direction]]
            assert perquery[direction]==expected
            metric={k:v for k,v in saved[direction].items() if k!='cols'}
            assert entry['metrics'][direction]==metric
            assert abs(counts[name][d,:,0].sum()/counts[name][d,:,1].sum()*100-metric['R1'])<1e-10
            table.append(dict(model=name,direction=direction,**metric))
        models[name]=entry['metrics']
    paired=np.stack([np.stack([counts[f'{arm}_s{seed}'] for arm in ['native','fp32']]) for seed in SEEDS])
    interval=cluster_interval(paired)
    interval['video_cluster_sizes']=[len(t2v[cid]) for cid in text_ids]
    def meanr(name):return sum(models[name][d]['R1'] for d in ['T2V','V2T'])/2
    initial=meanr('initial');effects=[]
    for seed in SEEDS:
        native,fp32=f'native_s{seed}',f'fp32_s{seed}'
        delta={d:models[fp32][d]['R1']-models[native][d]['R1'] for d in ['T2V','V2T']}
        selected_native,selected_fp32=lock['selected_mapping'][native],lock['selected_mapping'][fp32]
        effects.append(dict(seed=seed,direction_delta=delta,mean_delta=sum(delta.values())/2,
            native_mean=meanr(native),fp32_mean=meanr(fp32),fp32_minus_initial=meanr(fp32)-initial,
            selected_native_model=selected_native,selected_fp32_model=selected_fp32,
            selected_mean_delta=meanr(selected_fp32)-meanr(selected_native),
            selected_fp32_minus_initial=meanr(selected_fp32)-initial))
    mean=statistics.mean(x['mean_delta'] for x in effects)
    direction_means={d:statistics.mean(x['direction_delta'][d] for x in effects) for d in ['T2V','V2T']}
    passed=mean>=1 and sum(x['mean_delta']>0 for x in effects)>=2 and all(x>=0 for x in direction_means.values()) and interval['ci95'][0]>0
    report=dict(run_id=cli.run_id,status='completed',exit_status=0,gpu_used=False,test_loaded=True,
        command=sys.argv,script_sha256=sha(__file__),source_report_sha256=sha(source/'run.json'),
        lock_sha256=run['lock_sha256'],models=models,effects=effects,initial_mean_R1=initial,
        paired_endpoint_mean_delta=mean,paired_endpoint_delta_sample_std=statistics.stdev(x['mean_delta'] for x in effects),
        direction_delta_means=direction_means,conditional_cluster_bootstrap=interval,
        mechanism_confirmation_gate=passed,
        selected_mean_delta=statistics.mean(x['selected_mean_delta'] for x in effects),
        selected_delta_sample_std=statistics.stdev(x['selected_mean_delta'] for x in effects),
        limitations=['795/798 TEST caption IDs overlap DEV; held-out videos, not novel-caption confirmation.',
            'Caption-cluster CI conditional on models and fixed gallery; residual signer dependence unmodeled.',
            'Historical checkpoint/pretraining exposure incomplete; extension registered after PH TEST.',
            'Known moment correction, no new optimizer; prior selected-accuracy pilot gates failed.'],
        decision='Fixed-model CSL video-holdout extension; no retuning or best-TEST model selection.',
        wall_seconds=time.time()-started)
    with (out/'model_metrics.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','direction','R1','R5','R10','MedianR','MeanR'])
        writer.writeheader();writer.writerows(table)
    report['csv_sha256']=sha(out/'model_metrics.csv')
    atomic_json(out/'run.json',report)
    with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['models','conditional_cluster_bootstrap']},indent=2))
    print(json.dumps({k:v for k,v in interval.items() if k!='video_cluster_sizes'},indent=2))


if __name__=='__main__':main()
