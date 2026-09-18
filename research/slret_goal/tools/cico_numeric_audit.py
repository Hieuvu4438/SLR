"""CPU audit of completed numerical CiCo pair; does not execute a model."""
import argparse
import json
import sys
import time
import traceback

import numpy as np
import torch

from extraction_resume import atomic_json
from inventory import ROOT,sha
from optimizer_exposure import count_moments
from seds_pair_audit import same_tree


def select(evaluations):
    initial=evaluations['0']
    best=None
    for step in sorted(map(int,evaluations)):
        r1={d:evaluations[str(step)][d]['R1'] for d in ['T2V','V2T']}
        mean=sum(r1.values())/2
        if all(r1[d]>=initial[d]['R1']-.5 for d in r1) and (best is None or mean>best['mean_R1']):
            best=dict(step=step,R1=r1,mean_R1=mean)
    return best


def check_rows(a,b,steps_per_epoch=13):
    total=steps_per_epoch*20
    if len(a)!=total or len(b)!=total:raise ValueError('Full20epoch exposure required')
    for index,(x,y) in enumerate(zip(a,b)):
        if x['step']!=index+1 or x['epoch']!=index//steps_per_epoch or len(x['ids'])!=512:
            raise ValueError('Exposure count failed')
        for key in ['step','epoch','ids','input_sha256','lr','rng_digest']:
            if x[key]!=y[key]:raise ValueError(f'Paired {key} mismatch')
    for key in ['loss','gradient_norm','gradient_sha256']:
        if a[0][key]!=b[0][key]:raise ValueError(f'Initial {key} mismatch')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--native',default='cico-numeric-native-001')
    parser.add_argument('--fp32',default='cico-numeric-fp32-001')
    cli=parser.parse_args()
    roots=[ROOT/'artifacts/slret_goal'/n for n in [cli.native,cli.fp32]]
    runs=[json.loads((p/'run.json').read_text()) for p in roots]
    if any(r['status']!='completed' or r['exit_status']!=0 or r['smoke'] for r in runs):
        raise ValueError('Only completed full runs can be audited')
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started=time.time()
    report=dict(run_id=cli.run_id,status='running',gpu_used=False,test_loaded=False,
                script_sha256=sha(__file__),command=sys.argv,source_reports={str(p/'run.json'):sha(p/'run.json') for p in roots})
    try:
        torch.set_num_threads(4)
        keys=['seed','initial_checkpoint_sha256','config','source_sha256','script_sha256','protocol_sha256','manifests','initial_model_digest',
              'feature_hashes_sha256','schedule_total_steps','optimizer_updates','examples_seen','endpoint_rng_digest']
        assert all(runs[0][k]==runs[1][k] for k in keys)
        per_epoch=runs[0].get('steps_per_epoch',13)
        assert per_epoch==runs[1].get('steps_per_epoch',13)
        total=per_epoch*20
        assert runs[0]['optimizer_updates']==total and runs[0]['examples_seen']==total*512
        rows=[[json.loads(line) for line in (p/'train_steps.jsonl').read_text().splitlines()] for p in roots]
        check_rows(*rows,steps_per_epoch=per_epoch)
        for path,expected in runs[0]['source_sha256'].items():assert sha(ROOT/path)==expected
        features=json.loads((roots[0]/'feature_hashes.json').read_text())
        for path,expected in features.items():assert sha(path)==expected
        report.update(paired_steps=total,examples_per_arm=total*512,verified_feature_files=len(features),
                      dataset=runs[0].get('dataset','ph'),seed=runs[0]['seed'])
        sys.path.insert(0,str(ROOT/'shared'))
        from slr_common.data.manifest import load_manifest
        from slr_common.evaluation.cico_eval import evaluate_score_matrix
        records=load_manifest(ROOT/runs[0]['config']['data']['dev_manifest'],expected_split='dev')
        vids=list(dict.fromkeys(r.video_id for r in records))
        texts=list(dict.fromkeys(r.caption_id for r in records))
        expected=(519,519) if report['dataset']=='ph' else (1077,797)
        assert (len(vids),len(texts))==expected
        v2t,t2v={},{}
        for r in records:
            if r.caption_id not in v2t.setdefault(r.video_id,[]):v2t[r.video_id].append(r.caption_id)
            if r.video_id not in t2v.setdefault(r.caption_id,[]):t2v[r.caption_id].append(r.video_id)
        report['score_hashes']={}
        for root,run in zip(roots,runs):
            assert set(map(int,run['evaluations']))==set(range(0,total+1,per_epoch))
            assert sha(root/'feature_hashes.json')==run['feature_hashes_sha256']
            for step in range(0,total+1,per_epoch):
                folder=root/f'eval_step{step:03d}'
                matrix=np.load(folder/'scores_video_x_text.npy')
                metrics=evaluate_score_matrix(matrix,video_ids=vids,text_ids=texts,video_to_text=v2t,text_to_video=t2v)
                saved=json.loads((folder/'metrics.json').read_text())
                assert all(metrics[d]==saved[d] for d in ['T2V','V2T'])
                compact={d:{k:v for k,v in metrics[d].items() if k!='cols'} for d in ['T2V','V2T']}
                assert compact==run['evaluations'][str(step)]
                report['score_hashes'][str(folder/'scores_video_x_text.npy')]=sha(folder/'scores_video_x_text.npy')
            selection=select(run['evaluations'])
            assert all(run['selection'][k]==v for k,v in selection.items())
            assert sha(run['selection']['checkpoint'])==run['selection']['sha256']
            assert sha(run['final_checkpoint'])==run['final_checkpoint_sha256']
        assert np.array_equal(*[np.load(p/'eval_step000/scores_video_x_text.npy') for p in roots])
        checkpoints=[torch.load(r['final_checkpoint'],map_location='cpu',weights_only=False,mmap=True) for r in runs]
        for key in ['scheduler','rng','sampler_generator_state','completed_steps','completed_epochs','config']:
            assert same_tree(checkpoints[0][key],checkpoints[1][key]),key
        assert checkpoints[0]['model'].keys()==checkpoints[1]['model'].keys()
        assert all(v.dtype==checkpoints[1]['model'][k].dtype and v.shape==checkpoints[1]['model'][k].shape
                   for k,v in checkpoints[0]['model'].items())
        assert all(torch.isfinite(v).all() for c in checkpoints for v in c['model'].values())
        counts=[count_moments(c['optimizer']['state']) for c in checkpoints]
        assert counts==[r['moment_counts'] for r in runs]
        assert set(counts[1])=={'torch.float32'}
        assert counts[1]['torch.float32']['nonzero_m_zero_v']==0
        endpoint={d:runs[1]['evaluations'][str(total)][d]['R1']-runs[0]['evaluations'][str(total)][d]['R1'] for d in ['T2V','V2T']}
        selected=[select(r['evaluations']) for r in runs]
        gain=selected[1]['mean_R1']-selected[0]['mean_R1']
        accuracy=gain>=.5 and all(selected[1]['R1'][d]>=selected[0]['R1'][d]-.5 for d in endpoint)
        mechanism=sum(endpoint.values())/2>=1 and all(v>=0 for v in endpoint.values())
        report.update(status='completed',exit_status=0,selections=selected,selected_mean_delta=gain,
                      endpoint_delta=endpoint,endpoint_mean_delta=sum(endpoint.values())/2,
                      accuracy_gate=accuracy,mechanism_replication_gate=mechanism,moment_counts=counts,
                      decision='Artifact/matched-exposure verification only; not independent multiseed confirmation or SOTA.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['score_hashes','moment_counts']},indent=2))


if __name__=='__main__':main()
