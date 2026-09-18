"""CPU paired-provenance audit, run only after the full RGB pilot is terminal."""
import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from extraction_resume import atomic_json
from inventory import ROOT,sha
from seds_pair_audit import same_tree
from seds_runtime import compatible_load


def audit_scores(matrix, ids, saved, summary):
    """Recompute saved ranks and metrics; native kernels are a second reference.

    This checks evaluation artifacts, not a fresh model forward or confirmation.
    """
    sys.path.insert(0,str(ROOT/'shared'))
    from slr_common.evaluation.cico_eval import evaluate_score_matrix
    spec=importlib.util.spec_from_file_location('seds_native_metric_reference',ROOT/'third_party/SEDS/metrics.py')
    native=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    mapping={vid:[vid] for vid in ids}
    fresh=evaluate_score_matrix(matrix,video_ids=ids,text_ids=ids,
                                video_to_text=mapping,text_to_video=mapping)
    if fresh!=saved:
        raise ValueError('Saved score/rank/metric artifact mismatch')
    native_metrics={'V2T':native.tensor_text_to_video_metrics(matrix[:,None,:].copy()),
                    'T2V':native.compute_metrics(native.tensor_video_to_text_sim(matrix[:,None,:].copy()))}
    for direction in ['T2V','V2T']:
        reduced={k:v for k,v in fresh[direction].items() if k!='cols'}
        if reduced!=summary[direction]:
            raise ValueError('Run summary differs from recomputed metrics')
        # Native torch lower median and numpy median differ for even galleries.
        # PH DEV has 519 singleton V2T ranks; do not generalize its agreement.
        for key in ['R1','R5','R10','MeanR']:
            if abs(reduced[key]-native_metrics[direction][key])>=1e-5:
                raise ValueError('Native metric parity failed')
    if fresh['T2V']['cols']!=native_metrics['T2V']['cols']:
        raise ValueError('Native T2V expanded tie ranks differ')
    return {d:{k:v for k,v in fresh[d].items() if k!='cols'} for d in ['T2V','V2T']}


def select_checkpoint(evaluations):
    initial={d:evaluations['0']['fusion'][d]['R1'] for d in ['T2V','V2T']}
    selected=dict(step=0,R1=initial,mean_R1=sum(initial.values())/2)
    for step in [111,222]:
        r1={d:evaluations[str(step)]['fusion'][d]['R1'] for d in initial}
        mean=sum(r1.values())/2
        if mean>selected['mean_R1'] and all(r1[d]>=initial[d]-.5 for d in initial):
            selected=dict(step=step,R1=r1,mean_R1=mean)
    return selected


def check_pair_rows(pilot,control):
    if len(pilot)!=222 or len(control)!=222:
        raise ValueError('Full222step exposure required')
    if sum(len(r['ids']) for r in pilot)!=7096:
        raise ValueError('Full7096example exposure required')
    for a,b in zip(pilot,control):
        if a['step']!=b['step'] or a['ids']!=b['ids']:
            raise ValueError('Batch order mismatch')
    for a,b in zip(pilot[:2],control[:2]):
        if a['loss']!=b['loss'] or a['head_gradient_norm']!=b['gradient_norm']:
            raise ValueError('Initial loss/gradient mismatch')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli=parser.parse_args()
    pilot=ROOT/'artifacts/slret_goal/rgb-finetune-pilot-001'
    control=ROOT/'artifacts/slret_goal/seds-moment-control-001'
    run=json.loads((pilot/'run.json').read_text())
    ctrl=json.loads((control/'run.json').read_text())
    if any(r['status']!='completed' or r['exit_status']!=0 for r in [run,ctrl]):
        raise ValueError('Terminal completed runs required; do not audit a moving target')
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started=time.time()
    report=dict(run_id=cli.run_id,status='running',command=sys.argv,script_sha256=sha(__file__),
                gpu_used=False,test_loaded=False,source_reports={str(p/'run.json'):sha(p/'run.json') for p in [pilot,control]})
    try:
        torch.set_num_threads(4)
        keys=['checkpoint_sha256','train_assets_digest','dev_assets_digest','batch_order_sha256',
              'seed','batch_size','optimizer_updates','examples_seen']
        report['matched_fields']={k:run[k]==ctrl[k] for k in keys}
        assert all(report['matched_fields'].values())
        differences={k:[run['config'].get(k),ctrl['config'].get(k)] for k in run['config'].keys()|ctrl['config'].keys()
                     if run['config'].get(k)!=ctrl['config'].get(k)}
        assert set(differences)=={'output_dir'}
        report['config_differences']=differences
        assert all(run['source_hashes'][k]==v for k,v in ctrl['source_sha256'].items())
        rows=[json.loads(x) for x in (pilot/'train_steps.jsonl').read_text().splitlines()]
        controls=[json.loads(x) for x in (control/'train_steps.jsonl').read_text().splitlines()]
        check_pair_rows(rows,controls)
        feature_grad_hashes={}
        for version,row in enumerate(rows):
            assert row['encoding']['version']==version and row['backward']['version']==version
            assert row['commit']['version']==version+1 and row['commit']['frozen_and_bn_unchanged']
            assert row['backward']['replay_maxabs']==0
            path=pilot/f'features_v{version}.pt'
            assert sha(path)==row['encoding']['sha256']
            feature=torch.load(path,map_location='cpu',weights_only=True)
            gradpath=pilot/f'gradient_v{version}.pt'
            grad=torch.load(gradpath,map_location='cpu',weights_only=True)
            assert feature['ids']==row['ids'] and feature['version']==grad['version']==version
            assert feature['features'].shape==grad['gradient'].shape==(len(row['ids']),1024,64,1)
            assert torch.isfinite(feature['features']).all() and torch.isfinite(grad['gradient']).all()
            feature_grad_hashes[str(version)]=dict(features=sha(path),gradient=sha(gradpath))
        atomic_json(out/'batch_tensor_hashes.json',feature_grad_hashes)
        report['all222_versions_and_finite_cached_tensors_verified']=True
        report['initial_scores_bitexact']={}
        for stream in ['fusion','pose','rgb']:
            a=np.load(pilot/'eval_step000'/f'{stream}_video_x_text.npy')
            b=np.load(control/'eval_step000'/f'{stream}_video_x_text.npy')
            assert np.array_equal(a,b)
            report['initial_scores_bitexact'][stream]=True
        ids=[json.loads(x)['video_id'] for x in (ROOT/'artifacts/manifests/ph_dev.jsonl').read_text().splitlines()]
        assert len(ids)==len(set(ids))==519
        report['recomputed_score_metrics']={}
        for folder,source_run in [(pilot,run),(control,ctrl)]:
            checked={}
            for step in [0,111,222]:
                for stream in ['fusion','pose','rgb']:
                    prefix=folder/f'eval_step{step:03d}'
                    matrix_path=prefix/f'{stream}_video_x_text.npy'
                    metrics_path=prefix/f'{stream}_metrics.json'
                    summary=audit_scores(np.load(matrix_path),ids,json.loads(metrics_path.read_text()),
                                         source_run['evaluations'][str(step)][stream])
                    checked[f'{step}/{stream}']=dict(score_sha256=sha(matrix_path),
                                                   metrics_sha256=sha(metrics_path),metrics=summary)
            report['recomputed_score_metrics'][folder.name]=checked
        report['metric_sources_sha256']={str(p):sha(p) for p in
            [ROOT/'shared/slr_common/evaluation/cico_eval.py',ROOT/'third_party/SEDS/metrics.py']}
        sys.path.insert(0,str(ROOT/'shared'))
        from slr_common.features.i3d import load_i3d
        from rgb_gradient_replay import unfreeze_i3d_tail
        original=ROOT/'artifacts/pretrained/bsl5k.pth.tar'
        model=load_i3d(original,ROOT/'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py',torch.device('cpu'))
        tailkeys=set(unfreeze_i3d_tail(model))
        state=model.state_dict()
        initialtail={n:state[n].clone() for n in tailkeys}
        report['checkpoint_checks']={}
        for step in [111,222]:
            pair=json.loads((pilot/f'paired_checkpoint_step{step:03d}.json').read_text())
            assert pair['version']==step and sha(pair['head_checkpoint'])==pair['head_sha256']
            assert sha(pair['tail']['path'])==pair['tail']['sha256'] and pair['tail']['version']==step
            head=torch.load(pair['head_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
            baseline=torch.load(control/f'checkpoint_step{step:03d}.pt',map_location='cpu',weights_only=False,mmap=True)
            assert same_tree(head['rng'],baseline['rng']),('RNG mismatch',step)
            assert same_tree(head['batches'],baseline['batches'])
            assert head['model'].keys()==baseline['model'].keys()
            assert all(head['model'][n].shape==v.shape and head['model'][n].dtype==v.dtype for n,v in baseline['model'].items())
            assert head['completed_steps']==step and ('optimizer' in head)==(step==222)
            tail=torch.load(pair['tail']['path'],map_location='cpu',weights_only=True)
            assert set(tail['tail'])==tailkeys and tail['completed_steps']==step
            assert tail['backbone_sha256']==sha(original)
            assert all(torch.isfinite(p).all() for p in tail['tail'].values())
            merged=dict(state)
            merged.update(tail['tail'])
            model.load_state_dict(merged,strict=True)
            assert all(torch.equal(model.state_dict()[n],v) for n,v in tail['tail'].items())
            moments=tail['optimizer']['state']
            assert len(moments)==len(tailkeys)
            assert all(s['step']==step and s['next_m'].dtype==s['next_v'].dtype==torch.float32
                       and torch.isfinite(s['next_m']).all() and torch.isfinite(s['next_v']).all() for s in moments.values())
            devfolder=Path(pair['dev_features']['folder'])
            assert sha(devfolder/'manifest.json')==pair['dev_features']['manifest_sha256']
            manifest=json.loads((devfolder/'manifest.json').read_text())
            expectedids=[json.loads(x)['video_id'] for x in (ROOT/'artifacts/manifests/ph_dev.jsonl').read_text().splitlines()]
            assert set(manifest)==set(expectedids) and len(manifest)==519
            for vid,item in manifest.items():
                path=devfolder/'dev'/f'{vid}.pkl'
                assert sha(path)==item['feature_sha256']
                meta_path=ROOT/'artifacts/slret_goal/seds-adapted-dev-001/metadata'/f'{vid}.json'
                assert sha(meta_path)==item['metadata_sha256']
                meta=json.loads(meta_path.read_text())
                feature=compatible_load(path.open('rb'))['feature']
                assert feature.shape==(len(meta['clip_starts']),1024) and np.isfinite(feature).all()
            report['checkpoint_checks'][str(step)]=dict(rng_exact_to_control=True,model_shapes_dtypes_exact=True,
                    paired_hashes_verified=True,tail_strict_reconstruction=True,dev_features_verified=519,
                    changed_tail_tensors=sum(not torch.equal(initialtail[n],v) for n,v in tail['tail'].items()))
            del head,baseline,tail
        selection=select_checkpoint(run['evaluations'])
        assert all(run['selection'][k]==v for k,v in selection.items())
        strongest=select_checkpoint(ctrl['evaluations'])
        delta=selection['mean_R1']-strongest['mean_R1']
        passed=delta>=.5 and all(selection['R1'][d]>=strongest['R1'][d]-.5 for d in selection['R1'])
        assert run['pilot_gate_passed']==passed and abs(run['selected_delta_mean_R1']-delta)<1e-10
        report.update(status='completed',exit_status=0,selection=selection,control_selection=strongest,
                      selected_delta_mean_R1=delta,pilot_gate_passed=passed,
                      decision='Paired provenance and selector audit only; not independent confirmation or SOTA.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__': main()
