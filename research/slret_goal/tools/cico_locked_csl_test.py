"""One fixed CSL TEST batch; immutable input/model/source lock, no fitting."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from extraction_resume import atomic_json
from inventory import ROOT, sha

SEEDS=(42,1337,2026)
SELECTED_STEPS={'native_s42':36,'fp32_s42':0,'native_s1337':0,
                'fp32_s1337':228,'native_s2026':0,'fp32_s2026':216}
NAMES={'initial',*[f'{a}_s{s}' for s in SEEDS for a in ['native','fp32']],
       *[f'selected_{name}' for name,step in SELECTED_STEPS.items() if step]}
PROTOCOL=ROOT/'research/slret_goal/CICO_LOCKED_CSL_TEST_PROTOCOL.md'


def verify_lock(lock):
    if lock['gallery']!={'videos':1176,'texts':798}:
        raise ValueError('Fixed CSL gallery required')
    if sha(lock['test_manifest'])!=lock['test_manifest_sha256']:
        raise ValueError('Manifest drift')
    for field in ['sources','evidence','assets']:
        for path,expected in lock[field].items():
            if sha(path)!=expected:raise ValueError(f'Locked {field} drift: {path}')
    if len(lock['models'])!=10 or {m['name'] for m in lock['models']}!=NAMES:
        raise ValueError('All ten distinct models required')
    if lock['selected_mapping']!={name:f'selected_{name}' if step else 'initial'
                                  for name,step in SELECTED_STEPS.items()}:
        raise ValueError('DEV selection mapping drift')
    for m in lock['models']:
        expected_step=0 if m['name']=='initial' else (SELECTED_STEPS[m['name'].removeprefix('selected_')]
                      if m['name'].startswith('selected_') else 240)
        if m['step']!=expected_step or sha(m['checkpoint'])!=m['checkpoint_sha256']:
            raise ValueError('Checkpoint step or identity drift')


def prepare(out):
    from slr_common.data.manifest import load_manifest
    root=ROOT/'artifacts/slret_goal'
    prepdir=root/'csl-test-assets-001'
    preppath=prepdir/'run.json'
    prep=json.loads(preppath.read_text())
    if prep['status']!='completed':raise ValueError('Complete input preparation required')
    assert prep['test_scores_accessed'] is False
    manifest=Path(prep['manifest']['path'])
    assert sha(manifest)==prep['manifest']['manifest_sha256']
    assetpath=prepdir/'asset_hashes.json'
    assert sha(assetpath)==prep['asset_hashes_sha256']
    assets=json.loads(assetpath.read_text())
    rows=load_manifest(manifest,expected_split='test')
    assert len(rows)==len({r.video_id for r in rows})==1176
    assert len({r.caption_id for r in rows})==798
    evidence={str(preppath):sha(preppath),str(assetpath):sha(assetpath)}
    for path,expected in prep['source_hashes'].items():
        assert sha(path)==expected
        evidence[path]=expected
    for r in rows:
        assert r.dataset=='csl_daily' and r.caption_language=='en'
        for stream,path in [('agnostic',r.feature_agnostic),('aware',r.feature_aware),('temporal',r.temporal_metadata)]:
            assert Path(path).resolve().is_relative_to(prepdir/stream)
            assert str(path) in assets
    for split in ['train','dev']:
        p=ROOT/f'artifacts/manifests/csl_{split}.jsonl'
        reference=load_manifest(p,expected_split=split)
        assert not {r.video_id for r in rows}&{r.video_id for r in reference}
        count=len({r.caption_id for r in rows}&{r.caption_id for r in reference})
        assert count==(0 if split=='train' else 795)
        evidence[str(p)]=sha(p)
    replaypath=root/'cico-checkpoint-replay-001/run.json'
    replay=json.loads(replaypath.read_text())
    assert replay['status']=='completed' and len(replay['checkpoint_replays'])==12
    evidence[str(replaypath)]=sha(replaypath)
    selectionpath=ROOT/'runs/csl_base_b512_s42/selection.json'
    initial=json.loads(selectionpath.read_text())
    evidence[str(selectionpath)]=sha(selectionpath)
    models=[dict(name='initial',seed=42,step=0,envelope='historical',
        checkpoint=str(selectionpath.parent/initial['checkpoint']),
        checkpoint_sha256=initial['checkpoint_sha256'])]
    cfg=None
    for seed in SEEDS:
        auditpath=root/f'cico-repl-csl-s{seed}-audit-001/run.json'
        audit=json.loads(auditpath.read_text())
        assert audit['status']=='completed'
        evidence[str(auditpath)]=sha(auditpath)
        for path,expected in audit['source_reports'].items():
            assert sha(path)==expected
            run=json.loads(Path(path).read_text())
            assert run['dataset']=='csl' and run['seed']==seed and run['optimizer_updates']==240
            assert run['initial_checkpoint_sha256']==initial['checkpoint_sha256']
            name=f'{run["arm"]}_s{seed}'
            assert run['selection']['step']==SELECTED_STEPS[name]
            evidence[path]=expected
            current=run['config']
            if cfg is None:cfg=copy.deepcopy(current)
            assert {k:v for k,v in cfg.items() if k!='seed'}=={k:v for k,v in current.items() if k!='seed'}
            models.append(dict(name=name,seed=seed,step=240,envelope='final',
                checkpoint=run['final_checkpoint'],checkpoint_sha256=run['final_checkpoint_sha256']))
            selected=run['selection']
            if selected['step']:
                models.append(dict(name=f'selected_{name}',seed=seed,step=selected['step'],envelope='selected',
                    checkpoint=selected['checkpoint'],checkpoint_sha256=selected['sha256']))
            else:assert selected['sha256']==initial['checkpoint_sha256']
    assert cfg['data']['alpha']==.8 and cfg['data']['feature_len']==64 and cfg['data']['max_words']==32
    cfg['data']['test_manifest']=str(manifest)
    files=[Path(__file__),PROTOCOL,ROOT/'research/slret_goal/tools/cico_csl_test_analysis.py',
        ROOT/'methods/elsc/elsc/evaluate.py',ROOT/'methods/elsc/elsc/models/retriever.py',
        ROOT/'shared/slr_common/evaluation/runtime.py',ROOT/'shared/slr_common/evaluation/cico_eval.py',
        ROOT/'shared/slr_common/upstream/factory.py',ROOT/'shared/slr_common/upstream/cico_bridge.py',
        ROOT/'shared/slr_common/data/cico_dataset.py',ROOT/'shared/slr_common/data/views.py',
        ROOT/'shared/slr_common/data/tokenize.py',ROOT/'shared/slr_common/data/manifest.py',
        ROOT/'third_party/SLRT/CiCo/CLCL/modules/modeling.py',
        ROOT/'third_party/SLRT/CiCo/CLCL/modules/module_clip.py']
    files.extend((ROOT/'third_party/SLRT/CiCo/CLCL/modules').glob('*.py'))
    files.extend([ROOT/'third_party/SLRT/CiCo/CLCL/modules/bpe_simple_vocab_16e6.txt.gz',
                  ROOT/'methods/elsc/elsc/train.py',ROOT/'methods/elsc/elsc/upstream/factory.py'])
    lock=dict(created_unix=time.time(),test_scores_accessed=False,
        test_manifest=str(manifest),test_manifest_sha256=sha(manifest),config=cfg,
        gallery=dict(videos=1176,texts=798),models=models,assets=assets,evidence=evidence,
        selected_mapping={n:f'selected_{n}' if step else 'initial' for n,step in SELECTED_STEPS.items()},
        sources={str(p):sha(p) for p in files})
    verify_lock(lock)
    atomic_json(out/'lock.json',lock)
    return dict(lock_path=str(out/'lock.json'),lock_sha256=sha(out/'lock.json'),locked_models=10)


def evaluate(out,path,expected):
    assert sha(path)==expected
    lock=json.loads(Path(path).read_text())
    verify_lock(lock)
    from elsc.models.retriever import ELSCRetriever
    from elsc.train import _seed_everything
    from elsc.evaluate import evaluate_model
    from slr_common.upstream.factory import _load_cico_core_from_state
    results={}
    torch.cuda.reset_peak_memory_stats()
    for item in lock['models']:
        _seed_everything(item['seed'])
        ckpt=torch.load(item['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        if item['envelope']=='final':assert ckpt['completed_steps']==240
        cfg=lock['config']
        state={n.removeprefix('core.'):v for n,v in ckpt['model'].items() if n.startswith('core.')}
        core=_load_cico_core_from_state(cfg,state,cico_root=ROOT/cfg['upstream']['cico_root'],device='cuda')
        model=ELSCRetriever(core,input_dim=1024,hidden_dim=256,text_dim=core.clip.text_projection.shape[-1],
            core_frozen=False,adapter_enabled=False).cuda()
        model.load_state_dict(ckpt['model'],strict=True)
        assert all(torch.equal(v.cpu(),ckpt['model'][n]) for n,v in model.state_dict().items())
        model.eval().requires_grad_(False)
        scores,metrics=evaluate_model(model,cfg,'test',torch.device('cuda'))
        assert scores.shape==(1176,798) and metrics['gallery']==lock['gallery']
        assert metrics['metric_kernel']=='id_multi_positive_best_rank'
        folder=out/item['name'];folder.mkdir()
        np.save(folder/'scores_video_x_text.npy',scores)
        compact={k:v for k,v in metrics.items() if k not in ['per_query','diagnostic_best_positive']}
        atomic_json(folder/'metrics.json',compact)
        perquery={d:[{k:v for k,v in r.items() if k!='ranked_candidate_ids'} for r in metrics['per_query'][d]]
                  for d in ['T2V','V2T']}
        atomic_json(folder/'per_query.json',perquery)
        results[item['name']]=dict(checkpoint_sha256=item['checkpoint_sha256'],
            score_sha256=sha(folder/'scores_video_x_text.npy'),metrics_sha256=sha(folder/'metrics.json'),
            per_query_sha256=sha(folder/'per_query.json'),
            metrics={d:{k:v for k,v in metrics[d].items() if k!='cols'} for d in ['T2V','V2T']})
        atomic_json(out/'progress.json',dict(completed=len(results),results=results))
        print(json.dumps(dict(completed=item['name'],count=len(results))),flush=True)
        del model,core,ckpt,state,scores,metrics,compact,perquery
        gc.collect();torch.cuda.empty_cache()
    return dict(lock_path=str(path),lock_sha256=expected,results=results,test_scores_accessed=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),torch=torch.__version__,hardware=torch.cuda.get_device_name())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--mode',choices=['lock','evaluate'],required=True)
    parser.add_argument('--lock');parser.add_argument('--lock-sha256')
    cli=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/cli.run_id;out.mkdir(exist_ok=False)
    started=time.time()
    report=dict(run_id=cli.run_id,status='running',pid=os.getpid(),mode=cli.mode,command=sys.argv,
        optimizer_updates=0,script_sha256=sha(__file__),test_scores_accessed=False)
    atomic_json(out/'run.json',report)
    try:
        assert shutil.disk_usage(out).free>15*1024**3
        assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())+128*1024**2<44*1024**3
        torch.set_num_threads(4)
        sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'methods/elsc'),str(ROOT/'third_party/SLRT/CiCo/CLCL')]
        if cli.mode=='lock':report.update(prepare(out))
        else:
            assert cli.lock and cli.lock_sha256
            report['test_scores_accessed']=True
            atomic_json(out/'run.json',report)
            report.update(evaluate(out,cli.lock,cli.lock_sha256))
        report.update(status='completed',exit_status=0)
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc());raise
    finally:
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':main()
