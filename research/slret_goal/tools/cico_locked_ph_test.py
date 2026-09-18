"""Hash-locked fixed PH TEST contrasts; no training or test-based selection."""
import argparse
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
from inventory import ROOT,sha


TEST_SHA='562bc308e82937852659be0ca49793e02b4cf3c8c42ccdd065156885bb8ca5cf'
PROTOCOL=ROOT/'research/slret_goal/CICO_LOCKED_PH_TEST_PROTOCOL.md'


def verify_lock(lock):
    if lock['test_manifest_sha256']!=TEST_SHA or sha(lock['test_manifest'])!=TEST_SHA:
        raise ValueError('TEST manifest differs from frozen protocol')
    for field in ['sources','evidence','assets']:
        for path,expected in lock[field].items():
            if sha(path)!=expected:raise ValueError(f'Locked {field} identity mismatch: {path}')
    if len(lock['models'])!=7 or {m['name'] for m in lock['models']}!={
        'initial',*[f'{arm}_s{seed}' for arm in ['native','fp32'] for seed in [42,1337,2026]]}:
        raise ValueError('All seven registered models required')
    for item in lock['models']:
        if sha(item['checkpoint'])!=item['checkpoint_sha256']:raise ValueError('Checkpoint identity changed')


def prepare(out):
    from slr_common.data.manifest import load_manifest
    root=ROOT/'artifacts/slret_goal'
    replaypath=root/'cico-checkpoint-replay-001/run.json'
    replay=json.loads(replaypath.read_text())
    assert replay['status']=='completed' and len(replay['checkpoint_replays'])==12
    evidence={str(replaypath):sha(replaypath)}
    manifest=ROOT/'artifacts/manifests/ph_test.jsonl'
    assert sha(manifest)==TEST_SHA
    rows=load_manifest(manifest,expected_split='test')
    assert len(rows)==len({r.video_id for r in rows})==len({r.caption_id for r in rows})==642
    testids={r.video_id for r in rows}
    for split in ['train','dev']:
        path=ROOT/f'artifacts/manifests/ph_{split}.jsonl'
        assert not testids&{r.video_id for r in load_manifest(path,expected_split=split)}
        evidence[str(path)]=sha(path)
    modelitems=[]
    cfg=None
    for seed in [42,1337,2026]:
        auditname='cico-numeric-audit-001' if seed==42 else f'cico-repl-ph-s{seed}-audit-001'
        auditpath=root/auditname/'run.json'
        audit=json.loads(auditpath.read_text())
        assert audit['status']=='completed'
        evidence[str(auditpath)]=sha(auditpath)
        for reportpath,expected in audit['source_reports'].items():
            assert sha(reportpath)==expected
            report=json.loads(Path(reportpath).read_text())
            assert report['seed']==seed and report['optimizer_updates']==260 and report['selection']['step']==0
            evidence[reportpath]=expected
            current=report['config']
            if cfg is None:cfg=current.copy()
            assert {k:v for k,v in current.items() if k!='seed'}=={k:v for k,v in cfg.items() if k!='seed'}
            modelitems.append(dict(name=f'{report["arm"]}_s{seed}',seed=seed,arm=report['arm'],
                                   checkpoint=report['final_checkpoint'],checkpoint_sha256=report['final_checkpoint_sha256'],
                                   envelope='final',source_report=reportpath))
            if seed==42 and report['arm']=='native':
                modelitems.insert(0,dict(name='initial',seed=42,arm='initial',checkpoint=report['selection']['checkpoint'],
                                        checkpoint_sha256=report['selection']['sha256'],envelope='historical'))
    assert cfg['data']['alpha']==.9 and cfg['data']['feature_len']==64 and cfg['data']['max_words']==32
    assets={}
    for row in rows:
        for stream,path in [('agnostic',row.feature_agnostic),('aware',row.feature_aware)]:
            expectedroot=Path(cfg['sources'][f'feature_{stream}_root']).resolve()
            assert Path(path).resolve().is_relative_to(expectedroot)
            assets[str(Path(path))]=sha(path)
        assert row.temporal_metadata is not None
        assets[str(Path(row.temporal_metadata))]=sha(row.temporal_metadata)
    files=[Path(__file__),PROTOCOL,
           ROOT/'methods/elsc/elsc/evaluate.py',ROOT/'methods/elsc/elsc/models/retriever.py',
           ROOT/'shared/slr_common/evaluation/runtime.py',ROOT/'shared/slr_common/evaluation/cico_eval.py',
           ROOT/'shared/slr_common/upstream/factory.py',ROOT/'shared/slr_common/upstream/cico_bridge.py',
           ROOT/'shared/slr_common/data/cico_dataset.py',ROOT/'shared/slr_common/data/tokenize.py',
           ROOT/'third_party/SLRT/CiCo/CLCL/modules/modeling.py',ROOT/'third_party/SLRT/CiCo/CLCL/modules/module_clip.py']
    lock=dict(schema_version=1,created_unix=time.time(),test_scores_accessed=False,
              test_manifest=str(manifest),test_manifest_sha256=TEST_SHA,config=cfg,models=modelitems,
              sources={str(p):sha(p) for p in files},evidence=evidence,assets=assets,
              decision='Fixedendpoint numerical confirmation; allselectedPHmodels are sharedinitialization; no TESTselection.')
    verify_lock(lock)
    atomic_json(out/'lock.json',lock)
    return dict(lock_path=str(out/'lock.json'),lock_sha256=sha(out/'lock.json'),
                locked_models=len(modelitems),verified_test_assets=len(assets),test_scores_accessed=False)


def evaluate(out,path,expected):
    assert sha(path)==expected,'Explicit lockSHA mismatch'
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
        checkpoint=torch.load(item['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        cfg=lock['config']
        if item['envelope']=='final':assert checkpoint['completed_steps']==260
        state={n.removeprefix('core.'):v for n,v in checkpoint['model'].items() if n.startswith('core.')}
        core=_load_cico_core_from_state(cfg,state,cico_root=ROOT/cfg['upstream']['cico_root'],device='cuda')
        model=ELSCRetriever(core,input_dim=1024,hidden_dim=256,text_dim=core.clip.text_projection.shape[-1],
                           core_frozen=False,adapter_enabled=False).cuda()
        model.load_state_dict(checkpoint['model'],strict=True)
        assert all(torch.equal(v.cpu(),checkpoint['model'][n]) for n,v in model.state_dict().items())
        model.eval().requires_grad_(False)
        scores,metrics=evaluate_model(model,cfg,'test',torch.device('cuda'))
        assert scores.shape==(642,642) and metrics['gallery']==dict(videos=642,texts=642)
        folder=out/item['name']
        folder.mkdir()
        np.save(folder/'scores_video_x_text.npy',scores)
        compact={k:v for k,v in metrics.items() if k not in ['per_query','diagnostic_best_positive']}
        atomic_json(folder/'metrics.json',compact)
        # Preserve per-query ties for registered clusterbootstrap, not giant rankinglists.
        perquery={d:[{k:v for k,v in row.items() if k!='ranked_candidate_ids'} for row in metrics['per_query'][d]]
                  for d in ['T2V','V2T']}
        atomic_json(folder/'per_query.json',perquery)
        results[item['name']]=dict(checkpoint_sha256=item['checkpoint_sha256'],
            score_sha256=sha(folder/'scores_video_x_text.npy'),metrics_sha256=sha(folder/'metrics.json'),
            per_query_sha256=sha(folder/'per_query.json'),
            metrics={d:{k:v for k,v in metrics[d].items() if k!='cols'} for d in ['T2V','V2T']})
        atomic_json(out/'progress.json',dict(completed=len(results),results=results))
        print(json.dumps(dict(completed=item['name'],count=len(results))),flush=True)
        del model,core,checkpoint,state,scores,metrics,compact,perquery
        gc.collect()
        torch.cuda.empty_cache()
    return dict(lock_path=str(path),lock_sha256=expected,results=results,test_scores_accessed=True,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),torch=torch.__version__,hardware=torch.cuda.get_device_name())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--mode',choices=['lock','evaluate'],required=True)
    parser.add_argument('--lock')
    parser.add_argument('--lock-sha256')
    cli=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    start=time.time()
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
            report.update(evaluate(out,cli.lock,cli.lock_sha256))
        report.update(status='completed',exit_status=0)
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds']=time.time()-start
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':main()
