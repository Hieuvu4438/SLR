"""Fresh-process full-gallery DEV replay of every audited final CiCo model."""
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


def require_exact(matrix,metrics,expected_matrix,expected_metrics):
    if not np.array_equal(matrix,expected_matrix):
        raise ValueError(f'Score mismatch; maxabs={np.max(np.abs(matrix-expected_matrix))}')
    for key in ['T2V','V2T','gallery','metric_kernel','id_hashes']:
        if metrics[key]!=expected_metrics[key]:raise ValueError(f'Checkpoint replay {key} mismatch')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli=parser.parse_args()
    root=ROOT/'artifacts/slret_goal'
    queue=json.loads((root/'cico-numeric-replication-001/run.json').read_text())
    if queue['status']!='completed':raise ValueError('Wait for terminal completed replication queue')
    source_reports={}
    audits=['cico-numeric-audit-001']+[f'cico-repl-{d}-s{s}-audit-001' for d,s in queue['fixed_pairs']]
    for name in audits:
        audit=json.loads((root/name/'run.json').read_text())
        assert audit['status']=='completed' and audit['exit_status']==0
        source_reports.update(audit['source_reports'])
    assert len(source_reports)==12
    out=root/cli.run_id
    out.mkdir(exist_ok=False)
    started=time.time()
    report=dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,
                test_loaded=False,optimizer_updates=0,script_sha256=sha(__file__),
                protocol_sha256=sha(ROOT/'research/slret_goal/CICO_CHECKPOINT_REPLAY_PROTOCOL.md'),
                checkpoint_replays=[])
    def record():
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
    record()
    try:
        assert shutil.disk_usage(out).free>15*1024**3
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())+100*1024**2<44*1024**3
        torch.set_num_threads(4)
        sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'methods/elsc'),str(ROOT/'third_party/SLRT/CiCo/CLCL')]
        from elsc.models.retriever import ELSCRetriever
        from elsc.train import _seed_everything
        from elsc.evaluate import evaluate_model
        from slr_common.upstream.factory import _load_cico_core_from_state
        verified_features=set()
        torch.cuda.reset_peak_memory_stats()
        for source_path,expected_hash in source_reports.items():
            source_path=Path(source_path)
            assert sha(source_path)==expected_hash
            source=json.loads(source_path.read_text())
            assert source['status']=='completed' and source['torch']==torch.__version__
            assert source['hardware']==torch.cuda.get_device_name()
            for name,expected in source['source_sha256'].items():assert sha(ROOT/name)==expected
            features=source_path.parent/'feature_hashes.json'
            assert sha(features)==source['feature_hashes_sha256']
            if sha(features) not in verified_features:
                for name,expected in json.loads(features.read_text()).items():assert sha(name)==expected
                verified_features.add(sha(features))
            for split,expected in source['manifests'].items():
                assert sha(ROOT/source['config']['data'][f'{split}_manifest'])==expected
            checkpoint=Path(source['final_checkpoint'])
            assert sha(checkpoint)==source['final_checkpoint_sha256']
            _seed_everything(source['seed'])
            raw=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
            cfg=source['config']
            assert raw['config']==cfg and raw['completed_steps']==source['optimizer_updates']
            state={n.removeprefix('core.'):v for n,v in raw['model'].items() if n.startswith('core.')}
            core=_load_cico_core_from_state(cfg,state,cico_root=ROOT/cfg['upstream']['cico_root'],device='cuda')
            model=ELSCRetriever(core,input_dim=1024,hidden_dim=256,text_dim=core.clip.text_projection.shape[-1],
                               core_frozen=False,adapter_enabled=False).cuda()
            model.load_state_dict(raw['model'],strict=True)
            assert all(torch.equal(v.cpu(),raw['model'][n]) for n,v in model.state_dict().items())
            model.eval().requires_grad_(False)
            scores,metrics=evaluate_model(model,cfg,'dev',torch.device('cuda'))
            original=source_path.parent/f'eval_step{source["optimizer_updates"]:03d}'
            expected_metrics=json.loads((original/'metrics.json').read_text())
            expected_scores=np.load(original/'scores_video_x_text.npy')
            require_exact(scores,metrics,expected_scores,expected_metrics)
            path=out/f'{source["run_id"]}_scores.npy'
            np.save(path,scores)
            entry=dict(source_run=source['run_id'],checkpoint_sha256=sha(checkpoint),
                       source_report_sha256=expected_hash,replayed_score_sha256=sha(path),
                       score_maxabs=0.,all_metrics_ranks_ids_exact=True,gallery=metrics['gallery'],
                       metrics={d:{k:v for k,v in metrics[d].items() if k!='cols'} for d in ['T2V','V2T']})
            report['checkpoint_replays'].append(entry)
            record()
            print(json.dumps(dict(replayed=source['run_id'],count=len(report['checkpoint_replays']),score_maxabs=0.)),flush=True)
            del model,core,raw,state,scores,metrics,expected_scores,expected_metrics
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status='completed',exit_status=0,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      decision='All12 DEV checkpoint replays exact; not independent dataset confirmation or SOTA.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        record()
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps({k:report.get(k) for k in ['run_id','status','wall_seconds','error']},indent=2))


if __name__=='__main__':main()
