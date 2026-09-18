"""Full PH DEV inference from a verified partial corrected UPRet checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from inventory import ROOT, sha
from extraction_resume import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    cli = parser.parse_args()
    out = ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    report = dict(run_id=cli.run_id, status='running', pid=os.getpid(),
                  command=sys.argv, test_features_loaded=False, optimizer_updates=0,
                  selection_split='historically_exposed_PH_dev',
                  kind='partial_corrected_UPRet_not_release_or_method',
                  script_sha256=sha(__file__),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(out/'run.json',report)
    try:
        torch.set_num_threads(4)
        sys.path[:0] = [str(ROOT),str(ROOT/'shared')]
        from methods.sssc.method1.config import load_config
        from methods.sssc.method1.model_factory import _activate_upret_import, _load_clip_state, config_to_upret
        from methods.sssc.method1.inference import _encode_inference_pool
        from methods.sssc.method1.baseline import directional_scores_blocked
        from methods.sssc.method1.evaluation import evaluate_grouped_retrieval
        from methods.sssc.method1.upstream import create_upret_tokenizer
        from methods.sssc.method1.data import _load_records
        from methods.sssc.method1.schemas import VideoRecord
        from slr_common.evaluation.cico_eval import evaluate_score_matrix
        root=ROOT/'runs/method1/ph/base/seed42'
        manifest=json.loads((root/'run_manifest.json').read_text())
        bridge=json.loads((ROOT/'artifacts/slret_goal/upret-load-check-002/run.json').read_text())
        assert bridge['status']=='completed'
        for name,digest in bridge['verified_upret_source_sha256'].items():
            assert sha(ROOT/'third_party/UPRet'/name.removeprefix('upret/'))==digest
        for name,digest in manifest['artifact_hashes']['implementation_source']['files'].items():
            if name.startswith('method1/'):
                assert sha(ROOT/'methods/sssc'/name)==digest, name
        config=load_config(ROOT/'methods/sssc/configs/method1/ph_seed42_base_initial.yaml')
        assert config.digest==bridge['config_sha256']
        for name,digest in manifest['artifact_hashes']['manifests'].items():
            assert sha(Path(config.data.manifest_dir)/name)==digest, name
        videos=[v for v in _load_records(Path(config.data.manifest_dir)/'videos.jsonl',VideoRecord) if v.split=='dev']
        assert len(videos)==519
        for v in videos:
            assert sha(v.agnostic_path)==v.agnostic_sha256
            assert sha(v.aware_path)==v.aware_sha256
        checkpoint=root/'best_dev.pt'
        assert sha(checkpoint)==bridge['checkpoint_sha256']
        upret=ROOT/'third_party/UPRet'
        CLIP4Clip,CrossConfig=_activate_upret_import(upret)
        clip=_load_clip_state(config.model.clip_checkpoint_path)
        assert sha(config.model.clip_checkpoint_path)==manifest['artifact_hashes']['clip_initialization']
        model=CLIP4Clip(CrossConfig(str(upret/'modules/cross-base/cross_config.json')),
                       clip.copy(),task_config=config_to_upret(config)).float()
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=payload['student_state_dict']
        model.load_state_dict(state,strict=True)
        assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
        report.update(checkpoint_sha256=bridge['checkpoint_sha256'],checkpoint_step=payload['global_step'],
                      training_run_complete=payload['training_run_complete'],
                      tensor_parity='exact',config_sha256=config.digest,
                      manifest_hashes=manifest['artifact_hashes']['manifests'],
                      source_hashes=bridge['verified_upret_source_sha256'],
                      dev_feature_files_verified=2*len(videos),
                      hardware=torch.cuda.get_device_name(),torch=torch.__version__)
        del payload,state,clip
        model=model.cuda().eval().requires_grad_(False)
        tokenizer=create_upret_tokenizer(upret,config.model.bpe_path)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            video,vmask,text,tmask,videos,groups=_encode_inference_pool(model,config,
                split='dev',tokenizer=tokenizer,device='cuda')
            assert len(videos)==len(groups)==519
            print('Encoded official PH DEV519',flush=True)
            channels=directional_scores_blocked(model,video.cuda(),vmask.cuda(),text.cuda(),tmask.cuda(),
                temperature=config.model.inner_similarity_temperature,video_block=32,text_block=64)
            scores=(config.model.dual_mix*channels[0]+(1-config.model.dual_mix)*channels[1]).cpu().numpy()
        np.save(out/'scores_video_x_text.npy',scores)
        group_indexes={g.group_uid:i for i,g in enumerate(groups)}
        mapping=[group_indexes[v.group_uid] for v in videos]
        assert mapping==list(range(519)), 'PH singleton order'
        metric=evaluate_grouped_retrieval(scores,video_group_indexes=mapping)
        metric.pop('grouped_t2v_scores')
        atomic_json(out/'metrics.json',metric)
        old=json.loads((root/'dev_step_767.json').read_text())
        assert old['query_ids']['V2T']==[v.video_uid for v in videos]
        assert old['query_ids']['T2V']==[g.group_uid for g in groups]
        rank_changes={d:sum(a!=b for a,b in zip(metric[d]['ranks'],old[d]['ranks'],strict=True)) for d in ['T2V','V2T']}
        assert not any(rank_changes.values()), ('Historical rank mismatch',rank_changes)
        # Independent rank calculation with historical stable-manifest ties.
        for direction,matrix in [('V2T',scores),('T2V',scores.T)]:
            rank=1+(matrix>matrix.diagonal()[:,None]).sum(1)+((matrix==matrix.diagonal()[:,None]) &
                (np.arange(519)[None,:]<np.arange(519)[:,None])).sum(1)
            assert rank.tolist()==metric[direction]['ranks']
        ids=[v.video_uid for v in videos]
        positives={x:[x] for x in ids}
        shared=evaluate_score_matrix(scores,video_ids=ids,text_ids=ids,video_to_text=positives,text_to_video=positives)
        # CiCo shared singleton tie policy differs; expose, never silently replace.
        report['shared_cico_policy_diagnostic']={d:{k:shared[d][k] for k in ['R1','R5','R10','MeanR']} for d in ['T2V','V2T']}
        report['shared_tie_stats']=shared['tie_stats']
        report.update(status='completed',exit_status=0,historical_rank_changes=rank_changes,
                      metrics={d:{k:metric[d][k] for k in ['R1','R5','R10','MedR','MeanR']} for d in ['T2V','V2T']},
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      score_sha256=sha(out/'scores_video_x_text.npy'),
                      historical_metrics_sha256=sha(root/'dev_step_767.json'),
                      decision='Historical partial corrected checkpoint replay; not complete baseline or method gain.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds']=time.time()-start
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
