"""Bounded inference-cost diagnostic with unchanged SEDS retrieval scores."""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from extraction_resume import atomic_json, digest
from inventory import ROOT, sha
from seds_runtime import eval_dataset, patch_pickle, verify_assets


def normalized(x):
    return (x / x.norm(dim=-1, keepdim=True)).squeeze(1)


def dense_score(video, text, vm, tm, scale, block):
    """Native FILIP softmax algebra, sharing the identical inference dot products."""
    output = torch.empty((len(video), len(text)), device=video.device)
    for i in range(0, len(video), block):
        vmask = (vm[i:i+block] == 0)[:, None, :]
        for j in range(0, len(text), block):
            tmask = (tm[j:j+block] == 1)[None, :, :]
            dots = torch.einsum('ais,bjs->abij', video[i:i+block], text[j:j+block])
            a = torch.nansum(dots*torch.softmax(dots/.07, dim=3), dim=3)
            b = torch.nansum(dots*torch.softmax(dots/.07, dim=2), dim=2)
            a = a.masked_fill(~vmask, 0)
            b = b.masked_fill(~tmask, 0)
            i2t = scale*torch.nansum(a, dim=-1)/vmask.sum(-1)
            t2i = scale*torch.nansum(b*tmask, dim=-1)/tmask.sum(-1)
            output[i:i+block, j:j+block] = .5*i2t+.5*t2i
    return output


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter()-start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    cli = parser.parse_args()
    out = ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = dict(run_id=cli.run_id, status='running', pid=os.getpid(), command=sys.argv,
                  test_loaded=False, optimizer_updates=0, script_sha256=sha(__file__),
                  protocol_sha256=sha(ROOT/'research/slret_goal/SCORING_EFFICIENCY_PROTOCOL.md'))
    atomic_json(out/'run.json', report)
    try:
        campaign_bytes = sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())
        assert campaign_bytes+256*1024**2 < 15*1024**3
        assert shutil.disk_usage(out).free > 15*1024**3
        torch.set_num_threads(4)
        sys.path[:0] = [str(ROOT/'shared'), str(ROOT/'third_party/SEDS')]
        os.chdir(ROOT/'third_party/SEDS')
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose, ph_pose_collate_fn
        from slr_common.evaluation.cico_eval import evaluate_score_matrix
        patch_pickle(ph_DataLoader_pose)
        refroot = ROOT/'artifacts/slret_goal/seds-adapted-dev-eval-002'
        ref = json.loads((refroot/'run.json').read_text())
        args = argparse.Namespace(**ref['config'])
        args.output_dir, args.local_rank = str(out), int(os.environ.get('LOCAL_RANK',0))
        args.do_eval = False  # suppress redundant pickle exports, not model behavior
        args.batch_size_val, args.num_thread_reader = 32, 0
        args = native.set_seed_logger(args)
        device, ngpu = native.init_device(args, args.local_rank)
        model = native.init_model(args, device).eval()
        expected = torch.load(args.init_model, map_location='cpu')
        assert set(expected) == set(model.state_dict())
        assert all(torch.equal(v.cpu(), model.state_dict()[k].cpu()) for k,v in expected.items())
        del expected
        data, ids = eval_dataset(ph_DataLoader_pose, args, native.ClipTokenizer(),
                                ROOT/'artifacts/slret_goal/seds-adapted-dev-001')
        _, assets = verify_assets(ROOT/'artifacts/slret_goal/seds-adapted-dev-001', 'dev', ids)
        loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=0,
                                             collate_fn=ph_pose_collate_fn)
        captured = {}
        calls = []
        handle = model.fusion.register_forward_hook(lambda *a: calls.append(1))
        original = native._run_on_single_gpu_new_mix

        def capture(model_arg, vm, tm, text, pose, rgb, *a, **kw):
            result, seconds = timed(lambda: original(model_arg, vm, tm, text, pose, rgb, *a, **kw))
            captured.update(vm=vm, tm=tm, text=text, pose=pose, rgb=rgb,
                            score_seconds=seconds, native=result)
            return result

        native._run_on_single_gpu_new_mix = capture
        torch.cuda.reset_peak_memory_stats()
        _, total = timed(lambda: native.eval_epoch(args, model, loader, device, ngpu, False))
        native._run_on_single_gpu_new_mix = original
        handle.remove()
        report.update(native_eval_seconds=total, native_scoring_seconds=captured['score_seconds'],
                      native_fusion_calls=len(calls), hardware=torch.cuda.get_device_name(),
                      torch=torch.__version__, checkpoint_sha256=sha(args.init_model),
                      dev_assets_digest=digest(assets), peak_cuda_native=torch.cuda.max_memory_allocated())
        with torch.inference_mode():
            fused, fseconds = timed(lambda: [model.fusion(p,r,m) for p,r,m in
                           zip(captured['pose'],captured['rgb'],captured['vm'])])
            video = {name:torch.cat([normalized(x) for x in values]) for name,values in
                     [('fusion',fused),('pose',captured['pose']),('rgb',captured['rgb'])]}
            text = torch.cat([normalized(x) for x in captured['text']])
            vm, tm = torch.cat(captured['vm']), torch.cat(captured['tm'])
            scale = model.clip.logit_scale.exp()
            reference = {name:np.concatenate(captured['native'][idx], axis=0) for name,idx in
                         [('fusion',0),('pose',2),('rgb',4)]}
            assert max(float(np.max(np.abs(reference[s]-np.load(refroot/f'{s}_video_x_text.npy'))))
                       for s in reference) <= 1e-4
            mapping = {x:[x] for x in ids}
            refmetrics = {s:evaluate_score_matrix(m,video_ids=ids,text_ids=ids,
                          video_to_text=mapping,text_to_video=mapping) for s,m in reference.items()}
            report['cached_fusion_seconds'] = fseconds
            report['cached_fusion_calls'] = len(fused)
            report['profiles'] = {}
            for label, streams, block in [('all_streams_b32',['fusion','pose','rgb'],32),
                        ('fusion_b32',['fusion'],32),('fusion_b128',['fusion'],128)]:
                def execute():
                    return {s:dense_score(video[s],text,vm,tm,scale,block) for s in streams}
                execute()  # warmup excluded
                timings = []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(5):
                    result, seconds = timed(execute)
                    timings.append(seconds)
                checks = {}
                for s, matrix in result.items():
                    arr = matrix.cpu().numpy()
                    metric = evaluate_score_matrix(arr,video_ids=ids,text_ids=ids,
                                      video_to_text=mapping,text_to_video=mapping)
                    changes = {d:sum(a!=b for a,b in zip(metric[d]['cols'],refmetrics[s][d]['cols']))
                               for d in ['T2V','V2T']}
                    delta = float(np.max(np.abs(arr-reference[s])))
                    checks[s] = dict(maxabs=delta,rank_changes=changes)
                    assert delta <= 1e-4 and not any(changes.values()), (label,s,checks[s])
                    np.save(out/f'{label}_{s}.npy',arr)
                report['profiles'][label] = dict(seconds=timings, median_seconds=float(np.median(timings)),
                       checks=checks, peak_cuda_bytes=torch.cuda.max_memory_allocated())
            cache = dict(video=video['fusion'].cpu(),text=text.cpu(),video_mask=vm.cpu(),text_mask=tm.cpu(),
                         logit_scale=scale.cpu(),ids=ids,checkpoint_sha256=sha(args.init_model),
                         normalization='native_L2_before_dot_product',temperature=.07,dual_mix=.5)
            torch.save(cache,out/'normalized_fusion_cache.pt')
        report['cache_sha256'] = sha(out/'normalized_fusion_cache.pt')
        report['source_hashes'] = {str(p):sha(p) for p in [Path(__file__),ROOT/'third_party/SEDS/main_task_retrieval.py',
                    ROOT/'third_party/SEDS/modules/modeling.py',ROOT/'third_party/SEDS/modules/module_fusionencoder.py']}
        best = min(report['profiles'][x]['median_seconds'] for x in ['fusion_b32','fusion_b128'])
        report.update(status='completed',exit_status=0,optimized_dense_above_100ms=best>.1,
                      decision='Routine optimized baseline only; investigate certified screening only if dense scoring still exceeds100ms.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps(report,indent=2))
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
