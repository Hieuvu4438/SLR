"""Real SEDS feature-gradient to I3D VJP/throughput check, zero optimizer updates."""
import argparse
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from extraction_resume import atomic_json
from inventory import ROOT, sha
from rgb_gradient_replay import window_batch, unfreeze_i3d_tail
from seds_runtime import model_inputs, native_kwargs, patch_pickle


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value,time.perf_counter()-start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli = parser.parse_args()
    out = ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,
                  test_loaded=False,dev_loaded=False,optimizer_updates=0,script_sha256=sha(__file__),
                  helper_sha256=sha(Path(__file__).with_name('rgb_gradient_replay.py')),
                  protocol_sha256=sha(ROOT/'research/slret_goal/RGB_GRADIENT_FEASIBILITY.md'))
    atomic_json(out/'run.json',report)
    try:
        assert shutil.disk_usage(out).free > 15*1024**3
        size = sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())
        assert size+32*1024**2 < 15*1024**3
        torch.set_num_threads(4)
        sys.path[:0] = [str(ROOT/'shared'),str(ROOT/'third_party/SEDS')]
        os.chdir(ROOT/'third_party/SEDS')
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_train_pose import ph_DataLoader_train_pose,ph_train_pose_collate_fn
        from slr_common.features.i3d import ExtractionRecipe,decode_video,load_i3d
        import cv2
        report.update(opencv=cv2.__version__,numpy=np.__version__,
                      extraction_environment_opencv='4.13.0',
                      environment_note='Native SEDS keeps NumPy1.26.4; feature parity remains mandatory.')
        patch_pickle(ph_DataLoader_train_pose)
        control = ROOT/'artifacts/slret_goal/seds-moment-control-001'
        ctrl = json.loads((control/'run.json').read_text())
        args = argparse.Namespace(**ctrl['config'])
        args.output_dir,args.local_rank = str(out),int(os.environ.get('LOCAL_RANK',0))
        args = native.set_seed_logger(args)
        device,_ = native.init_device(args,args.local_rank)
        trainroot = ROOT/'artifacts/slret_goal/seds-adapted-train-001'
        data = ph_DataLoader_train_pose(**native_kwargs(args,native.ClipTokenizer(),trainroot,'train'))
        labels = pickle.load((trainroot/'labels/train.pkl').open('rb'))
        indices = json.loads((control/'batch_indices.json').read_text())[0]
        ids = [list(labels)[i] for i in indices]
        model = native.init_model(args,device).train().requires_grad_(False)
        expected = torch.load(args.init_model,map_location='cpu')
        assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in expected.items())
        del expected
        batch = ph_train_pose_collate_fn([data[i] for i in indices])
        rgb = batch['RGB_feature'].to(device).detach().requires_grad_(True)
        batch['RGB_feature'] = rgb
        torch.cuda.reset_peak_memory_stats()
        def head_backward():
            losses = model(*model_inputs(batch,device))
            losses[0].backward()
            return [float(x) for x in losses]
        losses,head_seconds = timed(head_backward)
        grad = rgb.grad.detach()
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad)>0
        first = json.loads((control/'train_steps.jsonl').read_text().splitlines()[0])
        report.update(ids=ids,indices=indices,losses=losses,head_gradient_seconds=head_seconds,
                      first_control_loss_maxabs=max(abs(a-b) for a,b in zip(losses,first['loss'])),
                      rgb_input_shape=list(rgb.shape),rgb_gradient_norm=float(grad.norm()),
                      rgb_gradient_nonzero=int(torch.count_nonzero(grad)),rgb_gradient_elements=grad.numel(),
                      peak_head_cuda_bytes=torch.cuda.max_memory_allocated(),hardware=torch.cuda.get_device_name(),
                      torch=torch.__version__,seds_checkpoint_sha256=sha(args.init_model))
        torch.save(dict(ids=ids,gradient=grad.cpu()),out/'representation_gradients.pt')
        checkpoint = ROOT/'artifacts/pretrained/bsl5k.pth.tar'
        implementation = ROOT/'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py'
        i3d = load_i3d(checkpoint,implementation,device)
        parameters = unfreeze_i3d_tail(i3d)
        before_bn = {n:b.clone() for n,b in i3d.named_buffers()}
        recipe = ExtractionRecipe()
        report.update(i3d_checkpoint_sha256=sha(checkpoint),i3d_source_sha256=sha(implementation),
                      recipe_sha256=recipe.digest,trainable_tail_parameters=sum(p.numel() for p in parameters.values()),
                      trainable_tail_tensors=len(parameters),videos=[])
        for idx,vid in enumerate(ids[:2]):
            meta_path = trainroot/'metadata'/f'{vid}.json'
            meta = json.loads(meta_path.read_text())
            assert sha(meta['raw_path']) == meta['raw_sha256']
            frames,seconds = timed(lambda: decode_video(Path(meta['raw_path']),recipe))
            frames,fps = frames
            assert fps == meta['fps']
            frames = frames[meta['retained_frame_indices']].to(device)
            starts = meta['clip_starts']
            g = grad[idx,:,:len(starts),0].T.contiguous()
            target = rgb.detach()[idx,:,:len(starts),0].T
            def forward_all():
                with torch.no_grad():
                    return torch.cat([i3d(window_batch(frames,starts[s:s+8],recipe,device))['embds'].flatten(1)
                                      for s in range(0,len(starts),8)])
            encoded,fwd = timed(forward_all)
            delta = float((encoded-target).abs().max())
            assert delta <= 1e-4, ('Feature parity',vid,delta)
            def backward_all():
                i3d.zero_grad(set_to_none=True)
                for s in range(0,len(starts),8):
                    emb = i3d(window_batch(frames,starts[s:s+8],recipe,device))['embds'].flatten(1)
                    emb.backward(g[s:s+8])
            _,bwd = timed(backward_all)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values())
            norm = float(torch.sqrt(sum(p.grad.double().square().sum() for p in parameters.values())))
            assert norm > 0
            row = dict(id=vid,windows=len(starts),decoded_frames=meta['decoded_frames'],
                       metadata_sha256=sha(meta_path),decode_seconds=seconds,
                       forward_seconds=fwd,replay_backward_seconds=bwd,feature_maxabs=delta,
                       parameter_gradient_norm=norm,
                       nonzero_parameter_gradient_tensors=sum(bool(torch.count_nonzero(p.grad)) for p in parameters.values()))
            if idx == 0:
                assert len(starts)>=8
                i3d.zero_grad(set_to_none=True)
                i3d(window_batch(frames,starts[:8],recipe,device))['embds'].flatten(1).backward(g[:8])
                reference = {n:p.grad.detach().clone() for n,p in parameters.items()}
                i3d.zero_grad(set_to_none=True)
                for s in [0,4]:
                    i3d(window_batch(frames,starts[s:s+4],recipe,device))['embds'].flatten(1).backward(g[s:s+4])
                denominator = sum(v.double().square().sum() for v in reference.values())
                numerator = sum((parameters[n].grad.double()-v.double()).square().sum() for n,v in reference.items())
                relative = float((numerator/denominator.clamp_min(1e-30)).sqrt())
                absolute = max(float((parameters[n].grad-v).abs().max()) for n,v in reference.items())
                assert relative<=1e-4 or absolute<=1e-8, ('Chunk VJP parity',relative,absolute)
                row['chunk_vs_full_gradient'] = dict(relative_L2=relative,maxabs=absolute)
            report['videos'].append(row)
            atomic_json(out/'run.json',report)
            del frames,encoded,g,target
        assert all(torch.equal(before_bn[n],b) for n,b in i3d.named_buffers())
        census = [json.loads(p.read_text()) for p in (trainroot/'metadata').glob('*.json')]
        assert len(census) == 7096
        total_windows = sum(len(m['clip_starts']) for m in census)
        total_frames = sum(m['decoded_frames'] for m in census)
        rows = report['videos']
        window_rate = max((r['forward_seconds']+r['replay_backward_seconds'])/r['windows'] for r in rows)
        decode_rate = max(r['decode_seconds']/r['decoded_frames'] for r in rows)
        report['provisional_full_train_projection'] = dict(
            videos=len(census),decoded_frames=total_frames,windows=total_windows,
            head_control_wall_seconds=ctrl['wall_seconds'],
            rgb_seconds_at_worst_observed_rates=window_rate*total_windows+decode_rate*total_frames,
            conservative_seconds_with_50pct_rgb_margin=ctrl['wall_seconds']+1.5*(window_rate*total_windows+decode_rate*total_frames),
            limitations='Only two videos; excludes fresh DEV RGB extraction and future optimizer overhead. Not long-run admission.')
        report.update(status='completed',exit_status=0,batchnorm_buffers_unchanged=True,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      gradient_sha256=sha(out/'representation_gradients.pt'),
                      decision='Gradient/feature feasibility only; no efficacy or long-run admission.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps(report,indent=2))
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
