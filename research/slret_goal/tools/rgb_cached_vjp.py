"""Replay native SEDS representation gradients in the original RGB runtime."""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

from extraction_resume import atomic_json
from inventory import ROOT, sha
from rgb_gradient_feasibility import timed
from rgb_gradient_replay import window_batch, unfreeze_i3d_tail
from seds_runtime import compatible_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    cli = parser.parse_args()
    out = ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,
                  script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('rgb_gradient_replay.py')),
                  protocol_sha256=sha(ROOT/'research/slret_goal/RGB_GRADIENT_FEASIBILITY.md'),
                  optimizer_updates=0,test_loaded=False,dev_loaded=False,torch=torch.__version__,videos=[])
    atomic_json(out/'run.json',report)
    try:
        assert shutil.disk_usage(out).free > 15*1024**3
        assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file()) < 15*1024**3
        sys.path.insert(0,str(ROOT/'shared'))
        from slr_common.features.i3d import ExtractionRecipe,decode_video,load_i3d
        import cv2
        source = ROOT/'artifacts/slret_goal/rgb-gradient-feasibility-002'
        source_report = json.loads((source/'run.json').read_text())
        assert source_report['first_control_loss_maxabs'] == 0
        assert source_report['optimizer_updates'] == 0
        cached = torch.load(source/'representation_gradients.pt',map_location='cpu',weights_only=True)
        ids,grad = cached['ids'],cached['gradient']
        norms = grad.flatten(1).norm(dim=1)
        selected = torch.where(norms>0)[0][:2].tolist()
        assert len(selected)==2 and torch.isfinite(grad).all()
        report.update(source_run=source_report['run_id'],source_report_sha256=sha(source/'run.json'),
                      gradient_sha256=sha(source/'representation_gradients.pt'),
                      selection_rule='First two batch indices with nonzero actual representation gradient; activation only.',
                      selected_indices=selected,all32_gradient_norms=norms.tolist(),
                      zero_gradient_videos=int((norms==0).sum()),opencv=cv2.__version__,
                      hardware=torch.cuda.get_device_name())
        torch.set_num_threads(4)
        device = torch.device('cuda:0')
        checkpoint = ROOT/'artifacts/pretrained/bsl5k.pth.tar'
        implementation = ROOT/'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py'
        model = load_i3d(checkpoint,implementation,device)
        parameters = unfreeze_i3d_tail(model)
        before_bn = {n:b.clone() for n,b in model.named_buffers()}
        recipe = ExtractionRecipe()
        root = ROOT/'artifacts/slret_goal/seds-adapted-train-001'
        report.update(i3d_checkpoint_sha256=sha(checkpoint),i3d_source_sha256=sha(implementation),
                      trainable_tail_parameters=sum(p.numel() for p in parameters.values()),
                      trainable_tail_tensors=len(parameters))
        torch.cuda.reset_peak_memory_stats()
        for pos,idx in enumerate(selected):
            vid=ids[idx]
            meta_path=root/'metadata'/f'{vid}.json'
            meta=json.loads(meta_path.read_text())
            assert sha(meta['raw_path'])==meta['raw_sha256']
            decoded,decode_seconds=timed(lambda: decode_video(Path(meta['raw_path']),recipe))
            frames,fps=decoded
            assert fps==meta['fps']
            frames=frames[meta['retained_frame_indices']].to(device)
            starts=meta['clip_starts']
            target_path=root/'rgb/train'/f'{vid}.pkl'
            assert sha(target_path)==meta['rgb_sha256']
            target=torch.from_numpy(compatible_load(target_path.open('rb'))['feature']).to(device)
            g=grad[idx,:,:len(starts),0].T.contiguous().to(device)
            def forward_all():
                with torch.no_grad():
                    return torch.cat([model(window_batch(frames,starts[s:s+8],recipe,device))['embds'].flatten(1)
                                      for s in range(0,len(starts),8)])
            encoded,fwd=timed(forward_all)
            delta=float((encoded-target).abs().max())
            assert delta<=1e-4,('Feature parity',vid,delta)
            def backward_all():
                model.zero_grad(set_to_none=True)
                for s in range(0,len(starts),8):
                    model(window_batch(frames,starts[s:s+8],recipe,device))['embds'].flatten(1).backward(g[s:s+8])
            _,bwd=timed(backward_all)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values())
            norm=float(torch.sqrt(sum(p.grad.double().square().sum() for p in parameters.values())))
            assert norm>0
            row=dict(id=vid,batch_index=idx,decoded_frames=meta['decoded_frames'],windows=len(starts),
                     metadata_sha256=sha(meta_path),decode_seconds=decode_seconds,forward_seconds=fwd,
                     replay_backward_seconds=bwd,feature_maxabs=delta,parameter_gradient_norm=norm,
                     nonzero_gradient_tensors=sum(bool(torch.count_nonzero(p.grad)) for p in parameters.values()))
            if pos==0:
                assert torch.count_nonzero(g[:8])>0
                model.zero_grad(set_to_none=True)
                model(window_batch(frames,starts[:8],recipe,device))['embds'].flatten(1).backward(g[:8])
                reference={n:p.grad.detach().clone() for n,p in parameters.items()}
                model.zero_grad(set_to_none=True)
                for s in [0,4]:
                    model(window_batch(frames,starts[s:s+4],recipe,device))['embds'].flatten(1).backward(g[s:s+4])
                denominator=sum(v.double().square().sum() for v in reference.values())
                numerator=sum((parameters[n].grad.double()-v.double()).square().sum() for n,v in reference.items())
                relative=float((numerator/denominator.clamp_min(1e-30)).sqrt())
                absolute=max(float((parameters[n].grad-v).abs().max()) for n,v in reference.items())
                row['chunk_vs_full_gradient']=dict(relative_L2=relative,maxabs=absolute)
                assert relative<=1e-4 or absolute<=1e-8,('Chunk VJP parity',relative,absolute)
            report['videos'].append(row)
            atomic_json(out/'run.json',report)
            del frames,decoded,encoded,g,target
        assert all(torch.equal(before_bn[n],b) for n,b in model.named_buffers())
        census=[json.loads(p.read_text()) for p in (root/'metadata').glob('*.json')]
        assert len(census)==7096
        total_windows=sum(len(m['clip_starts']) for m in census)
        total_frames=sum(m['decoded_frames'] for m in census)
        rows=report['videos']
        window_rate=max((r['forward_seconds']+r['replay_backward_seconds'])/r['windows'] for r in rows)
        decode_rate=max(r['decode_seconds']/r['decoded_frames'] for r in rows)
        control=json.loads((ROOT/'artifacts/slret_goal/seds-moment-control-001/run.json').read_text())
        report.update(status='completed',exit_status=0,batchnorm_buffers_unchanged=True,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      provisional_full_train_projection=dict(videos=7096,windows=total_windows,decoded_frames=total_frames,
                          head_control_wall_seconds=control['wall_seconds'],
                          rgb_seconds_at_worst_observed_rates=window_rate*total_windows+decode_rate*total_frames,
                          conservative_seconds_with_50pct_rgb_margin=control['wall_seconds']+1.5*(window_rate*total_windows+decode_rate*total_frames),
                          limitations='Two activated videos only; excludes IPC, new optimizer and DEV extraction. Not admission.'),
                      decision='Cross-runtime gradient replay feasibility only; no accuracy result.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__': main()
