"""Persistent original-runtime I3D worker; JSON stdin protocol, local tensors."""
import contextlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import torch

from inventory import ROOT, sha
from extraction_resume import atomic_json
from rgb_bridge_state import ReplayState
from rgb_gradient_replay import window_batch, unfreeze_i3d_tail
from seds_runtime import compatible_load


def main():
    output=Path(sys.argv[1]).resolve()
    pilot=len(sys.argv)>2 and sys.argv[2]=='--pilot'
    max_steps=222 if pilot else 2
    protocol_stdout=sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'third_party/SEDS')]
        from slr_common.features.i3d import ExtractionRecipe,decode_video,load_i3d
        from modules.optimization import BertAdam
        torch.set_num_threads(4)
        device=torch.device('cuda:0')
        model=load_i3d(ROOT/'artifacts/pretrained/bsl5k.pth.tar',
                       ROOT/'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py',device)
        params=unfreeze_i3d_tail(model)
        groups=[dict(params=[p for n,p in params.items() if not ('bias' in n or 'bn.' in n)],weight_decay=.001),
                dict(params=[p for n,p in params.items() if 'bias' in n or 'bn.' in n],weight_decay=0.)]
        optimizer=BertAdam(groups,lr=1e-5,b1=.9,b2=.98,e=1e-6,warmup=.1,
                           t_total=222,schedule='warmup_cosine',max_grad_norm=1.)
        frozen={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in params}
        recipe=ExtractionRecipe()
        root=ROOT/'artifacts/slret_goal/seds-adapted-train-001'
        state=ReplayState()
        cache=[]
        outputs=None

        def forward(frames,starts):
            with torch.no_grad():
                return torch.cat([model(window_batch(frames,starts[s:s+8],recipe,device))['embds'].flatten(1)
                                  for s in range(0,len(starts),8)])

        def reply(payload):
            print(json.dumps(payload),file=protocol_stdout,flush=True)

        reply(dict(status='ready',pid=os.getpid(),torch=torch.__version__))
        for line in sys.stdin:
            try:
                req=json.loads(line)
                cmd=req['op']
                version=req['version']
                start=time.time()
                if cmd=='encode':
                    state.require(version,'idle')
                    cache=[]
                    outputs=torch.zeros(len(req['ids']),1024,64,1)
                    maximum=0.
                    for i,vid in enumerate(req['ids']):
                        meta=json.loads((root/'metadata'/f'{vid}.json').read_text())
                        assert sha(meta['raw_path'])==meta['raw_sha256']
                        frames,fps=decode_video(Path(meta['raw_path']),recipe)
                        assert fps==meta['fps']
                        frames=frames[meta['retained_frame_indices']]
                        starts=meta['clip_starts']
                        result=forward(frames.to(device),starts).cpu()
                        path=root/'rgb/train'/f'{vid}.pkl'
                        assert sha(path)==meta['rgb_sha256']
                        expected=torch.from_numpy(compatible_load(path.open('rb'))['feature'])
                        delta=float((result-expected).abs().max())
                        maximum=max(maximum,delta)
                        if version<=1: assert delta<=1e-4,('Initial feature parity',vid,delta)
                        outputs[i,:,:len(starts),0]=result.T
                        cache.append((frames,starts))
                    path=output/f'features_v{version}.pt'
                    torch.save(dict(version=version,ids=req['ids'],features=outputs),path)
                    state.encoded(version)
                    reply(dict(status='encoded',version=version,path=str(path),sha256=sha(path),
                               feature_maxabs=maximum,seconds=time.time()-start,
                               windows=sum(len(s) for _,s in cache)))
                elif cmd=='backward':
                    state.require(version,'encoded')
                    path=output/f'gradient_v{version}.pt'
                    assert sha(path)==req['sha256']
                    payload=torch.load(path,map_location='cpu',weights_only=True)
                    assert payload['version']==version
                    gradients=payload['gradient']
                    assert gradients.shape==outputs.shape and torch.isfinite(gradients).all()
                    optimizer.zero_grad()
                    maximum=0.
                    for i,(host_frames,starts) in enumerate(cache):
                        frames=host_frames.to(device)
                        g=gradients[i,:,:len(starts),0].T.to(device)
                        for s in range(0,len(starts),8):
                            encoded=model(window_batch(frames,starts[s:s+8],recipe,device))['embds'].flatten(1)
                            delta=float((encoded.detach().cpu()-outputs[i,:,s:s+len(encoded),0].T).abs().max())
                            maximum=max(maximum,delta)
                            assert delta==0,('Stale/replay mismatch',version,i,s,delta)
                            encoded.backward(g[s:s+8])
                    norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
                    assert torch.isfinite(norm) and (norm>=0 if pilot else norm>0)
                    state.backward_ready(version)
                    reply(dict(status='backward_ready',version=version,gradient_norm=float(norm),
                               replay_maxabs=maximum,seconds=time.time()-start,
                               active_videos=int((gradients.flatten(1).norm(dim=1)>0).sum())))
                elif cmd=='commit':
                    state.require(version,'backward_ready')
                    before={n:p.detach().clone() for n,p in params.items()}
                    optimizer.step()
                    changed=sum(not torch.equal(before[n],p) for n,p in params.items())
                    assert (changed==0 if version==0 else (changed>0 if version==1 or not pilot else changed>=0))
                    assert all(torch.isfinite(p).all() for p in params.values())
                    assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in frozen.items())
                    state.committed(version)
                    reply(dict(status='committed',version=state.version,changed_tensors=changed,
                               frozen_and_bn_unchanged=True,seconds=time.time()-start))
                elif cmd=='checkpoint':
                    state.require(version,'idle')
                    assert pilot and version in [111,222]
                    path=output/f'tail_optimizer_step{version:03d}.pt'
                    torch.save(dict(tail={n:p.detach().cpu() for n,p in params.items()},optimizer=optimizer.state_dict(),
                                    completed_steps=version,backbone_sha256=sha(ROOT/'artifacts/pretrained/bsl5k.pth.tar'),
                                    rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)
                    reply(dict(status='checkpointed',version=version,path=str(path),sha256=sha(path)))
                elif cmd=='evaluate_features':
                    state.require(version,'idle')
                    assert pilot and version in [111,222]
                    devroot=ROOT/'artifacts/slret_goal/seds-adapted-dev-001'
                    folder=output/f'dev_rgb_step{version:03d}'/'dev'
                    folder.mkdir(parents=True,exist_ok=False)
                    manifest={}
                    for i,vid in enumerate(req['ids']):
                        meta=json.loads((devroot/'metadata'/f'{vid}.json').read_text())
                        assert sha(meta['raw_path'])==meta['raw_sha256']
                        frames,fps=decode_video(Path(meta['raw_path']),recipe)
                        assert fps==meta['fps']
                        features=forward(frames[meta['retained_frame_indices']].to(device),meta['clip_starts']).cpu().numpy()
                        assert __import__('numpy').isfinite(features).all()
                        path=folder/f'{vid}.pkl'
                        with path.open('xb') as f: pickle.dump(dict(name=vid,feature=features),f,protocol=4)
                        manifest[vid]=dict(feature_sha256=sha(path),metadata_sha256=sha(devroot/'metadata'/f'{vid}.json'))
                        if (i+1)%25==0:
                            atomic_json(output/'worker_progress.json',dict(phase='dev_features',version=version,completed=i+1,total=len(req['ids'])))
                    atomic_json(folder.parent/'manifest.json',manifest)
                    reply(dict(status='evaluation_features_ready',version=version,folder=str(folder.parent),
                               manifest_sha256=sha(folder.parent/'manifest.json'),seconds=time.time()-start))
                elif cmd=='finish':
                    state.require(version,'idle')
                    assert version==max_steps
                    if pilot:
                        reply(dict(status='finished',version=version,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
                        return
                    frames,starts=cache[0]
                    updated=forward(frames.to(device),starts).cpu()
                    delta=float((updated-outputs[0,:,:len(starts),0].T).abs().max())
                    assert delta>0 and torch.isfinite(updated).all()
                    path=output/'tail_optimizer_step002.pt'
                    torch.save(dict(tail={n:p.detach().cpu() for n,p in params.items()},optimizer=optimizer.state_dict(),
                                    completed_steps=2,resumable=False,reason='SEDS head checkpoint intentionally not retained'),path)
                    result=dict(status='finished',version=version,postupdate_feature_maxabs=delta,
                                tail_checkpoint_sha256=sha(path),peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    atomic_json(output/'worker_result.json',result)
                    reply(result)
                    return
                else: raise ValueError('Unknown operation')
            except Exception:
                traceback.print_exc()
                reply(dict(status='failed',error=traceback.format_exc()))
                raise


if __name__=='__main__': main()
