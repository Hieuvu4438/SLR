"""Non-vacuous fixed-microbatch direct-graph versus gradient-cache VJP check."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from inventory import ROOT, sha
from extraction_resume import atomic_json
from rgb_gradient_replay import window_batch, unfreeze_i3d_tail


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started=time.time()
    report=dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,
                script_sha256=sha(__file__),optimizer_updates=0,test_loaded=False,dev_loaded=False)
    atomic_json(out/'run.json',report)
    try:
        sys.path.insert(0,str(ROOT/'shared'))
        from slr_common.features.i3d import ExtractionRecipe,decode_video,load_i3d
        torch.set_num_threads(4)
        device=torch.device('cuda:0')
        source=ROOT/'artifacts/slret_goal/rgb-gradient-feasibility-002/representation_gradients.pt'
        payload=torch.load(source,map_location='cpu',weights_only=True)
        vid=payload['ids'][3]
        meta=json.loads((ROOT/'artifacts/slret_goal/seds-adapted-train-001/metadata'/f'{vid}.json').read_text())
        assert sha(meta['raw_path'])==meta['raw_sha256']
        recipe=ExtractionRecipe()
        frames,_=decode_video(Path(meta['raw_path']),recipe)
        frames=frames[meta['retained_frame_indices']].to(device)
        starts=meta['clip_starts'][:16]
        assert len(starts)==16
        g=payload['gradient'][3,:,:16,0].T.contiguous().to(device)
        original_norm=float(g.norm())
        assert original_norm>0
        g=g/g.norm()
        model=load_i3d(ROOT/'artifacts/pretrained/bsl5k.pth.tar',
                       ROOT/'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py',device)
        params=unfreeze_i3d_tail(model)
        torch.cuda.reset_peak_memory_stats()
        # Retain both microbatch graphs and backpropagate through their concatenation.
        direct=torch.cat([model(window_batch(frames,starts[s:s+8],recipe,device))['embds'].flatten(1)
                          for s in [0,8]])
        direct.backward(g)
        reference={n:p.grad.detach().clone() for n,p in params.items()}
        direct_features=direct.detach()
        del direct
        model.zero_grad(set_to_none=True)
        parity=[]
        for s in [0,8]:
            replay=model(window_batch(frames,starts[s:s+8],recipe,device))['embds'].flatten(1)
            parity.append(float((replay.detach()-direct_features[s:s+8]).abs().max()))
            replay.backward(g[s:s+8])
        assert all(torch.isfinite(p.grad).all() for p in params.values())
        denominator=sum(v.double().square().sum() for v in reference.values())
        numerator=sum((params[n].grad.double()-v.double()).square().sum() for n,v in reference.items())
        relative=float((numerator/denominator).sqrt())
        absolute=max(float((params[n].grad-v).abs().max()) for n,v in reference.items())
        report.update(id=vid,gradient_sha256=sha(source),gradient_original_norm=original_norm,
                      gradient_test_norm=float(g.norm()),relative_L2=relative,maxabs=absolute,
                      feature_deltas=parity,torch=torch.__version__,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        assert max(parity)==0 and relative<=1e-4,('Fixed-microbatch VJP equivalence',report)
        report.update(status='completed',exit_status=0,
                      decision='Fixed-eight-window computational graph matches cache replay; not arbitrary batch-size equivalence.')
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
