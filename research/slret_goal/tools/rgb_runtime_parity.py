"""Localize RGB numerical parity by runtime/cuDNN policy; no weight updates."""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from inventory import ROOT, sha
from extraction_resume import atomic_json
from rgb_gradient_replay import window_batch
from seds_runtime import compatible_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    cli = parser.parse_args()
    out = ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,
                  script_sha256=sha(__file__),optimizer_updates=0,test_loaded=False,dev_loaded=False,
                  torch=torch.__version__,cases=[])
    atomic_json(out/'run.json',report)
    try:
        sys.path.insert(0,str(ROOT/'shared'))
        from slr_common.features.i3d import ExtractionRecipe,decode_video,load_i3d,infer_video_features
        import cv2
        import numpy as np
        torch.set_num_threads(4)
        recipe = ExtractionRecipe()
        device = torch.device('cuda:0')
        model = load_i3d(ROOT/'artifacts/pretrained/bsl5k.pth.tar',
                         ROOT/'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py',device)
        root = ROOT/'artifacts/slret_goal/seds-adapted-train-001'
        vid = '26April_2010_Monday_heute-3396'
        meta = json.loads((root/'metadata'/f'{vid}.json').read_text())
        assert sha(meta['raw_path']) == meta['raw_sha256']
        frames,_ = decode_video(Path(meta['raw_path']),recipe)
        report.update(opencv=cv2.__version__,numpy=np.__version__,id=vid,
                      preprocessed_frame_sha256=hashlib.sha256(frames.numpy().tobytes()).hexdigest(),
                      checkpoint_sha256=sha(ROOT/'artifacts/pretrained/bsl5k.pth.tar'))
        frames = frames[meta['retained_frame_indices']].to(device)
        target_path = root/'rgb/train'/f'{vid}.pkl'
        assert sha(target_path) == meta['rgb_sha256']
        target = compatible_load(target_path.open('rb'))['feature']
        starts = meta['clip_starts']
        for deterministic in [False,True]:
            with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=deterministic,allow_tf32=True):
                reference,_ = infer_video_features(model,frames,starts,recipe,device,8)
                with torch.no_grad():
                    replay = torch.cat([model(window_batch(frames,starts[s:s+8],recipe,device))['embds'].flatten(1)
                                        for s in range(0,len(starts),8)]).cpu().numpy()
            report['cases'].append(dict(deterministic=deterministic,
                shared_vs_saved_maxabs=float(np.max(np.abs(reference-target))),
                helper_vs_shared_maxabs=float(np.max(np.abs(replay-reference))),
                helper_vs_saved_maxabs=float(np.max(np.abs(replay-target)))))
            atomic_json(out/'run.json',report)
        report.update(status='completed',exit_status=0,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
