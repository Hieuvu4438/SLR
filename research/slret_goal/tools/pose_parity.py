"""Bounded TRAIN-only reconstruction of released SEDS RTMPose keypoints."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
import urllib.request

import numpy as np
import torch

from inventory import ROOT, sha

MODELS = {
    '384': ('rtmpose-l_8xb32-270e_coco-wholebody-384x288.py',
            'rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth'),
    '256': ('rtmpose-l_8xb64-270e_coco-wholebody-256x192.py',
            'rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    args = parser.parse_args()
    out = ROOT / 'artifacts/slret_goal' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    report = dict(run_id=args.run_id, status='running', start_unix=start,
                  pid=os.getpid(), command=sys.argv, split='train_diagnostic',
                  test_loaded=False, selection='no retrieval selection',
                  protocol_sha256=sha(ROOT / 'research/slret_goal/POSE_PARITY_PROTOCOL.md'),
                  script_sha256=sha(__file__),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary'])).hexdigest(),
                  hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], text=True),
                  environment={k: importlib.metadata.version(k) for k in ['torch', 'numpy', 'mmpose', 'mmcv-lite', 'mmengine']})

    def record():
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        with (ROOT / 'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report | {'artifact_dir': str(out)}) + '\n')

    record()
    try:
        from mmengine.config import Config
        from mmpose.apis import init_model, inference_topdown
        torch.set_num_threads(4)
        torch.manual_seed(0)
        config_name, weight_name = MODELS[args.model]
        config = Path('/home/haipd/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody') / config_name
        weight = ROOT / 'artifacts/slret_goal/public_models' / weight_name
        weight.parent.mkdir(exist_ok=True)
        url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/' + weight_name
        if not weight.exists():
            partial = weight.with_suffix('.download')
            with urllib.request.urlopen(url, timeout=60) as r, partial.open('xb') as f:
                while chunk := r.read(1024 * 1024):
                    f.write(chunk)
            partial.rename(weight)
        print(json.dumps({'stage': 'downloaded', 'weight': str(weight), 'bytes': weight.stat().st_size}), flush=True)
        cfg = Config.fromfile(str(config))
        cfg.model.backbone.init_cfg = None
        cfg.model.test_cfg.flip_test = False
        (out / 'resolved_config.py').write_text(cfg.pretty_text)
        report.update(weight_url=url, weight_sha256=sha(weight), config_sha256=sha(config),
                      resolved_config_sha256=sha(out / 'resolved_config.py'), flip_test=False,
                      bbox='full image; native affine padding', seed=0)
        # This specific public OpenMMLab checkpoint contains NumPy metadata;
        # MMEngine's implicit torch.load is incompatible with PyTorch >=2.6.
        checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
        model = init_model(cfg, checkpoint=None, device='cuda:0')
        state = checkpoint['state_dict']
        model.load_state_dict(state, strict=True)
        if 'dataset_meta' in checkpoint.get('meta', {}):
            model.dataset_meta = checkpoint['meta']['dataset_meta']
        report['checkpoint_tensor_mismatches'] = [k for k, v in model.state_dict().items()
                                                  if not torch.equal(v.cpu(), state[k])]
        assert not report['checkpoint_tensor_mismatches']
        train_path = ROOT / 'third_party/SEDS/data_ph/train.pkl'
        train = pickle.load(train_path.open('rb'))
        report['train_manifest_sha256'] = sha(train_path)
        frame_root = Path('/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train')
        rows, predicted, reference = [], [], []
        torch.cuda.reset_peak_memory_stats()
        infer_start = time.time()
        for vid in list(train)[:8]:
            native_path = ROOT / 'third_party/SEDS/PHOENIX-2014-T/RTM_Keypoints' / (vid + '.pkl')
            native = pickle.load(native_path.open('rb'))
            for idx in sorted({0, len(native['img_list']) // 2, len(native['img_list']) - 1}):
                frame = frame_root / vid / native['img_list'][idx]
                pred = inference_topdown(model, str(frame))[0].pred_instances
                arr = np.concatenate([pred.keypoints[0], pred.keypoint_scores[0, :, None]], axis=-1)
                ref = native['keypoints'][idx]
                predicted.append(arr)
                reference.append(ref)
                rows.append(dict(id=vid, frame_index=idx, frame=str(frame), frame_sha256=sha(frame),
                                 native_sha256=sha(native_path), xy_max=float(np.abs(arr[:, :2] - ref[:, :2]).max()),
                                 score_max=float(np.abs(arr[:, 2] - ref[:, 2]).max())))
            print(json.dumps({'stage': 'inference', 'id': vid, 'frames_done': len(rows)}), flush=True)
        torch.cuda.synchronize()
        report['inference_seconds'] = time.time() - infer_start
        pred, ref = np.stack(predicted), np.stack(reference)
        delta = np.abs(pred - ref)
        # Seven upper-body points and both hands as used by SEDS.
        selected = [0, 5, 6, 7, 8, 9, 10] + list(range(91, 133))
        metrics = dict(frames=len(rows), xy_max=float(delta[..., :2].max()),
                       xy_mean=float(delta[..., :2].mean()), score_max=float(delta[..., 2].max()),
                       score_mean=float(delta[..., 2].mean()),
                       body_hand_xy_mean=float(delta[:, selected, :2].mean()),
                       threshold_disagreements={str(t): int(((pred[..., 2] >= t) != (ref[..., 2] >= t)).sum()) for t in [.1, .4]})
        metrics['exact_gate_pass'] = metrics['xy_max'] <= .01 and metrics['score_max'] <= 1e-4
        np.savez_compressed(out / 'pose_comparison.npz', predicted=pred, reference=ref)
        (out / 'frames.json').write_text(json.dumps(rows, indent=2) + '\n')
        report.update(status='completed', exit_status=0, metrics=metrics,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      decision='validate additional TRAIN if gate passes; otherwise not release-equivalent')
        print(json.dumps(metrics), flush=True)
    except Exception:
        report.update(status='failed', exit_status=1, error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time() - start
        record()


if __name__ == '__main__':
    main()
