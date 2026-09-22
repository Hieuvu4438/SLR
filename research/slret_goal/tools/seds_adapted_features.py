"""Explicit adapted SEDS PH features; never replaces native release assets."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
import shutil

import cv2
import numpy as np
import torch

from inventory import ROOT, sha
from pose_parity import MODELS
from extraction_resume import acquire_lock, atomic_json, digest, preserve_orphans, verify_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--split', choices=['train', 'dev'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-videos', type=int, default=0,
                        help='Planned clean pause after this many new videos;0 means all')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'artifacts/slret_goal',
                        help='Parent directory for a new or resumed adapted-feature run')
    parser.add_argument('--pose-checkpoint', type=Path,
                        help='Explicit RTMPose-L checkpoint; defaults to the historical location')
    args = parser.parse_args()
    out = args.output_root.resolve() / args.run_id
    if args.resume and not (out / 'run.json').is_file():
        raise ValueError('Resume requires an existing run report')
    out.mkdir(parents=True, exist_ok=args.resume)
    lease = acquire_lock(out)
    previous = json.loads((out / 'run.json').read_text()) if args.resume else None
    if previous and (previous['split'] != args.split or previous.get('limit') != args.limit):
        raise ValueError('Resume split/limit differs from original run')
    if previous and previous['script_sha256'] != sha(__file__):
        raise ValueError('Resume source differs; explicit migration is required')
    if shutil.disk_usage(out).free < 18 * 1024**3:
        raise RuntimeError('Need >=18GiB free before extraction (15GiB reserve +3GiB plan)')
    for folder in ['pose', 'rgb', 'metadata', 'labels']:
        (out / folder).mkdir(exist_ok=args.resume)
    (out / 'rgb' / args.split).mkdir(exist_ok=args.resume)
    start = time.time()
    report = dict(run_id=args.run_id, status='running', split=args.split,
                  kind='adapted_preprocessing_not_release_reproduction', test_loaded=False,
                  command=sys.argv, start_unix=start, pid=os.getpid(), completed=0,
                  limit=args.limit, resumed=0, new_videos=0, attempt=1 if previous is None else previous['attempt']+1,
                  script_sha256=sha(__file__), protocol_sha256=sha(ROOT / 'research/slret_goal/ADAPTED_SEDS_PROTOCOL.md'),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary'])).hexdigest())

    def record():
        report['wall_seconds'] = time.time() - start
        report['total_wall_seconds'] = report['wall_seconds'] + (previous.get('total_wall_seconds', 0) if previous else 0)
        atomic_json(out / 'run.json', report)

    def ledger():
        with (ROOT / 'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report | {'artifact_dir': str(out)}) + '\n')

    if previous:
        atomic_json(out / f'attempt-{previous["attempt"]:03d}.json', previous)
    record()
    ledger()
    try:
        from mmengine.config import Config
        from mmpose.apis import init_model, inference_topdown
        sys.path.insert(0, str(ROOT / 'shared'))
        from slr_common.features.i3d import ExtractionRecipe, preprocess_rgb_frame, load_i3d, infer_video_features
        torch.set_num_threads(4)
        torch.manual_seed(0)
        cv2.setNumThreads(0)
        config_name, weight_name = MODELS['384']
        pose_ckpt = (args.pose_checkpoint.resolve() if args.pose_checkpoint else
                     ROOT / 'artifacts/slret_goal/public_models' / weight_name)
        cfg_path = Path('/home/haipd/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody') / config_name
        cfg = Config.fromfile(str(cfg_path))
        cfg.model.backbone.init_cfg = None
        cfg.model.test_cfg.flip_test = False
        pose_model = init_model(cfg, checkpoint=None, device='cuda:0')
        ckpt = torch.load(pose_ckpt, map_location='cpu', weights_only=False)
        pose_model.load_state_dict(ckpt['state_dict'], strict=True)
        pose_model.dataset_meta = ckpt['meta']['dataset_meta']
        rgb_ckpt = ROOT / 'artifacts/pretrained/bsl5k.pth.tar'
        rgb_source = ROOT / 'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py'
        device = torch.device('cuda:0')
        rgb_model = load_i3d(rgb_ckpt, rgb_source, device)
        recipe = ExtractionRecipe()
        source = ROOT / 'third_party/SEDS/dataloaders/dataloader_ph_retrieval_pose.py'
        spec = importlib.util.spec_from_file_location('native_seds_loader', source)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        loader = object.__new__(mod.ph_DataLoader_pose)
        loader.interval, loader.max_length_frames = 2, 300
        loader.threshold, loader.frames_threshold = .4, .1
        loader.slide_windows, loader.windows_stride, loader.feature_len = 16, 1, 64
        loader.crop_img_size = np.array([[256, 256]], dtype=np.float32)
        label_path = ROOT / 'third_party/SEDS/data_ph' / (args.split + '.pkl')
        labels = pickle.load(label_path.open('rb'))
        ids = list(labels) if args.split == 'train' else [json.loads(x)['video_id'] for x in
            (ROOT / 'artifacts/manifests/ph_dev.jsonl').read_text().splitlines()]
        if args.limit:
            ids = ids[:args.limit]
        labels = {vid: labels[vid] for vid in ids}
        label_output = out / 'labels' / (args.split + '.pkl')
        label_bytes = pickle.dumps(labels, protocol=4)
        if label_output.exists():
            if sha(label_output) != hashlib.sha256(label_bytes).hexdigest():
                raise ValueError('Resume label manifest changed')
        else:
            with label_output.open('xb') as f:
                f.write(label_bytes)
        report.update(videos=len(ids), seed=0, pose_checkpoint_sha256=sha(pose_ckpt),
                      pose_config_sha256=sha(cfg_path), rgb_checkpoint_sha256=sha(rgb_ckpt),
                      rgb_source_sha256=sha(rgb_source), rgb_recipe_sha256=recipe.digest,
                      native_loader_sha256=sha(source), labels_sha256=sha(label_path),
                      hardware=torch.cuda.get_device_name(), torch=torch.__version__,
                      numpy=np.__version__)
        contract = {k: report[k] for k in ['script_sha256', 'split', 'seed', 'pose_checkpoint_sha256',
                    'pose_config_sha256', 'rgb_checkpoint_sha256', 'rgb_source_sha256',
                    'rgb_recipe_sha256', 'native_loader_sha256', 'labels_sha256', 'torch', 'numpy']}
        contract.update(ids_sha256=digest(ids), batch_size=args.batch_size,
                        resume_helper_sha256=sha(Path(__file__).with_name('extraction_resume.py')),
                        shared_i3d_sha256=sha(ROOT / 'shared/slr_common/features/i3d.py'))
        report['input_contract_sha256'] = digest(contract)
        if previous and previous.get('input_contract_sha256') != report['input_contract_sha256']:
            raise ValueError('Resume input/recipe/source contract mismatch')
        atomic_json(out / 'input_contract.json', contract)
        torch.cuda.reset_peak_memory_stats()
        for vid in ids:
            video_start = time.time()
            raw_path = Path('/home/dongvk/datasets/phoenix14T/videos_phoenix/videos') / args.split / (vid + '.mp4')
            pose_path, rgb_path = out / 'pose' / (vid + '.pkl'), out / 'rgb' / args.split / (vid + '.pkl')
            meta_path = out / 'metadata' / (vid + '.json')
            if args.resume and verify_item(meta_path, pose_path, rgb_path, raw_path, report['input_contract_sha256']):
                report['completed'] += 1
                report['resumed'] += 1
                record()
                continue
            if args.resume:
                preserve_orphans(out, vid, [pose_path, rgb_path])
            if shutil.disk_usage(out).free < 15 * 1024**3:
                raise RuntimeError('15GiB disk reserve reached')
            cap = cv2.VideoCapture(str(raw_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames, points = [], []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                assert frame.shape == (260, 210, 3), frame.shape
                pred = inference_topdown(pose_model, frame)[0].pred_instances
                points.append(np.concatenate([pred.keypoints[0], pred.keypoint_scores[0, :, None]], axis=-1))
                frames.append(frame)
            cap.release()
            if not frames:
                raise ValueError(f'No decoded frames: {vid}')
            names = [f'images{i+1:04d}.png' for i in range(len(frames))]
            pose = dict(keypoints=np.stack(points), img_list=names)
            kept = loader.GetTotalFrameList(pose, np.array([210, 260], dtype=np.float32))
            if not kept:
                raise ValueError(f'No native hand support: {vid}')
            frame_indices = [names.index(n) for n in kept]
            starts = loader._get_pose_clips(torch.zeros(len(kept), 1, 2))['clips_start'].tolist()
            starts = [x for x in starts if x >= 0]
            tensor = torch.from_numpy(np.stack([preprocess_rgb_frame(frames[i][:, :, ::-1]) for i in frame_indices]))
            features, batch = infer_video_features(rgb_model, tensor, starts, recipe, device, args.batch_size)
            assert features.shape == (len(starts), 1024) and np.isfinite(features).all()
            for path, value in [(pose_path, pose), (rgb_path, dict(name=vid, feature=features))]:
                with path.open('xb') as f:
                    pickle.dump(value, f, protocol=4)
            # Exercise the actual native hand/body and RGB loading on outputs.
            loader.video_dict = {0: (vid, str(pose_path))}
            loader.video_RGB_dict = {0: (vid, str(rgb_path))}
            loaded_pose, _ = loader._get_pose(0)
            loaded_rgb, rgb_mask = loader._get_rawvideo(0)
            assert torch.isfinite(loaded_rgb).all()
            assert int(rgb_mask.sum()) == int(loaded_pose['right']['pose_mask'].sum())
            for part in ['right', 'left', 'body']:
                assert torch.isfinite(loaded_pose[part]['pose_sample']).all()
            metadata = dict(id=vid, raw_path=str(raw_path), raw_sha256=sha(raw_path), fps=fps,
                input_contract_sha256=report['input_contract_sha256'],
                decoded_frames=len(frames), retained_frame_indices=frame_indices, clip_starts=starts,
                original_frame_indices_per_window=[[frame_indices[min(s+j, len(frame_indices)-1)] for j in range(16)] for s in starts],
                feature_shape=list(features.shape), effective_batch=batch,
                pose_sha256=sha(pose_path), rgb_sha256=sha(rgb_path), native_loader_smoke_pass=True,
                wall_seconds=time.time()-video_start)
            atomic_json(meta_path, metadata)
            report['completed'] += 1
            report['new_videos'] += 1
            record()
            print(json.dumps({'id':vid, 'completed':report['completed'], 'total':len(ids),
                              'seconds':metadata['wall_seconds'], 'frames':len(frames), 'windows':len(starts)}), flush=True)
            if args.max_new_videos and report['new_videos'] >= args.max_new_videos:
                break
        report.update(status='completed' if report['completed'] == len(ids) else 'paused',
                      exit_status=0, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception:
        report.update(status='failed', exit_status=1, error=traceback.format_exc())
        raise
    finally:
        record()
        ledger()
        lease.close()


if __name__ == '__main__':
    main()
