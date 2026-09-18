"""Compare existing TRAIN RGB caches only at exactly matched frame windows."""
import importlib.util
import json
import pickle
import time

import numpy as np
import torch

from inventory import ROOT, sha


def main():
    start = time.time()
    out = ROOT / 'artifacts/slret_goal/rgb-cache-parity-001.json'
    if out.exists():
        raise FileExistsError(out)
    path = ROOT / 'third_party/SEDS/dataloaders/dataloader_ph_retrieval_pose.py'
    spec = importlib.util.spec_from_file_location('seds_pose_loader', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    loader = object.__new__(mod.ph_DataLoader_pose)
    loader.interval, loader.max_length_frames = 2, 300
    loader.threshold, loader.frames_threshold = .4, .1
    loader.slide_windows, loader.windows_stride, loader.feature_len = 16, 1, 64
    train = pickle.load((ROOT / 'third_party/SEDS/data_ph/train.pkl').open('rb'))
    rows = []
    for vid in list(train)[:8]:
        pose_path = ROOT / 'third_party/SEDS/PHOENIX-2014-T/RTM_Keypoints' / (vid + '.pkl')
        rgb_path = ROOT / 'third_party/SEDS/PHOENIX-2014-T/I3D_features/train' / (vid + '.pkl')
        pose = pickle.load(pose_path.open('rb'))
        native = pickle.load(rgb_path.open('rb'))['feature']
        names = loader.GetTotalFrameList(pose, np.array([210, 260], dtype=np.float32))
        clips = loader._get_pose_clips(torch.zeros(len(names), 1, 2))['clips_start'].tolist()
        clips = [x for x in clips if x >= 0]
        assert len(clips) == len(native)
        raw_indices = [pose['img_list'].index(n) for n in names]
        matched = []
        for i, clip in enumerate(clips):
            sequence = raw_indices[clip:clip + 16]
            if len(sequence) == 16 and sequence == list(range(sequence[0], sequence[0] + 16)):
                matched.append((i, sequence[0]))
        row = dict(id=vid, raw_frames=len(pose['img_list']), retained_frames=len(names),
                   windows=len(clips), matched_windows=len(matched), pose_sha256=sha(pose_path),
                   release_rgb_sha256=sha(rgb_path), streams={})
        for stream in ['ph_domain_agnostic', 'ph_domain_aware']:
            cache_path = ROOT / 'artifacts/sign_features' / stream / 'train' / (vid + '.pkl')
            cache = pickle.load(cache_path.open('rb'))['feature']
            a = np.stack([native[i] for i, _ in matched]) if matched else None
            b = np.stack([cache[j] for _, j in matched]) if matched else None
            row['streams'][stream] = dict(cache_sha256=sha(cache_path),
                max_absolute_delta=float(np.abs(a - b).max()) if matched else None,
                mean_cosine=float(((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))).mean()) if matched else None)
        rows.append(row)
    report = dict(run_id='rgb-cache-parity-001', status='completed', exit_status=0,
                  test_loaded=False, selection_split='train_only_no_selection', rows=rows,
                  source_sha256=sha(path), script_sha256=sha(__file__),
                  protocol_sha256=sha(ROOT / 'research/slret_goal/POSE_PARITY_PROTOCOL.md'),
                  wall_seconds=time.time() - start)
    report['exact_gate_pass'] = all(r['matched_windows'] == r['windows'] and
        r['streams']['ph_domain_agnostic']['max_absolute_delta'] <= 1e-4 for r in rows)
    out.write_text(json.dumps(report, indent=2) + '\n')
    with (ROOT / 'research/slret_goal/experiments.jsonl').open('a') as f:
        f.write(json.dumps({k: v for k, v in report.items() if k != 'rows'} | {'artifact': str(out)}) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
