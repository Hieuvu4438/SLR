"""Check user-supplied raw-data route against canonical PH DEV IDs."""
import csv
import json
from pathlib import Path
import pickle
import time

import numpy as np
from inventory import ROOT, sha


def main():
    start = time.time()
    out = ROOT / 'artifacts/slret_goal/dataset-bridge-001.json'
    if out.exists():
        raise FileExistsError(out)
    rows = [json.loads(x) for x in (ROOT / 'artifacts/manifests/ph_dev.jsonl').read_text().splitlines()]
    base = Path('/home/dongvk/datasets/phoenix14T')
    release = base / 'PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
    csv_path = release / 'annotations/manual/PHOENIX-2014-T.dev.corpus.csv'
    official = list(csv.DictReader(csv_path.open(), delimiter='|'))
    ids = {r['video_id'] for r in rows}
    assert ids == {r['name'] for r in official}
    report = {'run_id': 'dataset-bridge-001', 'status': 'completed', 'start_unix': start,
        'input_guide': str(ROOT / 'docs/proposal1/datasets.md'), 'input_guide_sha256': sha(ROOT / 'docs/proposal1/datasets.md'),
        'official_csv_sha256': sha(csv_path), 'canonical_dev_count': len(rows), 'canonical_id_set_equal': True,
        'raw_video_present': 0, 'local_pose_present': 0, 'test_loaded': False, 'manifest': []}
    for row in rows:
        video = base / 'videos_phoenix/videos/dev' / (row['video_id'] + '.mp4')
        pose = release / 'output/keypoint' / (row['video_id'] + '.pkl')
        report['raw_video_present'] += video.is_file()
        report['local_pose_present'] += pose.is_file()
        report['manifest'].append({'id': row['video_id'], 'raw_video': str(video), 'pose': str(pose)})
    comparisons = []
    # Fixed first eight release-TRAIN IDs. No dev-feature tuning or calibration.
    train = pickle.load((ROOT / 'third_party/SEDS/data_ph/train.pkl').open('rb'))
    for vid in list(train)[:8]:
        native_path = ROOT / 'third_party/SEDS/PHOENIX-2014-T/RTM_Keypoints' / (vid + '.pkl')
        local_path = release / 'output/keypoint' / (vid + '.pkl')
        native, local = pickle.load(native_path.open('rb')), pickle.load(local_path.open('rb'))
        kp, score = np.asarray(local['keypoints']), np.asarray(local['scores'])
        item = {'id': vid, 'native_sha256': sha(native_path), 'local_sha256': sha(local_path),
            'native_shape': list(native['keypoints'].shape), 'local_xy_shape': list(kp.shape),
            'local_scores_min_max': [float(score.min()), float(score.max())],
            'native_scores_min_max': [float(native['keypoints'][..., 2].min()), float(native['keypoints'][..., 2].max())]}
        if kp.shape == (len(native['keypoints']), 1, 133, 2):
            item['xy_max_abs_delta_after_width_height_scaling'] = float(np.max(np.abs(kp[:, 0] * [210, 260] - native['keypoints'][..., :2])))
        comparisons.append(item)
    report['train_pose_comparisons'] = comparisons
    report['decision'] = 'Raw PH DEV exists; pose is schema/extractor-different from SEDS release. No unverified conversion or native-parity claim. Establish extractor/coordinate/confidence/frame provenance before adapted dev extraction.'
    report['wall_seconds'] = time.time() - start
    report['exit_status'] = 0
    out.write_text(json.dumps(report, indent=2) + '\n')
    with (ROOT / 'research/slret_goal/experiments.jsonl').open('a') as f:
        f.write(json.dumps({k: v for k, v in report.items() if k not in ['manifest', 'train_pose_comparisons']} | {'artifact': str(out), 'artifact_sha256': sha(out), 'script_sha256': sha(__file__)}) + '\n')
    print(json.dumps({k: report[k] for k in ['canonical_dev_count', 'canonical_id_set_equal', 'raw_video_present', 'local_pose_present', 'decision']}))


if __name__ == '__main__':
    main()
