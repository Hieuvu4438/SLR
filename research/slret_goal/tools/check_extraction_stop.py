"""Audit committed generated assets after a terminal extraction stop; no TEST."""
import argparse
import json
import pickle
import shutil
import subprocess
import time

import cv2

from inventory import ROOT, sha
from extraction_resume import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.time()
    out = ROOT/'artifacts/slret_goal'/args.run_id
    out.mkdir(exist_ok=False)
    source = ROOT/'artifacts/slret_goal/seds-adapted-train-001'
    run = json.loads((source/'run.json').read_text())
    assert run['status'] == 'failed', 'Only inspect terminal failure'
    process = subprocess.run(['ps','-p',str(run['pid']),'-o','args='],capture_output=True,text=True)
    assert process.returncode != 0, 'Recorded PID still exists; inspect identity first'
    with (source/'labels/train.pkl').open('rb') as f:
        ids = list(pickle.load(f))
    verified, remaining, errors, frames = [], [], [], 0
    committed_bytes = 0
    for vid in ids:
        metadata_path = source/'metadata'/(vid+'.json')
        if not metadata_path.exists():
            remaining.append(vid)
            path = '/home/dongvk/datasets/phoenix14T/videos_phoenix/videos/train/'+vid+'.mp4'
            video = cv2.VideoCapture(path)
            if not video.isOpened():
                errors.append(dict(id=vid,error='missing raw video'))
            frames += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
            continue
        meta = json.loads(metadata_path.read_text())
        assert meta['input_contract_sha256'] == run['input_contract_sha256']
        assert meta['native_loader_smoke_pass']
        for folder, key in [('pose','pose_sha256'),('rgb/train','rgb_sha256')]:
            p = source/folder/(vid+'.pkl')
            if sha(p) != meta[key]:
                errors.append(dict(id=vid,error='hash mismatch',path=str(p)))
            committed_bytes += p.stat().st_size
        verified.append(vid)
    assert len(verified) == run['completed']
    free = shutil.disk_usage(ROOT).free
    report = dict(run_id=args.run_id,status='completed',kind='terminal_extraction_asset_audit',
                  test_loaded=False,exit_status=0,source_run_sha256=sha(source/'run.json'),
                  script_sha256=sha(__file__),verified_completed=len(verified),remaining_ids=remaining,
                  remaining_count=len(remaining),remaining_header_frames=frames,errors=errors,
                  committed_feature_bytes=committed_bytes,disk_free_bytes=free,
                  remaining_original_timeout_seconds=6300-run['total_wall_seconds'],
                  estimated_remaining_feature_bytes=committed_bytes/len(verified)*len(remaining),
                  remaining_frame_scaled_seconds=frames*390.5559079647064/55575,
                  wall_seconds=time.time()-started,
                  decision='No restart below 15GiB reserve; preserve completed files. Not a scientific failure.')
    atomic_json(out/'run.json',report)
    with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
        f.write(json.dumps(report)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='remaining_ids'}))


if __name__ == '__main__':
    main()
