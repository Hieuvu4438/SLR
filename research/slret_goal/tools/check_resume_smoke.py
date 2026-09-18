"""Verify pause/resume on actual generated TRAIN pose and RGB tensors."""
import json
import pickle
from pathlib import Path
import time

import numpy as np

from inventory import ROOT, sha


def main():
    start = time.time()
    old = ROOT / 'artifacts/slret_goal/seds-adapted-train-smoke-001'
    new = ROOT / 'artifacts/slret_goal/seds-adapted-resume-smoke-001'
    out = ROOT / 'artifacts/slret_goal/resume-smoke-check-001.json'
    if out.exists():
        raise FileExistsError(out)
    paused = json.loads((new / 'attempt-001.json').read_text())
    resumed = json.loads((new / 'run.json').read_text())
    assert paused['status'] == 'paused' and paused['completed'] == 1
    assert resumed['status'] == 'completed' and resumed['completed'] == 2
    assert resumed['resumed'] == 1 and resumed['new_videos'] == 1
    rows = []
    for path in sorted((new / 'metadata').glob('*.json')):
        vid = path.stem
        a, b = json.loads((old / 'metadata' / path.name).read_text()), json.loads(path.read_text())
        for key in ['retained_frame_indices', 'clip_starts', 'original_frame_indices_per_window', 'feature_shape', 'raw_sha256']:
            assert a[key] == b[key], (vid, key)
        for kind, key in [('pose', 'keypoints'), ('rgb/train', 'feature')]:
            pa, pb = old / kind / (vid + '.pkl'), new / kind / (vid + '.pkl')
            xa, xb = pickle.load(pa.open('rb'))[key], pickle.load(pb.open('rb'))[key]
            delta = float(np.max(np.abs(xa - xb)))
            assert delta <= 1e-4, (vid, kind, delta)
            rows.append(dict(id=vid, kind=kind, max_absolute_delta=delta,
                             original_sha256=sha(pa), resumed_sha256=sha(pb)))
    report = dict(run_id='resume-smoke-check-001', status='completed', exit_status=0,
                  test_loaded=False, rows=rows, resumed_videos=1, newly_computed_videos=1,
                  pause_report_sha256=sha(new / 'attempt-001.json'), resume_report_sha256=sha(new / 'run.json'),
                  script_sha256=sha(__file__), wall_seconds=time.time()-start,
                  decision='Resume smoke passed; unchanged numeric recipe and source-frame indices. Full TRAIN admitted within registered timeout.')
    out.write_text(json.dumps(report, indent=2) + '\n')
    with (ROOT / 'research/slret_goal/experiments.jsonl').open('a') as f:
        f.write(json.dumps(report | {'artifact': str(out)}) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
