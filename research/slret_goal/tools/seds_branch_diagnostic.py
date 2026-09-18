"""Preregistered branch-error overlap; oracle is not a deployable method."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from inventory import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    start = time.time()
    root = args.evaluation_root.resolve()
    out = ROOT / 'artifacts/slret_goal' / (args.run_id + '.json')
    if out.exists():
        raise FileExistsError(out)
    run = json.loads((root / 'run.json').read_text())
    assert run['status'] == 'completed' and run['limit'] == 519
    assert run['evaluation_split'] == 'dev_adapted_input_transfer'
    ids = json.loads((root / 'ids.json').read_text())
    metrics = {s: json.loads((root / (s + '_metrics.json')).read_text()) for s in ['fusion', 'pose', 'rgb']}
    means = {s: (m['T2V']['R1'] + m['V2T']['R1']) / 2 for s, m in metrics.items()}
    directions = {}
    for direction in ['T2V', 'V2T']:
        correct = {}
        for stream, result in metrics.items():
            per = result['per_query'][direction]
            assert [x['query_id'] for x in per] == ids
            correct[stream] = np.array([x['rank'] == 0 for x in per])
        fused, pose, rgb = [correct[k] for k in ['fusion', 'pose', 'rgb']]
        failures = int((~fused).sum())
        common = ~pose & ~rgb
        directions[direction] = dict(
            official_R1={s: metrics[s][direction]['R1'] for s in metrics},
            correct_at1_counts={s: int(v.sum()) for s, v in correct.items()},
            tie_stats={s: metrics[s]['tie_stats'] for s in metrics},
            joint_correct_patterns={f'{int(f)}{int(p)}{int(r)}': int(((fused == f) & (pose == p) & (rgb == r)).sum())
                                    for f in [False, True] for p in [False, True] for r in [False, True]},
            pattern_order='fusion,pose,rgb',
            oracle_union_pose_rgb_count=int((pose | rgb).sum()),
            fusion_only_success_count=int((fused & common).sum()),
            fusion_fail_branch_success_count=int((~fused & (pose | rgb)).sum()),
            common_branch_error_fraction_of_fusion_errors=float((common & ~fused).sum() / failures) if failures else None,
            fusion_fail_branch_success_ids=[vid for vid, hit in zip(ids, ~fused & (pose | rgb)) if hit],
        )
    delta = means['fusion'] - max(means['pose'], means['rgb'])
    if delta <= -.5:
        decision = 'Prioritize tracing fusion computation; no causal attribution under adapted inputs and shared branch training.'
    elif delta > 0 and all((d['common_branch_error_fraction_of_fusion_errors'] or 0) > .5 for d in directions.values()):
        decision = 'Prioritize input/representation analysis; most fused errors are common branch errors.'
    else:
        decision = 'Mixed branch evidence; no method admission or scale-up justified by overlap alone.'
    report = dict(run_id=args.run_id, status='completed', exit_status=0,
                  selection_split='historically_exposed_PH_dev', test_loaded=False,
                  input_run_sha256=sha(root / 'run.json'),
                  input_metrics_sha256={s: sha(root / (s + '_metrics.json')) for s in metrics},
                  script_sha256=sha(__file__), protocol_sha256=sha(ROOT / 'research/slret_goal/ADAPTED_SEDS_PROTOCOL.md'),
                  mean_R1=means, fused_minus_stronger_branch_pp=delta, directions=directions,
                  decision=decision, wall_seconds=time.time()-start,
                  oracle_warning='Diagnostic only; no deployable retrieval gain. T2V per-query overlaps use minimum rank for ties; official expanded-tie recall is reported separately.')
    out.write_text(json.dumps(report, indent=2) + '\n')
    with (ROOT / 'research/slret_goal/experiments.jsonl').open('a') as f:
        f.write(json.dumps({k: v for k, v in report.items() if k != 'directions'} | {'artifact': str(out)}) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'directions'}, indent=2))


if __name__ == '__main__':
    main()
