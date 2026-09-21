"""Prepared C21 pilot; launch only when GPU contention/budget permit."""
import json
import sys
from pathlib import Path

import run_c16_control_v4 as runner


RUN = 'seds-geometry-canonical-motion-003'
JOB = 'v4-c21-geometry-canonical-motion-003'
REFERENCE = 'seds-geometry-xyz-001'


def validate(result):
    base = runner.BASE
    if result.get('status') != 'completed' or result.get('steps') != 128:
        raise ValueError('C21 did not complete all registered updates')
    if not (base/RUN/'last.pt').is_file():
        raise ValueError('Missing C21 final checkpoint')
    anchor = json.loads((base/REFERENCE/'run.json').read_text())
    if result['config'].get('geometry_representation') != 'canonical_motion':
        raise ValueError('Wrong C21 representation')
    for key, value in anchor['config'].items():
        if key not in ('run_id','artifact_cap_gib') and result['config'].get(key) != value:
            raise ValueError('Historical raw-XYZ recipe mismatch: '+key)
    for key in ('batch_order_sha256','checkpoint_sha256_inherited','inherited_asset_digests','geometry_data'):
        if result[key] != anchor[key]:
            raise ValueError('Base/data/order mismatch: '+key)
    rates = {g['adaptation_group']:g['lr'] for g in result['actual_optimizer_groups']}
    if rates != {'geometry':1e-4,'fusion':1e-5}:
        raise ValueError('Optimizer group mismatch')
    for key in ('geometry_zero_output_weight_passed','geometry_output_updated',
                'adaptation_roundtrip_passed','frozen_buffers_unchanged'):
        if not result.get(key):
            raise ValueError('Missing C21 gate: '+key)
    if result.get('stopped_early') or set(result['evaluations']) != {'0','64','128'}:
        raise ValueError('C21 evaluation/stop contract failed')
    best = Path(result['selection']['checkpoint'])
    if not best.is_file():
        raise ValueError('Selected checkpoint absent')
    if result['selection'].get('checkpoint_sha256') and runner.sha(best) != result['selection']['checkpoint_sha256']:
        raise ValueError('Selected checksum mismatch')
    return result['selection']['mean_R1']-anchor['selection']['mean_R1']


def configure():
    runner.RUN = RUN
    runner.JOB = JOB
    runner.OUT = runner.ROOT/'artifacts/slret_goal/jobs'/JOB
    runner.MODE = 'canonical_motion_3d'
    runner.REFERENCE = REFERENCE
    runner.TOTAL_UNITS = 128
    runner.HARD_SECONDS = 1000
    runner.OUTER_SECONDS = 1050
    runner.ESTIMATED_SECONDS = 450
    runner.GPU_QUERY_TIMEOUT_SECONDS = 20
    runner.ARTIFACT_CAP = 42
    runner.EXTRA_SOURCES = [runner.ROOT/'methods/seds_adaptation/pose3d_branch.py']
    python = '/home/haipd/miniconda3/envs/seds/bin/python'
    runner.COMMAND_OVERRIDE = [python,'-u','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=1',str(runner.ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
        '--run-id',RUN,'--geometry','xyz','--geometry-representation','canonical_motion',
        '--policy','fusion','--aux-weight','1','--batch-size','32','--seed','42',
        '--lr','1e-5','--sign-lr','1e-4','--epochs','8','--early-stop-drop-pp','2',
        '--allocator-cap-gib','16','--artifact-cap-gib','42']
    runner.validate = validate
    return runner


if __name__ == '__main__':
    configure()
    if sys.argv[1:] == ['--launch']:
        runner.launch(__file__)
    elif sys.argv[1:]:
        raise SystemExit('Only --launch is accepted')
    else:
        runner.main()
