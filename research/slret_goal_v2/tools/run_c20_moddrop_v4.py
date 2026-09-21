"""C20: one bounded full-TRAIN annealed dominant-stream dropout pilot."""
import json
import sys
from pathlib import Path

import run_c16_control_v4 as runner


RUN = 'seds-rgb-moddrop-annealed-001'
JOB = 'v4-c20-rgb-moddrop-annealed-001'
REFERENCE = 'seds-gcn-horizon3-001'


def validate(result):
    base = runner.BASE
    if result.get('status') != 'completed' or result.get('steps') != 666:
        raise ValueError('C20 did not complete all registered updates')
    if not (base/RUN/'last.pt').is_file():
        raise ValueError('Missing C20 final checkpoint')
    anchor = json.loads((base/REFERENCE/'run.json').read_text())
    if result['config'].get('rgb_moddrop_start') != .2:
        raise ValueError('Wrong C20 intervention strength')
    for key, value in anchor['config'].items():
        if key not in ('run_id','artifact_cap_gib') and result['config'].get(key) != value:
            raise ValueError('Historical R1 recipe mismatch: '+key)
    for key in ('batch_order_sha256','checkpoint_sha256_inherited','inherited_asset_digests'):
        if result[key] != anchor[key]:
            raise ValueError('Base/data/order mismatch: '+key)
    if {g['adaptation_group']:g['lr'] for g in result['actual_optimizer_groups']} != {
            'encoder':1e-6,'fusion':1e-5}:
        raise ValueError('Optimizer group mismatch')
    for key in ('rgb_moddrop_eval_identity_passed','rgb_moddrop_training_passed',
                'rgb_moddrop_zero_terminal_passed','masked_update_checks_passed'):
        if not result.get(key):
            raise ValueError('Missing C20 gate: '+key)
    if result.get('rgb_moddrop_parameters') != 0 or result.get('stopped_early'):
        raise ValueError('C20 parameter/inference or early-stop contract failed')
    if set(result['evaluations']) != {'0','111','222','444','666'}:
        raise ValueError('Registered evaluation schedule incomplete')
    rows = [json.loads(x) for x in (base/RUN/'train_steps.jsonl').read_text().splitlines()]
    if len(rows) != 666 or rows[0]['rgb_drop_probability'] != .2:
        raise ValueError('C20 training log incomplete')
    if rows[-1]['rgb_drop_probability'] != 0 or rows[-1]['rgb_dropped_samples'] != 0:
        raise ValueError('C20 did not finish on complete inputs')
    if rows[-1]['rgb_dropped_total'] <= 0:
        raise ValueError('C20 intervention never activated')
    best = Path(result['selection']['checkpoint'])
    if not best.is_file():
        raise ValueError('Selected checkpoint absent')
    if result['selection'].get('checkpoint_sha256') and runner.sha(best) != result['selection']['checkpoint_sha256']:
        raise ValueError('Selected checksum mismatch')
    return result['selection']['mean_R1'] - anchor['selection']['mean_R1']


def configure():
    runner.RUN = RUN
    runner.JOB = JOB
    runner.OUT = runner.ROOT/'artifacts/slret_goal/jobs'/JOB
    runner.MODE = 'rgb_moddrop'
    runner.REFERENCE = REFERENCE
    runner.TOTAL_UNITS = 666
    runner.HARD_SECONDS = 2300
    runner.OUTER_SECONDS = 2350
    runner.ESTIMATED_SECONDS = 1850
    runner.GPU_QUERY_TIMEOUT_SECONDS = 20
    runner.ARTIFACT_CAP = 42
    runner.EXTRA_SOURCES = [runner.ROOT/'methods/seds_adaptation/dominant_stream_dropout.py']
    python = '/home/haipd/miniconda3/envs/seds/bin/python'
    runner.COMMAND_OVERRIDE = [python,'-u','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=1',str(runner.ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
        '--run-id',RUN,'--masked-pose','control','--policy','fusion','--aux-weight','1',
        '--batch-size','32','--seed','42','--epochs','3','--lr','1e-5','--sign-lr','1e-6',
        '--gcn-long-horizon','--rgb-moddrop-start','.2','--early-stop-drop-pp','2',
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
