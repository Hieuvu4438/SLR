"""C22 bounded pilot: shared normalized phase modulation on clean GCN."""
import json
import sys
from pathlib import Path

import run_c16_control_v4 as runner


RUN='seds-phase-modulation-001'
JOB='v4-c22-phase-modulation-001'
REFERENCE='seds-masked-control-002'


def validate(result):
    base=runner.BASE
    if result.get('status')!='completed' or result.get('steps')!=222:
        raise ValueError('C22 did not complete all registered updates')
    if result.get('stopped_early') or set(result['evaluations'])!={'0','111','222'}:
        raise ValueError('C22 evaluation/stopping contract failed')
    anchor=json.loads((base/REFERENCE/'run.json').read_text())
    if not result['config'].get('temporal_modulation'):
        raise ValueError('C22 intervention absent')
    for key,value in anchor['config'].items():
        if key not in ('run_id','artifact_cap_gib','allocator_cap_gib') and result['config'].get(key)!=value:
            raise ValueError('Matched clean-GCN recipe mismatch: '+key)
    for key in ('batch_order_sha256','checkpoint_sha256_inherited','inherited_asset_digests'):
        if result[key]!=anchor[key]:
            raise ValueError('Base/data/order mismatch: '+key)
    rates={g['adaptation_group']:g['lr'] for g in result['actual_optimizer_groups']}
    if rates!={'encoder':1e-6,'fusion':1e-5,'temporal_modulation':1e-4}:
        raise ValueError('Optimizer group mismatch')
    for key in ('temporal_modulation_zero_identity_passed',
                'temporal_modulation_stream_factors_updated',
                'temporal_modulation_shared_phase_updated','masked_update_checks_passed'):
        if not result.get(key):raise ValueError('Missing C22 gate: '+key)
    selected=Path(result['selection']['checkpoint'])
    if not selected.is_file():raise ValueError('Selected checkpoint absent')
    if result['selection'].get('checkpoint_sha256') and runner.sha(selected)!=result['selection']['checkpoint_sha256']:
        raise ValueError('Selected checksum mismatch')
    return result['selection']['mean_R1']-anchor['selection']['mean_R1']


def configure():
    runner.RUN=RUN;runner.JOB=JOB
    runner.OUT=runner.ROOT/'artifacts/slret_goal/jobs'/JOB
    runner.MODE='phase_modulation';runner.REFERENCE=REFERENCE
    runner.TOTAL_UNITS=222;runner.HARD_SECONDS=1150;runner.OUTER_SECONDS=1200
    runner.ESTIMATED_SECONDS=650;runner.GPU_QUERY_TIMEOUT_SECONDS=20
    runner.ARTIFACT_CAP=42
    runner.EXTRA_SOURCES=[runner.ROOT/'methods/seds_adaptation/phase_modulation.py']
    python='/home/haipd/miniconda3/envs/seds/bin/python'
    runner.COMMAND_OVERRIDE=[python,'-u','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=1',str(runner.ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
        '--run-id',RUN,'--masked-pose','control','--temporal-modulation',
        '--policy','fusion','--aux-weight','1','--batch-size','32','--seed','42',
        '--lr','1e-5','--sign-lr','1e-6','--epochs','1','--early-stop-drop-pp','2',
        '--allocator-cap-gib','16','--artifact-cap-gib','42']
    runner.validate=validate
    return runner


if __name__=='__main__':
    configure()
    if sys.argv[1:]==['--launch']:runner.launch(__file__)
    elif sys.argv[1:]:raise SystemExit('Only --launch is accepted')
    else:runner.main()
