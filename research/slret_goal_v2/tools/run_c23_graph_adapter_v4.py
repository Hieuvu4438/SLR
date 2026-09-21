"""C23 bounded pilot: frozen hierarchical ST-GCN bottleneck adapters."""
import json
import sys
from pathlib import Path

import run_c16_control_v4 as runner


RUN='seds-graph-bottleneck-adapter-001'
JOB='v4-c23-graph-bottleneck-adapter-001'
REFERENCE='seds-masked-control-002'


def validate(result):
    base=runner.BASE
    if result.get('status')!='completed' or result.get('steps')!=222:
        raise ValueError('C23 did not complete all registered updates')
    if result.get('stopped_early') or set(result['evaluations'])!={'0','111','222'}:
        raise ValueError('C23 evaluation/stopping contract failed')
    anchor=json.loads((base/REFERENCE/'run.json').read_text())
    if not result['config'].get('graph_bottleneck_adapter'):
        raise ValueError('C23 intervention absent')
    for key,value in anchor['config'].items():
        if key not in ('run_id','artifact_cap_gib','allocator_cap_gib') and result['config'].get(key)!=value:
            raise ValueError('Matched clean-GCN recipe mismatch: '+key)
    for key in ('batch_order_sha256','checkpoint_sha256_inherited','inherited_asset_digests'):
        if result[key]!=anchor[key]:
            raise ValueError('Base/data/order mismatch: '+key)
    rates={group['adaptation_group']:group['lr'] for group in result['actual_optimizer_groups']}
    if rates!={'fusion':1e-5,'graph_adapter':1e-4}:
        raise ValueError('Optimizer group mismatch')
    for key in ('graph_bottleneck_adapter_zero_identity_passed',
                'graph_bottleneck_adapter_outputs_updated',
                'graph_bottleneck_adapters_updated','masked_update_checks_passed'):
        if not result.get(key):
            raise ValueError('Missing C23 gate: '+key)
    selected=Path(result['selection']['checkpoint'])
    if not selected.is_file():
        raise ValueError('Selected checkpoint absent')
    if result['selection'].get('checkpoint_sha256') and runner.sha(selected)!=result['selection']['checkpoint_sha256']:
        raise ValueError('Selected checksum mismatch')
    return result['selection']['mean_R1']-anchor['selection']['mean_R1']


def configure():
    runner.RUN=RUN;runner.JOB=JOB
    runner.OUT=runner.ROOT/'artifacts/slret_goal/jobs'/JOB
    runner.MODE='graph_bottleneck_adapter';runner.REFERENCE=REFERENCE
    runner.TOTAL_UNITS=222;runner.HARD_SECONDS=900;runner.OUTER_SECONDS=950
    runner.ESTIMATED_SECONDS=650;runner.GPU_QUERY_TIMEOUT_SECONDS=20
    runner.ARTIFACT_CAP=42
    runner.EXTRA_SOURCES=[runner.ROOT/'methods/seds_adaptation/graph_bottleneck_adapter.py']
    python='/home/haipd/miniconda3/envs/seds/bin/python'
    runner.COMMAND_OVERRIDE=[python,'-u','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=1',str(runner.ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
        '--run-id',RUN,'--masked-pose','control','--graph-bottleneck-adapter',
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
