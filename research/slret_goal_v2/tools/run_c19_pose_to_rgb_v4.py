"""C19-R1: one directed pose-to-RGB exchange pilot; no automatic follow-up."""
import sys
from pathlib import Path

import run_c16_control_v4 as runner
import run_c19_global_exchange_v4 as c19


RUN = 'seds-global-exchange-pose-to-rgb-001'
JOB = 'v4-c19-r1-pose-to-rgb-001'


def validate(result):
    delta = c19.validate_mode(result,'pose_to_rgb')
    if result.get('global_exchange_parameters') != 262656:
        raise ValueError('Unexpected total C19 parameter schema')
    if result.get('global_exchange_trainable_parameters') != 131328:
        raise ValueError('Unexpected directed adapter capacity')
    if not any(g.get('adaptation_group') == 'global_exchange' and g.get('parameters') == 131328
               for g in result['actual_optimizer_groups']):
        raise ValueError('Missing directed adapter optimizer group')
    return delta


def configure(run=RUN, job=JOB):
    runner.RUN = run
    runner.JOB = job
    runner.OUT = runner.ROOT/'artifacts/slret_goal/jobs'/job
    runner.MODE = 'global_exchange_pose_to_rgb'
    runner.REFERENCE = c19.REFERENCE
    runner.TOTAL_UNITS = 160
    runner.HARD_SECONDS = 3600
    runner.OUTER_SECONDS = 3650
    runner.ESTIMATED_SECONDS = 3200
    runner.GPU_QUERY_TIMEOUT_SECONDS = 20
    runner.EXTRA_SOURCES = [Path(c19.__file__).resolve()]
    runner.COMMAND_OVERRIDE = [sys.executable,'-u','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=1',str(runner.ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
        '--run-id',run,'--native-subset','--global-exchange','pose_to_rgb',
        '--fusion-objective','native','--batch-size','32','--masked-pose','control',
        '--policy','fusion','--aux-weight','1','--seed','42','--epochs','10',
        '--lr','1e-5','--sign-lr','1e-6','--early-stop-drop-pp','2',
        '--activation-offload','--allocator-cap-gib','16','--artifact-cap-gib','42']
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
