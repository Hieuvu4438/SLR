"""Bounded evaluation recovery only; original training timeout remains immutable."""
import json
from pathlib import Path
import sys
import run_c16_control_v4 as runner
from recover_c17_final_eval import RUN,SOURCE,LAST_SHA


def validate(result):
    if result.get('status')!='completed' or result.get('new_training_steps')!=0:
        raise ValueError('Evaluation recovery unsuccessful or unexpectedly trained')
    for key in ('source_checkpoint_step_verified','delta_load_exact'):
        if not result.get(key):raise ValueError('Missing recovery gate: '+key)
    if result.get('test_loaded') or result.get('checkpoint_sha256')!=LAST_SHA:
        raise ValueError('Invalid data/checkpoint')
    for stream in ('fusion','pose','rgb'):
        for suffix in ('_video_x_text.npy','_metrics.json'):
            if not (runner.BASE/RUN/'eval_step0160'/(stream+suffix)).is_file():
                raise ValueError('Incomplete finalDEV outputs')
    selected=Path(result['selection']['checkpoint'])
    if not selected.is_file() or runner.sha(selected)!=result['selection']['checkpoint_sha256']:
        raise ValueError('Selected checkpoint mismatch')
    control=json.loads((runner.BASE/runner.REFERENCE/'run.json').read_text())
    return result['selection']['mean_R1']-control['selection']['mean_R1']


def configure():
    runner.RUN=RUN
    runner.JOB='v4-c17-final-eval-001'
    runner.OUT=runner.ROOT/'artifacts/slret_goal/jobs'/runner.JOB
    runner.MODE='evaluation_recovery'
    runner.REFERENCE='seds-signrep-control-offload-001'
    worker=runner.ROOT/'research/slret_goal_v2/tools/recover_c17_final_eval.py'
    runner.COMMAND_OVERRIDE=[sys.executable,'-u','-m','torch.distributed.run',
        '--standalone','--nproc_per_node=1',str(worker)]
    runner.EXTRA_SOURCES=[worker]
    runner.TOTAL_UNITS=1
    runner.HARD_SECONDS=900
    runner.OUTER_SECONDS=950
    runner.ESTIMATED_SECONDS=300
    runner.CHECKPOINT_LABEL=str(SOURCE/'last.pt')
    runner.validate=validate


if __name__=='__main__':
    configure()
    if sys.argv[1:]==['--launch']:runner.launch(__file__)
    elif sys.argv[1:]:raise SystemExit('Only --launch is accepted')
    else:runner.main()
