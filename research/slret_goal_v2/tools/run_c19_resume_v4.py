"""Recover C19 updates113..160 from pinned last112, no training recipe change."""
import sys
from pathlib import Path
import run_c19_global_exchange_v4 as pilot
from c19_resume import PARENT, LAST_SHA


def configure():
    pilot.RUN = 'seds-global-exchange-cross-resume-001'
    pilot.configure()
    runner = pilot.runner
    original_validate = runner.validate

    def validate(result):
        resume = result.get('resume',{})
        if (resume.get('checkpoint_sha256')!=LAST_SHA or resume.get('next_batch_index')!=112
                or not resume.get('model_optimizer_restored')
                or not resume.get('first_resumed_update_passed')):
            raise ValueError('Missing actual C19 recovery gates')
        return original_validate(result)

    runner.validate = validate
    runner.JOB = 'v4-c19-resume-001'
    runner.OUT = runner.ROOT/'artifacts/slret_goal/jobs'/runner.JOB
    runner.HARD_SECONDS = 1800
    runner.OUTER_SECONDS = 1850
    runner.ESTIMATED_SECONDS = 1200
    runner.COMMAND_OVERRIDE += ['--resume-from',str(runner.BASE/PARENT/'last.pt')]
    runner.EXTRA_SOURCES = [Path(pilot.__file__).resolve(),Path(__file__).with_name('c19_resume.py')]
    return runner


if __name__=='__main__':
    runner = configure()
    if sys.argv[1:]==['--launch']:runner.launch(__file__)
    elif sys.argv[1:]:raise SystemExit('Only --launch is accepted')
    else:runner.main()
