"""One technical repair retry of C19-R1; scientific recipe unchanged."""
import sys
from pathlib import Path

import run_c19_pose_to_rgb_v4 as pilot


def configure():
    runner = pilot.configure(
        run='seds-global-exchange-pose-to-rgb-002',
        job='v4-c19-r1-pose-to-rgb-002')
    runner.EXTRA_SOURCES = [Path(pilot.__file__).resolve(),
                            Path(pilot.c19.__file__).resolve()]
    return runner


if __name__ == '__main__':
    runner = configure()
    if sys.argv[1:] == ['--launch']:
        runner.launch(__file__)
    elif sys.argv[1:]:
        raise SystemExit('Only --launch is accepted')
    else:
        runner.main()
