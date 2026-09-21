"""Single infrastructure retry of C18; unchanged training recipe, no retry queue."""
import sys
from pathlib import Path
import run_c18_batch64_v4 as pilot


def configure():
    pilot.RUN = 'seds-native-batch64-offload-002'
    pilot.configure()
    runner = pilot.runner
    runner.JOB = 'v4-c18-batch64-002'
    runner.OUT = runner.ROOT/'artifacts/slret_goal/jobs'/runner.JOB
    # A 5s nvidia-smi timeout killed 001. Still fail closed on timeout/N/A;
    # never substitute zero/stale readings or remove the 19GB memory trip.
    runner.GPU_QUERY_TIMEOUT_SECONDS = 20
    runner.EXTRA_SOURCES = [Path(pilot.__file__).resolve()]
    return runner


if __name__ == '__main__':
    runner = configure()
    if sys.argv[1:] == ['--launch']:
        runner.launch(__file__)
    elif sys.argv[1:]:
        raise SystemExit('Only --launch is accepted')
    else:
        runner.main()
