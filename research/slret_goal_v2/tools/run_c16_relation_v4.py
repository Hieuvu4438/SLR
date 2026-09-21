"""One preregistered relational-loss pilot; no adaptive follow-on queue."""
import sys
import run_c16_control_v4 as runner

runner.RUN = 'seds-signrep-relational-offload-001'
runner.JOB = 'v4-c16-relation-001'
runner.OUT = runner.ROOT/'artifacts/slret_goal/jobs'/runner.JOB
runner.MODE = 'transfer'
runner.LOSS = 'relational'
runner.REFERENCE = 'seds-signrep-control-offload-001'
runner.ARTIFACT_CAP = 38

if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        runner.launch(__file__)
    elif sys.argv[1:]:
        raise SystemExit('Only --launch is accepted')
    else:
        runner.main()
