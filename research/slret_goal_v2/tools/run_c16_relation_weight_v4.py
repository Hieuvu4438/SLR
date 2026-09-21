"""Final registered C16 refinement: RKD-D weight 1, all other factors fixed."""
import sys
import run_c16_control_v4 as runner

runner.RUN = 'seds-signrep-relational-weight1-offload-001'
runner.JOB = 'v4-c16-relation-weight1-001'
runner.OUT = runner.ROOT/'artifacts/slret_goal/jobs'/runner.JOB
runner.MODE = 'transfer'
runner.LOSS = 'relational'
runner.WEIGHT = 1.
runner.REFERENCE = 'seds-signrep-control-offload-001'
runner.ARTIFACT_CAP = 38

if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        runner.launch(__file__)
    elif sys.argv[1:]:
        raise SystemExit('Only --launch is accepted')
    else:
        runner.main()
