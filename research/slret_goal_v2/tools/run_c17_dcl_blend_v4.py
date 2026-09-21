"""One registered warm-start strength refinement; no adaptive follow-on."""
import sys
import run_c17_dcl_v4 as pilot

pilot.RUN = 'seds-fused-dcl-blend005-offload-001'
pilot.JOB = 'v4-c17-dcl-blend005-001'
pilot.DCL_WEIGHT = .05
pilot.ARTIFACT_CAP = 40

if __name__ == '__main__':
    pilot.configure()
    if sys.argv[1:] == ['--launch']:
        pilot.runner.launch(__file__)
    elif sys.argv[1:]:
        raise SystemExit('Only --launch is accepted')
    else:
        pilot.runner.main()
