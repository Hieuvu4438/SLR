## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (campaign incomplete; completed control checks below)
- Version Label: AS-C40-progress-v1

## Completed control and live queue

FINAL STATUS: allsixconditions completed and validated; noactivequeue. See
AS-C40_result.md and AS-C40-VALIDATION_run.json.0/3diagnosticgatespass; OFF
minusONmeans−.385356/+3.660886/−1.541426pp, everyOFFbelowinitialization.
Everything below is retained historical monitoring, not current liveness.

Updated after original queue terminated:42ON/OFF and1337ON/OFF complete.
Their initial states,batchorder,clean/visualinputtraces,schedules andinitialization
score hashes match within each pair; allOFFaugmented counts zero. Endpointmeans
42ON67.437380/OFF67.052023;1337ON65.703276/OFF69.364162. These are partial
descriptive results, not the registered>=2/3seed gate. BothOFFbelow initialization.

Original2026ONfailed duringprogressprint withBrokenPipeError at169updates;
failedrecord retained. Durable-output attempt2and subsequent2026OFF are now
queued in execsession21612,controllerPID78021,initialworkerPID78023. At snapshot
attempt2is live. See AS-C40_infrastructure_repair.md. Do not restart original
fourcompleted runs or treat oldsession9831as stilllive. Inspect actualstate.

Seed42ONcompleted260updates/20epochs,exit0,256.934556seconds,
43,259,711,488peakGPUbytes. EndpointT2V67.630058,V2T67.244701,mean67.437380.
Own separately evaluated initializationT2V74.181118,V2T75.337187.
This endpoint decline does not identify augmentation as the cause.

All280training/evaluation logs match the historical42first20epochs exactly,
excluding walltime/peakmemory. Firstepoch model/optimizer/scheduler/scaler/RNG/
sampler andfullscores exactly match AS-C39and original checkpoint.133120row
occurrences recorded,65224changed augmentedinputs. No checkpointwritten.

Sequential remaining queue launched in execsession9831:42OFF,1337ON,1337OFF,
2026ON,2026OFF; controller exits on any nonzero worker status. Initial active
worker42OFF; later four have not started at this snapshot. Inspect current
AS-C40-s*-*_run.json status ANDprocess/handle before any restart. Queue state
changes with time; this document is a snapshot, not proof a process is live now.

No intervention comparison or diagnostic gate calculated yet. After six runs
complete, validate_augmentation_training.py checks input/state/batch matching,
officialmetrics,initializationreference and registeredbootstrap/gates. It refuses
missing/nonterminal runs. No results selected by intermediate epoch; endpoint260
remains fixed. No novel candidate orGO asserted. Final11/11fallacy scan pending
the completed comparison; do not use this progress snapshot as the result report.
