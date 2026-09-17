## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run recovery
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (process absence verified; cause unknown)
- Version Label: AS-C42-recovery-v1

## Interrupted original attempt

**Recovery finished:** attempt2completed8800updates,exit0; all264loggedtraining
and3evaluationprefixrecordsexact. Finalheldmean22.472727failsoriginaladequacy.
Finalscore replay andall8matrixmetricchecks pass; see AS-C42_result.md. No live
recovery process remains. Originalinterruptedattempt stays unchanged.

At 10:40 UTC, original worker140715 and timeout controller are absent, the GPU
has no compute processes, and execsession15558 returns Unknown process id.
The original JSON still says running at6600updates,2855.577688seconds. Console
ends at6600 with finite loss and no traceback. Six score files exist through
step4400; no final checkpoint. Accessible kernel journal has no entries; this
does not establish absence of an OOM or external termination. Exit code and
termination cause are unknown. This is not a completed or scientifically failed
condition. Preserve its JSON, console and all matrices unchanged.

Second intermediate result4400 has fit mean95.890909 and held mean17.781818.
It is not a selected endpoint or a replacement for required8800. No final
adequacy evaluation is authorized from the truncated run.

## Disclosed recovery

The user-authorized autonomous workflow permits a separately identified repair.
No resumable state exists, so attempt2 starts from the same initialization.
Keep the original protocol/harness/source unchanged. A wrapper executes the
original harness AST with ONLY its three distinct AS-C42 output-name literals
(six occurrences) renamed
to AS-C42-BUDGET-attempt2. Record wrapper hash and each substitution. Require
inverse substitution to reconstruct the original AST exactly.

Intercept output serialization only: before accepting any corresponding prefix
record, compare all training entries and complete evaluation records to the
interrupted attempt. Existing score-file hashes in those records also must
match. Stop and preserve a mismatch instead of silently tolerating it. Expected
prefix:264 training records through6600 and3 evaluation records at0/2200/4400.
No changes to seed, horizon, schedule, losses, batching, precision, parameters,
checkpoint timing, data or numeric code. No performance-based retry selection.

Launch the unchanged computation under a new process session with stdin closed
and durable stdout/stderr bootstrap and worker logs. This removes dependence on
the old observation session; it is a transport precaution, NOT a proven fix for
an identified termination cause. Internal7200/external7300-second timeouts
remain. Monitor actual PID/output every30–60seconds. No automatic retry loop.
Record launchPID and recoverable terminal returncode; do not restart if only
observation times out. Approximately one hour and <1GB additional disk expected.

Final validation must explicitly use the completed attempt2, require exact
prefix agreement, recheck all8 matrices and final checkpoint score replay, and
retain the interrupted attempt. No best-of-two result, extra method replicate,
GO or inference of global inadequacy.

## Execution record

The output-only transform inverse check passed. During test authoring, a patch
placed the last assertions of the existing adequacy test inside the new recovery
test, producing NameError before any recovery launch. The assertions were moved
back; all4 focused tests passed in0.73seconds. No training or scientific code
was changed to address this test-edit error.

Recovery launcher completed exit0, spawning detached timeout controller195592.
Launch record AS-C42-BUDGET-attempt2-launch.json; worker and terminal records
use the same prefix with -worker.json and -exit.json. Inspect those records and
actual processes for subsequent liveness, not the original15558session.
Final validation command is explicitly:

    PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.validate_extended_budget --attempt 2

The validator refuses incomplete training and additionally requires all264
training/3evaluation interrupted-prefix records exact. It has not run yet.

Later recovery milestone:2200updates reached; separate direct check confirms
all4 saved arrays and SHA256s at0and2200, first88trainingrecords and2evaluation
records exactly match interruptedattempt (exit0). This is partial-prefix
verification only; full6600prefix and final8800outcome remain pending.
