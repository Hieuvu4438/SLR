## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate / run
- Origin Date: 2026-09-15
- Verification Status: VERIFIED (specified historical first-epoch numerical replay only)
- Version Label: AS-C39-v1

## Exact replay succeeds

The archived seed42baseline first epoch was reproduced exactly:312model state
tensors,302optimizer parameter states, scheduler, AMPscaler, Python/NumPy/Torch/
CUDA RNG, sampler generator, trainable-name list,13training log records and the
epoch0evaluation log (excluding walltime/peakmemory). The complete519x519DEV
score matrix is bit-identical to the archived selected checkpoint's evaluation.
T2V74.181118,V2T76.300578,mean75.240848. Replay gate TRUE.

Historical trainer retrieved from commit39449e18def6b154944ceeaed39dfd5c570882a3,
old pathelsc/train.py, SHA256
95081c539510ec51b508a7813ff16329ae9c249a17c03a0756bfb1f366843800.
The current tree had moved the module; a naive path comparison initially
displayed it as entirely new. Correct historical path recovered before running.
Current trainer additionally evaluates initialization; historical trainer did
not. The unmodified archived train function was compiled into checked globals,
avoiding this random-state/order confound. Baseline-active dependencies passed
ASTchecks; factory-only docstring/import movement normalized explicitly.
Changed paired_score is unused by the baseline path; active score/encoders exact.

The original200epoch/2600step schedule and260step warmup remained unchanged.
Original512batch,13drop-last updates, AMPbf16, AdamW, seed42, augmentation and
release initialization. At first checkpoint-save callback, after epoch0DEV
evaluation, states were compared and execution intentionally ended. No checkpoint
was written; this is a registered stop boundary, not a crash or timeout.

Completed exit0,16.666957seconds,43,259,711,488peakGPUbytes;24CPUthreads,
Torch2.11.0+cu128/CUDA12.8,RTX5880Ada. Two new tests passed before launch.
Full probe directory later passed64tests in3.05s, including AS-C40trace test;
AS-C40additional matched-batch validator test passed separately. Compilation
and gitdiffcheck passed. These tests do not replace numerical replay evidence.
Output AS-C39-TRAIN-REPLAY_run.json, separate run logs/config/provenance and
AS-C39-epoch0_scores.npy. No raw data, original checkpoint or source edits;
no test access, teacher, ELSCauxiliary objective or external upload.

## Scope and fallacy scan:11/11 checked

1. Simpson: no subgroup effect inferred from identical aggregate scores.
2. Ecological: replay of one seed/epoch is not all-training reproducibility.
3. Berkson: seed42chosen because its archived selected checkpoint is epoch0,
   not because an augmentation result was favorable.
4. Collider: no fitted adjustment or outcome-conditioned intervention.
5. Base rate: one seed,13updates,14logs,312state tensors explicit.
6. Regression to mean: no pre/post improvement claimed.
7. Survivorship: all declared numerical checks passed, no dropped mismatch.
8. Look-elsewhere: one fixed replay, no precision/recipe tuning for agreement.
9. Forking paths: historical/current path and evaluation difference found before
   run, protocol fixed before launch, no hidden retry.
10. Correlation/causation: exact replay establishes control fidelity, not word-swap
    harm, novelty, clean-pretraining provenance or generalization sufficiency.
11. Reverse causality: archived epoch0availability determined replay endpoint;
    no augmentation outcome used to choose it.

ARS required exact numerical rather than approximate metric agreement. This
successful control permits an isolated training intervention; it is not a method
candidate or GO. Next register matched on/off runs at the original warmup
boundary, three seeds, unchanged original schedule, no endpoint selection.
