## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before extended training)
- Version Label: AS-C42-v1

## Question and open decision consequence

Can the generic-initialized TRAIN-only calibration meet its original absolute
adequacy criterion at a substantially larger, fixed training budget? AS-C20
used 1000 updates, about 23 passes, and AS-C30's isolated freezing comparison
did not meet adequacy. Neither result makes the frozen representation sufficient
or proves the absence of useful held-out residuals. Q01 remains a high-value
unresolved diagnostic with an inadequate probe, not a rejected architecture to
rename. This run addresses budget adequacy, not table freezing or optimizer
choice. Intervening cycles have examined other causal layers.

Passing would permit a separately registered, TRAIN-only residual diagnosis
with an adequately learned retriever, before any bridge to strong R0. It does
not authorize a correction teacher trained on the same held labels, OTTER,
ELSC/PMGR/RPCA, a new benchmark, or any closed method. Failing means this larger
budget remains inadequate, not a global information barrier. Do not rescue
failure with a duration, learning-rate, optimizer, freezing or seed sweep.

An initial exact-duplicate headroom check merely re-read AS-C04: 507 distinct
DEV text inputs, 17 rows in repeated-input classes. That is already-known
evidence, not a new cycle, semantic ceiling, or reason for a duplicate method.
The AS-C28 upstream-default audit does not recover the historical PH recipe.
This run explicitly is NOT an upstream reproduction.

## Locked training contract

Reuse immutable `clean_train_calibration.py`, SHA256
e812d98277cd7f7d00c02600722ba53a5d2dafc767f3db44df6a26f572d7dc71.
An audited AST harness changes only:

- Total updates 1000 to 8800 = 200 passes times 44 complete batches per pass.
- Warmup 100 to 880; retain the same linear-warmup/post-warmup cosine function.
  This stretches the schedule with the budget. It changes learning rates at
  matched update numbers, so the comparison is NOT pure extra exposure on an
  identical optimization trajectory. No restart from the old zero-LR endpoint.
- Fixed evaluation steps 0/250/500/1000 to 0/2200/4400/8800.
- Hard internal timeout 3600 to 7200 seconds, and output/provenance names.

Keep seed42, batch128, 5721 fit/1375 held source-prefix partition, the same
1375 fit-evaluation subset, all active trainable parameters, FP32/no AMP/no
TF32, AdamW, learning-rate peaks, decay, global clipping, augmentation, scorer,
initialization and data order code unchanged. In particular, the multiplication
by 100 in percentage metrics MUST NOT be changed when replacing warmup100.
Tests reverse every registered AST change and require original AST identity.

Generic CLIP initialization only; no PH retrieval checkpoint, R0 embeddings,
official DEV/test input, new positives or new data. Retain AS-C19 extractor and
partition provenance limits. Filename-prefix disjointness is not independently
verified recording disjointness. Existing 72 held rows with fit text overlap
remain; repeated exploration means this is not untouched confirmation data.

Before the first update, require both initialization score files and evaluation
records to match AS-C20 exactly. AS-C30 already reproduced the entire short
control, so another 1000-update control is not silently added. Compare endpoints
descriptively; one seed and one extended budget, no significance claim.

## Outcomes and immutable adequacy gate

At final8800 only, require BOTH fit R1>=80%, BOTH held R1>=50%, mean held R1
improvement>=5pp over initialization, and at least50 strict different-text-input
held errors in EACH direction. Reuse the unchanged AS-C20 adequacy function.
Report all fixed evaluations, both directional R1/R5/R10 and full rank arrays.
No best-checkpoint selection, early stopping on metrics, or threshold lowering.
The original 1000-update and freeze results remain visible regardless of outcome.

A passing calibration is not novelty, B0/B1/B2/B3 survival, a three-seed method
pilot, independent confirmation, or GO. It is not directly comparable to R0's
519-DEV gallery or the AS-C32 ensemble. It can only support the next diagnostic
question under a newly registered analysis.

## Execution and monitoring

    PYTHONPATH=shared:. timeout 7300 /home/haipd/miniconda3/bin/python -m methods.information_probe.extended_budget_calibration

Working directory `/home/haipd/SLR`. Expect roughly 60–70 minutes based on the
446-second short run; <=20GB GPU, approximately 1GB disk. GPU is free and 56GB
disk available at registration. JSON progress every25 updates; four evaluations.
Durable exclusive stdout/stderr file avoids the AS-C40 observation-pipe failure.
Monitor process/output every30–60 seconds. Internal7200-second and external7300-
second timeouts are hard limits. Plateaus advisory; no automatic metric stop.

Outputs `AS-C42-BUDGET_run.json`, `artifacts/proposal7/phase2/AS-C42-BUDGET/`
scores and final8800 checkpoint, and `AS-C42-BUDGET-console.log` beside that
directory. Refuse pre-existing outputs. No overwrite, automatic restart or
checkpoint selection. A crash preserves the failure and needs a separately
disclosed recovery decision; the source program saves a checkpoint only at end.

After completion, independently recompute all saved-score metrics and original
adequacy gate, check unchanged source/helper hashes, evaluate the final checkpoint
for deterministic score replay, and report all11 statistical fallacy checks.
