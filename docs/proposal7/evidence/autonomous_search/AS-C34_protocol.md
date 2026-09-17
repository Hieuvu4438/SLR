## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C34-v1

## Question

AS-C32 established a positive ordinary3-model ensemble control; AS-C33 failed
to collapse its gain into uniformly averaged weights. Does the score-ensemble
benefit require the original matched video/text encoder pairs, or survive
cross-checkpoint encoder pairing? This is a causal-location diagnostic, not
a new model, ensemble router, or proof of a linguistic bottleneck.

## Fixed3x3 cells and scorer contract

Use selected checkpoint seeds42/1337/2026 from AS-C32, all nine video-source
by text-source combinations. Each cell uses the VIDEO source's original
scorer/logit-scale. No temperature/scale tuning or additional normalization.
Thus diagonal cells have exactly their original contract; matched and crossed
aggregates preserve each source's marginal encoder/scale frequency. Directional
asymmetries must not be attributed to text/video semantics alone under this
explicit ownership convention. The post-encoding Filip scorer uses that scale,
not learned cross-checkpoint alignment. No learned coordinate alignment.

Reencode DEV519 using original native precision,128batch,feature_len64,alpha.9,
text_len32,no augmentation. Checkpoint/config/manifest provenance recorded;
ordered IDs and masks must agree. ALL THREE diagonal score matrices must be
bitwise identical to original artifacts. No score parity tolerance relaxation.
No test access, changed positives, selected best cross or additional training.

## Locked aggregates and comparisons

- Matched3: mean of diagonal, identical to AS-C32 uniform3 score artifact.
- Mismatched6: mean of all six off-diagonal cells.
- All9: mean of all nine cells.
- Each fixed video encoder averaged over three text sources (3row means).
- Each fixed text encoder averaged over three video sources (3column means).

Float64 accumulation then float32 output, same AS-C32 operation. All cells and
aggregates reported, no subset or new weights selected. Primary contrast is
mismatched6 minus matched3. Pairing-dependence signal: mean R1 difference<=−.5pp
AND conditional95% bootstrap upper<0. This is a diagnostic signal, not GO.
If it fails, do not infer equivalence or absence of all co-adaptation; report
actual differences/intervals and directional behavior. Row/column means are
localization descriptions, not independently confirmed best-cell claims.

Each diagonal is actually scored three times; verify repeated scores exact.
Six-pass and nine-pass repeated-diagonal aggregates must equal matched3 exactly.
This gives scoring-call-matched arithmetic controls for mismatched6/all9;
all retain3video and3text encoders. The complete diagnostic executes15full-gallery
score calls (9unique +6repeats), not15independent models. Row/column variants
have asymmetric encoder counts, which must be stated if discussed as controls.
Cached token reuse avoids repeated encoder work; no speed/latency claim.

Metrics: official R1/R5/R10, fixed persistent ranks, tie-aware source-prefix
cluster bootstrap10000draws,seed20260915,315clusters, full gallery fixed.
Override the helper's descriptive scope for each actual comparison; no inherited
mislabeling. No training/selection or cross-dataset uncertainty covered by CI.

## Execution

    PYTHONPATH=shared:. timeout 300 /home/haipd/miniconda3/bin/python -m methods.information_probe.encoder_grid_probe

Output AS-C34-ENCODER-GRID_run.json and18small score matrices under phase2.
No overwrite. Expected tens of seconds,<=4GB GPU,300s timeout; process/output
monitoring30–60s if needed. Unit tests precede execution. No new checkpoint
write. Report all failures and11/11 fallacy scan. Original GO requirements and
closed families unchanged; generic encoder mixing alone supplies no novelty.
