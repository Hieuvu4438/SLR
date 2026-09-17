## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (AS-C20 control training exactly reproduced;
  freeze final inference independently replayed, freeze training not replicated)
- Version Label: AS-C30-v1

## Result and decision

The control exactly reproduces AS-C20. Freezing only the two text tables raises
final held mean R1 from17.490909% to18.254545% (+.763636pp), but DOES NOT meet
the original diagnostic adequacy. This is a bounded single-seed calibration
benefit, not evidence of no effect, not an adequate residual model, and not a
novel method or a GO pilot. No dev/test evaluation or PH-fitted retrieval
checkpoint was used. No new proposal is justified by this comparison.

## Locked execution and integrity

Protocol: `AS-C30_protocol.md`, registered before training. The harness
`methods/information_probe/freeze_training_comparison.py` executes the immutable
AS-C20 source after changing exactly four output/provenance literals in its AST.
The test restores those four nodes and requires complete AST identity. The
initializer wrapper changes only requires_grad for clip.positional_embedding
and clip.token_embedding.weight in the freeze arm. Optimizer groups retain both
parameters, but AdamW skips them because they have no gradients. Their absence
from global gradient clipping is an intended consequence of freezing.

Both arms: seed42, identical generic initialization,5721 TRAIN fit/1375 TRAIN
held, fixed1375 fit-evaluation subset, batch128,1000updates, sameFP32/noTF32,
augmentation, AdamW, per-group learning rates, warmup/cosine, decay, norm clipping
and unclamped scale. No best-checkpoint selection. AS-C28's other upstream
optimizer/decay/schedule/scale differences were NOT changed. This is not a
reproduction of an independently verified PH release training recipe.

Control exact-replay gate PASSED before freeze launch:

- All8 score files SHA256-identical and arrays bitwise equal to AS-C20.
- All40 training records and all4 evaluation records exactly equal.
- All305 final state tensors bitwise equal, no missing/extra keys.
- Serialized checkpoint container bytes intentionally differ because the
  preregistered protocol metadata differs; tensor-level equivalence is exact.
- Original AS-C20 source, run JSON and checkpoint hashes rechecked in validation.

Freeze audit:25,336,320 frozen parameters and124,387,843 trainable parameters.
Only the two registered tables frozen. Initial evaluation records exactly equal
between arms. Both tables remain bitwise equal to initializedFP32 values in the
saved freeze checkpoint, checked both at completion and independent reload.
Both tables changed in the control, as expected. The immutable training source
and harness were not edited during either run.

## Fixed results

All numbers below are percentages; differences are percentage points.

| Update | Control fit mean R1 | Freeze fit mean R1 | Control held mean R1 | Freeze held mean R1 |
|---:|---:|---:|---:|---:|
|0|0|0|.036364|.036364|
|250|77.490909|76.254545|12.545455|12.581818|
|500|92.254545|92.145455|15.200000|14.800000|
|1000|95.781818|95.454545|17.490909|18.254545|

| Final held metric | Control | Freeze | Freeze−control |
|---|---:|---:|---:|
|T2V R1|17.090909|17.381818|+.290909|
|V2T R1|17.890909|19.127273|+1.236364|
|T2V R5|41.309091|41.454545|+.145455|
|V2T R5|42.109091|42.036364|−.072727|
|T2V R10|54.109091|54.472727|+.363636|
|V2T R10|54.836364|55.272727|+.436364|

Final fit R1: control95.272727T/96.290909V;
freeze94.981818T/95.927273V. Strict different-input held errors:
control1120T/1102V, freeze1120T/1086V. Many errors exist, but that count alone
does not establish useful supervision from a sufficiently learned model.

Original gate, unchanged: BOTH fit R1>=80, BOTH held R1>=50, held mean gain
from initialization>=5pp, and>=50 strict different-input held errors in EACH
direction. Both arms pass fit, improvement-from-init and residual-count
requirements; both fail the held>=50-each requirement. Overall adequacy FALSE.
The relative+.763636pp improvement does not replace the absolute requirement.
These are one seed and one fixed split, not two independent seeds or a
strong-baseline PH-dev comparison. No p-value or confidence interval claimed.

## Independent validation and execution record

`validate_freeze_comparison.py` completed exit0: all16 saved matrices checked
against shared official R1/R5/R10 kernels, original asymmetric tie semantics
preserved. Reloaded final freeze checkpoint and raw held TRAIN features reproduce
the entire held score matrix exactly. Initial tables independently reconstructed
and checked. These verify metrics and final inference, not a second freeze
training trajectory or generalization across seeds/datasets.

| Run | Exit | Recorded seconds | Peak allocated GPU bytes |
|---|---:|---:|---:|
|AS-C30-CONTROL|0|444.783708|14,128,786,944|
|AS-C30-FREEZE|0|452.340418|13,926,030,848|
|AS-C30-VALIDATION|0|13.841306|1,115,479,552|

Training record wall times precede terminal wrapper validation/write. Timing is
not a scientific comparison or a speed claim. Each arm retains about629MiB
of score matrices/checkpoint, no deletion. JSON run records include code,
protocol, initialization, partition, generic checkpoint and output hashes.

Commands in /home/haipd/SLR:

    PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.freeze_training_comparison --arm control
    PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.freeze_training_comparison --arm freeze
    PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.validate_freeze_comparison

Preflight anomalies disclosed: first test collection failed on OUT imported
from the wrong module, corrected before training. Second test failed because
its inverse dictionary merged the artifact and experiment labels. The control
was inadvertently launched before checking that second failure. Corrected
node-wise reversal test passed while the harness remained unchanged. This was
a test defect, not a training defect; neither training run failed or restarted.
All51 focused tests subsequently passed. Process and output monitoring stayed
active; no timeout, nonfinite gradient/loss, or stalled output was observed.

## Fallacy scan:11/11 checked

1. Simpson: both directions and all registered checkpoints reported; no
   subgroup invariance claim from the mean improvement.
2. Ecological: fixed1375-row gallery outcomes not signer-level or semantic
   equivalence conclusions. Prefix folds are inferred, not verified recordings.
3. Berkson: the fixed internal held split is not a random population sample;
   no transfer of its absolute R1 to PH dev or other datasets.
4. Collider: no error-conditioned fitting or causal adjustment; fixed full
   galleries used for every comparison.
5. Base rate: gallery size, strict error counts, both fit/held outcomes and
   unchanged official positives/tie kernels reported.
6. Regression to mean: matched same-initialization control included; terminal
   checkpoint fixed in advance, not selected from a favorable intermediate peak.
7. Survivorship: both complete arms, both preflight test failures and every
   registered checkpoint retained; no failed training attempt hidden.
8. Look-elsewhere: exploratory cycle within a larger search; no significance
   claim, favorable-checkpoint selection or cross-seed inference.
9. Forking paths: preregistered one factor and endpoint; no AdamW/BertAdam,
   schedule, decay, learning-rate, scale clamp or adequacy change after results.
10. Correlation/causation: controlled computational contrast supports the
    observed effect in this specific run, not semantic forgetting as its cause
    or a universal benefit of freezing.
11. Reverse causality: post-training table drift alone was not treated as the
    cause of weak learning. This training intervention is distinct from AS-C29
    rollback and still does not explain the remaining generalization gap.

## Search consequence / next layer

AS-C29 rollback and AS-C30 frozen training both fail to supply an adequate
clean residual model; do not continue with table/optimizer rescue sweeps.
The isolated freeze factor has a positive final relative effect but does not
repair the prerequisite. Other recipe differences remain unresolved, not
falsified; ordinary corrections cannot supply novelty by themselves.

Next branch: text-side context availability on the strong R0 model (layerD,
distinct from table freezing or score-mask removal). Local module_clip.py
lines588–594 construct a causal text attention mask, and lines615–638 return
tokenwise hidden states as well as the sentence EOT representation. Before any
attention-mask intervention or proposed encoder, audit shared-prefix token
invariance and its prevalence/contribution among persistent confusers with
whole-gallery comparisons. EOT and later tokens CAN encode later content;
the code fact alone is not a failure certificate or a novel mechanism.
No new model or AS-C31 run launched in this turn. Preregister that diagnostic
first, preserve identical-text exclusions and full official evaluation, and
do not turn common-prefix effects into a renamed candidate-prior correction.

ARS influenced the pre-registration, exact replay gate, active monitoring,
independent metric/checkpoint validation and explicit11/11 interpretation scan.
Objective remains active; no GO, no global exhaustion, no Proposal8.
