## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (final held-score replay exact; full training not independently repeated)
- Version Label: AS-C42-final-v1

## Result: larger-budget calibration still fails absolute adequacy

The completed attempt2 reached all8800updates/200training passes with exit0.
Its final held T2V/V2T R1 is22.545455/22.400000%, mean22.472727%. Both directions
remain below the ORIGINAL50% adequacy requirement. Fit recall and held residual
counts pass their respective thresholds, but do not replace that requirement.
This is not an adequate residual probe, a method candidate, GO, or a global
research barrier. The registered larger-budget calibration lead is insufficient.

| Fixed step | Fit T2V R1 | Fit V2T R1 | Held T2V R1 | Held V2T R1 | Held mean R1 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 0.000000 | 0.000000 | 0.072727 | 0.036364 |
| 2200 | 94.036364 | 94.981818 | 14.545455 | 13.454545 | 14.000000 |
| 4400 | 95.781818 | 96.000000 | 18.763636 | 16.800000 | 17.781818 |
| 8800 | 96.145455 | 96.363636 | 22.545455 | 22.400000 | 22.472727 |

All evaluations use the fixed1375fit subset and all1375held TRAIN rows;
training uses5721fit rows. No best-step selection. Original source-prefix
partition, singleton positives,64visual/32text inputs and official tie policies
remain unchanged. The held population has been used exploratorily before and
is not untouched confirmation data. Filename-prefix disjointness does not
independently establish recording disjointness;72held rows share exact fit text.

## Fixed endpoint versus short control

AS-C20's1000-update held mean is17.490909%; AS-C42's8800-update mean is22.472727%,
a descriptive+4.981818pp. T2V improves5.454545pp and V2T4.509091pp. AS-C30's
1000-update freeze-only mean18.254545% also failed the same absolute gate;
AS-C42 does not apply that freeze intervention. The original AS-C20 short run
was fully reproduced by AS-C30 control before this extension was registered.

| Final gallery | T2V R1/R5/R10 | V2T R1/R5/R10 |
|---|---|---|
| Fit subset | 96.145455 /98.181818 /99.418182 | 96.363636 /98.254545 /99.709091 |
| Held | 22.545455 /45.818182 /58.400000 | 22.400000 /45.090909 /56.509091 |

Held gain from initialization is22.436364pp, exceeding the5pp initialization
criterion. Strict different-text-input held errors number1042T2V and1037V2T,
each exceeding50. Both fit R1 exceed80%. Nevertheless, both held R1 fail50%:
learning_gate=false, residual_count_gate=true, diagnostic_adequacy=false.
Error abundance does not demonstrate that those errors are suitable residual
supervision. It is especially not evidence of irreducible semantic ambiguity.

This comparison jointly extends training and proportionally stretches the
warmup/cosine schedule; it is NOT pure exposure on a common optimization
trajectory. The same peak learning rates, AdamW, all trainable parameters,
batch128, seed42, FP32, augmentation and scorer were retained. No causal
attribution to duration alone, no optimizer superiority, no upstream PH recipe
reproduction, no statistical significance or cross-seed generalization claim.
The result cannot be directly compared with R0's519DEV gallery or AS-C32's
three-model ensemble as if their populations and initializations were matched.

## Interruption and recovery integrity

Original worker140715 disappeared after its6600-update record,2855.577688seconds,
without a traceback or final checkpoint. Exitcode/cause remain unknown; the
stale-running JSON is preserved, not relabeled as a completed scientific run.
Its six matrices, logs and metadata remain unchanged. Accessible kernel journal
absence does not rule out external termination/OOM. No claim that detachment
fixed a proven root cause is made.

The separately disclosed attempt2 restarted identical computation because no
resumable checkpoint existed. An output-only wrapper renamed three distinct
strings at six AST locations; inverse substitution reconstructs the original
harness AST exactly. A detached session and durable logging changed process
supervision, not numerical code or protocol. No seed, horizon, data, schedule,
precision, objective or threshold rescue. No performance-selected replicate.

All264 available original training summaries through6600 and all3 original
evaluation records at0/2200/4400 match attempt2 exactly. The records include
saved-score hashes. A separate direct comparison of these full prefixes passed
at finalization; earlier direct array/hash comparisons passed for0/2200.
This scope does NOT imply every intermediate parameter, optimizer or RNG tensor
matched: those states were not saved. Nor is the unique6601–8800segment an
independent full training reproduction. The recovery is one successful condition,
not two method seeds or two independent completed replicates.

Attempt2 training duration3816.357260seconds, supervisor exit0 at3817.620507seconds;
peak allocated GPU14,128,786,944bytes. There are352 logged training records and
four fixed evaluations. No active training worker remains. Original progress
snapshots are historical and must never trigger a restart.

## Final validation

Executed after confirmed training completion:

    PYTHONPATH=shared:. timeout 180 /home/haipd/miniconda3/bin/python -m methods.information_probe.validate_extended_budget --attempt 2

Validator exit0,10.775077seconds, peak1,115,479,552GPUbytes. All8saved matrices
pass production-evaluator R1/R5/R10 checks, exact per-query rank comparisons,
and independently recomputed strict residual index/count comparisons. The
absolute adequacy logic is independently recomputed and agrees with training.
Protocol, source, recovery wrapper, helper, parent, partition, generic checkpoint
and final checkpoint hashes are checked. Reinitializing the architecture,
loading final8800weights and re-encoding held inputs reproduces the entire
1375x1375held score matrix EXACTLY (max absolute difference0).

- Successful run SHA256:
  08dc062979c5235b78b76219062f5f902de842c2a1a1ab6df0ec8404114ad191
- Final checkpoint SHA256:
  b4f78c9cdbc61672a3f44c640edd2ea83a7ef58a5b8d341035df1316bc5b175d
- Validation result SHA256:
  94f5d9127d8a1a57ce94174ce04d679cb211d10b515f1b7409753d57ecd4acbe

Files: AS-C42-BUDGET-attempt2_run.json; AS-C42-VALIDATION-attempt2_run.json;
artifacts/proposal7/phase2/AS-C42-BUDGET-attempt2/final_step8800.pt and eight
score matrices. No original output overwritten. All71probe tests pass1.78s.
Test-authoring NameError and correction before recovery are disclosed in the
recovery report; it was not a training failure. No test-split access, new
positives, SEDS assets, new data, commits or private external uploads.

## Interpretation scan:11/11 checked

| Check | Scope and limitation |
|---|---|
| Simpson's paradox | Both directions and both fixed galleries retained; no hidden pooled success. |
| Ecological fallacy | Aggregate retrieval does not certify individual linguistic distinctions. |
| Berkson selection | Fixed exploratory held partition and fit subset, not a population sample. |
| Collider bias | No result-selected subset used for a causal estimate or adequacy decision. |
| Base-rate neglect |1375gallery denominators,1042/1037strict errors and all thresholds disclosed. |
| Regression to mean | No selected worst-query recovery or best-checkpoint claim. |
| Survivorship bias | Interrupted attempt retained; retry determined by missing process/checkpoint, not performance. |
| Look-elsewhere | One registered extended budget; no p-value, confirmatory claim or search-wide guarantee. |
| Forking paths | Fixed8800endpoint and original gates unchanged despite intermediate failures. |
| Correlation versus causation | Joint budget/schedule effect not duration alone; fit/held gap not a diagnosed semantic defect. |
| Reverse causality | No causal mechanism inferred from loss/recall association or observed error abundance. |

## Search consequence

Do not promote this model's residuals as calibrated supervision, lower the50%
gate, or infer sufficiency from failed probes. Close this fixed larger-budget
calibration lead and do not launch a duration/schedule/optimizer/freeze/seed
rescue sweep. The broader Q01 representation-sufficiency question remains
unresolved. A materially different mechanism must carry a new measurable
decision consequence before another candidate cycle.

The auxiliary H2S release-member check verifies bytes against the public archive
only. It does not prove PH-unfitted training lineage or authorize substituting
that checkpoint into this failed calibration. No AS-C43 has been registered or
launched. No candidate meets GO; no global exhaustion or stronger barrier is
established. The autonomous research goal remains active.
