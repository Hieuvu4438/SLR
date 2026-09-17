## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run monitoring / intermediate validation
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (saved intermediate metrics checked; training ongoing)
- Version Label: AS-C42-progress-v1

## First fixed evaluation: update2200

**FINAL STATUS:** attempt2 completed8800updates with exit0; finalheldmean22.472727,
bothdirectionsbelow50%gate. All8savedmatrixmetrics/ranks/residuals independently
checked and finalheldscores replayexact. See AS-C42_result.md and
AS-C42-VALIDATION-attempt2_run.json. No active worker remains; all progress
snapshots below are historical, not instructions to monitor/restart a process.

**Attempt2 replay milestone:** the recovery reached2200updates and reproduced
the first scheduled evaluation. A separate read-only CPU check compared all4
score arrays/file hashes at0and2200, the first88 training records and both full
evaluation records directly against the interrupted attempt; all match EXACTLY,
exit0. Fitmean94.509091 andheldmean14.000000 are reproduced, not new replicate
estimates or favorable checkpoint selection. No parameter/optimizer/RNG state
comparison is claimed. Next fixed evaluation4400, final8800 remain unchanged.

At the milestone liveness check, worker195622 was alive at16m04s; evaluation
record time954.401940seconds. Monitoring this turn advanced from575through2200,
with stable RSSaround4.8GB and no output stall, crash, mismatch or restart.
The entire264-record interrupted prefix is not yet verified;88records are.

**Superseding status:** the original process disappeared at6600updates without
an exitcode or checkpoint. These observations remain partial, not terminal.
See AS-C42_interruption_recovery.md. Separate attempt2 is live, worker195622;
at100updates its initialization and first4trainingrecords match exactly. All71
tests passed2.91s. Never use olderPID140715/session15558as current liveness.

The original worker continues toward update8800. This is not an endpoint
decision, a method result, or an early-stopping recommendation.

| Gallery | T2V R1/R5/R10 | V2T R1/R5/R10 | Mean R1 |
|---|---|---|---:|
| Fit subset,1375 | 94.036364 /98.109091 /99.418182 | 94.981818 /98.036364 /99.709091 | 94.509091 |
| Held,1375 | 14.545455 /36.727273 /49.600000 | 13.454545 /32.727273 /45.163636 | 14.000000 |

Both held directions are below the original50% adequacy threshold at this
intermediate step. High fit recall and low training loss do not replace that
absolute requirement. Final adequacy remains unevaluated; no checkpoint is
selected and no learning-rate, schedule, parameter or data change is made.

This longer-budget trajectory has a proportionally stretched schedule; its
2200-update result is not a same-trajectory continuation of AS-C20's final1000.
The short model's17.490909 held mean is not an equal-update causal comparison.
No significance, population generalization, semantic-ambiguity or information-
ceiling claim is made from either number.

## Independent saved-metric check

### Additional check of original step4400

Both original step4400 matrices were checked with the same production evaluator
and unchanged ordered singleton positives. All12 R1/R5/R10 values agree EXACTLY;
read-only CPU check exit0. This adds metric validation, not a new training run.

| Gallery | T2V R1/R5/R10 | V2T R1/R5/R10 | Mean R1 |
|---|---|---|---:|
| Fit subset,1375 | 95.781818 /98.181818 /99.418182 | 96.000000 /98.545455 /99.636364 | 95.890909 |
| Held,1375 | 18.763636 /42.472727 /54.909091 | 16.800000 /39.781818 /52.145455 | 17.781818 |

FitSHA256:c7a6331965a82f7d1294a5c1940084bcb5a289c02348e5ce86ec78a60d37fe09.
HeldSHA256:f5d040bba9c827ad2bb5a3400211803b881636687ae33899802676b5e67d21ab.
Both held directions remain below50% at this intermediate point. The eleven
interpretation limitations below apply unchanged; no endpoint or significance
claim follows. Attempt2 has not reached this step yet.

### Original step2200 check

A read-only CPU check loads both step2200 matrices and calls the production
`slr_common.evaluation.cico_eval.evaluate_score_matrix` with the unchanged
singleton positives and original ordered TRAIN IDs. All12 R1/R5/R10 cells
match the training report EXACTLY; the check completed with exit0.

- Fit score SHA256:
  44a113eb56804fe0093d9b4c863a765c419991631f6ddd895864d1cae1a847ae
- Held score SHA256:
  0f59da798d9e3f243b1259a216c1a01292db157e7feb38d21b079ff1a79b0d70

This does not independently replay encoders or training. The final validator
has not run; it correctly requires terminal completion first. The running
harness and all five recorded helper hashes were rechecked unchanged.

## Liveness and continuation

Later actual check of attempt2: worker195622 and detached timeout195592 live
at3m12s,425updates. All17 logged training records and initialization evaluation
match the interrupted attempt exactly. RSS approximately4.82GB, finite loss,
no stall or restart. This is a verified wait, not completed training. The
recovery's exact-prefix scope is **logged training summaries plus full saved
evaluation records/score hashes**, not every parameter/optimizer/RNG state:
those were not saved by the interrupted attempt. Do not inflate a matching
summary trace into an unobserved bitwise model-state guarantee.

The remainder of this section records the original attempt's historical
monitoring; it is not current liveness.

Exec session15558; workerPID140715 remains live. The latest recorded process
check is elapsed17m11s,2350updates,epoch53, RSS approximately4.81GB. Progress
checks during this turn advanced through425,625,800,925,1125,1325,1475,1600,
1750,1900,2025,2150 and2275updates; loss remained finite and outputs advanced.
No crash, retry, numerical failure or output stall was observed. These are
historical snapshots: verify the actual process/run JSON on every continuation.

Next fixed evaluation4400, final8800. Keep the internal7200s/external7300s
limits and durable logging. Do not restart solely because a tool handle expires.
No second training run or DEV/test data access occurred.

## Interpretation checks:11/11

1. Simpson: both directions and both fixed galleries reported separately.
2. Ecological: aggregate recall is not an individual linguistic diagnosis.
3. Berkson: the fixed internal held partition is not a new population sample.
4. Collider: no post-result conditioning or confuser-selected causal model.
5. Base rates: both1375-gallery denominators retained.
6. Regression to mean: no worst-query improvement or early selection claim.
7. Survivorship: the fixed intermediate result is retained despite low held R1.
8. Look-elsewhere: exploratory reused held set, not confirmatory significance.
9. Forking paths: no changed endpoint, schedule or adequacy threshold.
10. Correlation/causation: fit/held disparity is not attributed to a specific
    representation defect; duration and schedule are a joint budget condition.
11. Reverse causality: no causal direction inferred from loss/recall association.

The separate `AS-C42_auxiliary_checkpoint_audit.md` records an already-present
H2S checkpoint's schema and unresolved lineage, including correction of an
initial bad filename filter. It is not a resource substitution or AS-C43 run.
