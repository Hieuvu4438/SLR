## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (original scores and padded repetitions exactly reproduced)
- Version Label: AS-C38-v1

## Finding

The fixed tail-padding control changes some numerical outputs and official
single-model V2T ranks, but does not change ANY ensemble query rank. The
AS-C32ensemble mean77.263969 and gain+2.023121pp versus seed42 survive exactly.
This is robustness of one ordinary3-model control, not a new method or GO.

|Model|Original meanR1|Padded meanR1|Changed T2V/V2T ranks|V2T top1 lost|
|---|---:|---:|---:|---:|
|42|75.240848|75.240848|0/2|0|
|1337|74.759152|74.759152|0/1|0|
|2026|74.951830|74.855491|0/2|1|
|Uniform3ensemble|77.263969|77.263969|0/0|0|

EnsembleT2V76.878613,V2T77.649326. R5/R10 also unchanged for all variants.
Padded ensemble versus padded42: conditional315filename-prefix cluster95%CI
[.465506,3.619048]pp,10000draws,seed20260915. Versus original ensemble0pp,
CI[0,0] because query outcomes are identical, NOT a population equivalence
certificate. All original AS-C32diagnostic lead gates remain true. Three
component checkpoints are not three independent ensemble-method seeds.

## What actually changed: tie order, not strict retrieval preference

Follow-up read-only inspection of all five model/query rank changes found:
the positive score, exact tied candidate set and entire strictly-above-positive
candidate set were unchanged. Official torch double-argsort rank changed
WITHIN the same tie. Thus no strict preference crossing was observed for any
changed query. Native batch-shape variation is real, but the observed R1loss
must not be described as new semantic confusion or lost score margin.

|Seed/query index|Zero-based V2T rank before→after|Candidates tied at positive|Strictly above positive count|
|---|---|---|---:|
|42/123|1→2|107,123,187|1|
|42/248|2→1|88,225,248,339|0|
|1337/248|1→2|88,225,248,339|0|
|2026/225|0→2|88,225,248,339|0|
|2026/345|13→14|8,35,345|12|

All changed queries are among the first512rows. Their positive and tied
candidate scores lie in the exactly unchanged512x512block. Only columns/rows
involving the final7encodings change, yet ordering of equal scores elsewhere
can change under the preserved official sorter. T2Vtie-expanded denominator
remains519 for all8matrices. These checks do not authorize replacing the
official evaluator, changing positives or claiming benchmark improvement.

## Intervention and exact controls

Original519DEV,3pinned best checkpoints, nativeFP16 unchanged. Encoder batch
sizes128/128/128/128/7 compared with128/128/128/128/128, final batch consisting
of7real rows and121copies of its first row. All encoder input tensors padded
consistently; fillers discarded before scoring. Gallery stays519x519. Masks
unchanged, all first512encoder outputs exact. Same128-block scorer and scales.

All3original score matrices replay bit-exact. Padded token/mask/CLS encodings
and scores repeated exactly for every checkpoint. Original ensemble exact
AS-C32. All8saved matrices independently checked against officialR1/R5/R10.

Tail video-token maximum differences:.0078125/.0078125/.0234375;
text-token:.00390625/.00390625/.0078125, in seed order42/1337/2026.
Max score differences:.007425308/.008674622/.011365891; ensemble.005115509.
All score changes localized to the last7rows or last7columns;512x512interior
exact for each model. Repeated filler rows are computation only, not candidates.
No precision selection, order shuffle, per-tower variant or hardware sweep.

## Execution and integrity

AS-C38-BATCH-SHAPE_run.json completed exit0,14.968374seconds,
peakGPU1,798,405,120bytes. Nine full-gallery score calls;8saved score matrices.
No encoder cache/checkpoint write, training, test access, upload or data changes.
Source/checkpoint/config/manifest/parent-score hashes and environment recorded.
AS-C32parent hash pinned before execution. No experiment crash/retry/timeout.

The first unit-test handle expired during continuation. A process check found
no test/experiment active and no run JSON. Only the two small tests were rerun,
passing before launch; no experiment was restarted. Full directory61tests pass
in1.61s; compileall and gitdiffcheck pass. Post-run score locality/tie checks
are additional interpretation, not preregistered model variants or score edits.

## Fallacy scan:11/11 checked

1. Simpson: individual models and fixed ensemble reported, neither hides another.
2. Ecological: aggregate stability not robustness of every model/query/batch.
3. Berkson: full declared DEV gallery, not only favorable queries.
4. Collider: no outcome-conditioned training/selection or causal adjustment.
5. Base rate:519queries,3models,oneensemble; all5rank changes reported.
6. Regression to mean: no model chosen after observing this perturbation.
7. Survivorship: all3original/padded/repeated conditions completed, none dropped.
8. Look-elsewhere: single fixed batching intervention; no search for a favorable
   hardware/precision/batch size. Search-wide inference remains exploratory.
9. Forking paths: protocol before run; post-run tie audit disclosed separately.
10. Correlation/causation: fixed intervention supports local numerical sensitivity;
    R1change is evaluator tie order, not demonstrated semantic representation harm.
11. Reverse causality: fillers/padding fixed without ranks; tie membership read
    after the outcome is a bounded explanation, not a learned policy.

## Consequence

AS-C32remains a stronger compute-explicit control. A padding/evaluator correction
is not a novel research candidate, and this result does not support pursuing it
as one. No GO or exhaustion. ARS influenced the preregistration, exact replay
requirements and distinction between numerical deltas, official tie ranks and
semantic interpretation. Retain original official protocol for future pilots.

Next layerG question is actual word-swap augmentation learning effect (Q21),
not another batching control. AS-C08inference and AS-C12/C18gradients leave that
causal question unresolved. First inspect and reproduce the original baseline
training recipe/first epoch; only then register an isolated on/off comparison.
Generic augmentation removal cannot itself qualify as a novel candidate.
No AS-C39 launched, no replay or training effect claimed by this plan.
