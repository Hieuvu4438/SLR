# AS-C45: exact-gloss persistent-confuser lead unsupported

2026-09-15; academic-research-suite, ANALYZED annotation/score audit. AI-assisted,
not expert linguistic review, new positive labels or a method pilot.

The registered screen completed first execution, exit0, 1.281996s. Exact gloss
means whitespace-token sequence equality, not synonym or semantic equivalence.

| Split | Unique gloss sequences | Repeated groups / rows | All equal-gloss pairs | Different native / different native and model translations |
|---|---:|---:|---:|---:|
| TRAIN, 7,096 | 6,900 | 70 / 266 | 843 | 372 / 345 |
| DEV, 519 | 512 | 5 / 12 | 9 | 4 / 4 |

Seven DEV queries have an eligible different-text candidate. Among original
persistent queries, candidate availability is3/92 T2V and4/87 V2T. Strictly
higher-scored eligible confusers per seed42/1337/2026: T2V1/1/3; V2T2/2/3.
Across all three seeds: **1/92 T2V (1.09%) and2/87 V2T (2.30%)**. Each count also
has the same strict competitor across seeds. Both fall below the registered10%
material-burden threshold: diagnostic_lead=false, method_go=false.

Group-based pair enumeration agrees with an independent exhaustive all-pairs
comparison on TRAIN and DEV. A separate scalar candidate/score-loop validator
reconstructs all persistent counts exactly, with all input hashes unchanged;
completed exit0,.196058s. It intentionally shares the historical torch V2T tie
convention; no independent tie-policy claim. Full suite81passed3.14s, exit0.

Run: [AS-C45-GLOSS_run.json](AS-C45-GLOSS_run.json),
SHA6e3c912ef8a11eb14201579810cb0fa79cffddaf48f1639842174d9300176163.
Validation: [AS-C45-VALIDATION_run.json](AS-C45-VALIDATION_run.json).
All input/code/protocol hashes retained. No retries, model/checkpoint/features,
TEST data, new encodings, fitting or new annotation. No active worker remains.

This is descriptive evidence conditional on this heavily reused DEV gallery,
not an inferential population estimate or an efficacy effect size. Threshold
failure is not proof of no ambiguity. Gloss collisions do not prove equivalent
meaning or isolate non-manual distinctions; differing translations can reflect
wording variability. The baseline consumes visual features, not glosses, so no
visual information ceiling follows. TRAIN pair prevalence is not a held-out
retrieval failure. Strict confuser checks do not replace official tie-expanded
recall. No p-values or confidence intervals are appropriate for a claimed
method gain because no method was tested.

Stop this exact-gloss persistent-confuser route. No fuzzy-gloss, group-positive,
hard-mining or lexical-support rescue. User's latest direction now prioritizes
source-level weaknesses in CMCM/SAN/SEDS/SLRT/UPRet/CiCo. Continue by comparing
actual training/inference call paths and testing a new source-derived hypothesis;
do not repeat already recorded runtime/masking issues as new research progress.
