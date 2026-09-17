# AS-C26/C27 — source overlap does not by itself explain weak calibration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (no independent full-run replication)
- Version Label: AS-C26-C27-result-v1

## Common-reference input support

AS-C26 compares internal held TRAIN and official DEV against the SAME5721fit
rows. Complete BPE unigram/bigram Jaccard is a simple lexical descriptor, not
semantic similarity. No error label determines neighbors or reference selection.

| Query/reference | N query | Exact deployed-text matches | Same-prefix queries | Mean nearest unigram Jaccard | Mean nearest bigram Jaccard |
|---|---:|---:|---:|---:|---:|
| Held / fit5721 | 1375 | 72 | 0 | .485774 | .315136 |
| Dev / fit5721 | 519 | 23 | 402 | .496031 | .326295 |
| Dev / all7096train | 519 | 26 | 519 | .505737 | .339342 |

Excluding same-prefix reference rows leaves dev/fit unigram mean unchanged and
bigram mean.326160; dev/alltrain means become.505499/.339087. Thus the measured
nearest lexical support changes little under this exclusion. It does not prove
source cues are irrelevant to videos or guarantee distributional equivalence.
Mean complete caption length15.748364held/14.990366dev. Per-row neighbors,
source counts, distributions and fixed-support-bin rank summaries are retained.

Important: inferred filename prefixes are NOT verified recording identities,
and overlap is NOT proof of clip leakage. The internal source-held construction
deliberately has zero overlap. Comparing historical R0dev to clean-held accuracy
would confound model, initialization, training set and gallery size. AS-C26
records each regime separately; no explanation of R0's advantage is inferred.

## Common model and gallery size

AS-C27 uses the unchanged clean AS-C20 update1000 model on official519DEV. It
compares against five predeclared519-row internal-held subgalleries, using the
saved1375×1375 score matrix and matching row/column indices. These overlapping
subsets are sensitivity analyses, NOT five trained models or independent seeds.
Salt0 remains the primary held subset, chosen before this evaluation.

| Population | T2V R1 | V2T R1 | Mean R1 |
|---|---:|---:|---:|
| Held519, primary salt0 | 27.167630 | 28.131021 | 27.649326 |
| Held519, salt1 | 26.011561 | 24.855491 | 25.433526 |
| Held519, salt2 | 28.323699 | 29.865125 | 29.094412 |
| Held519, salt3 | 29.094412 | 28.516378 | 28.805395 |
| Held519, salt4 | 27.167630 | 27.552987 | 27.360308 |
| Official dev519, same clean model | 33.911368 | 30.635838 | 32.273603 |

Dev exceeds the primary held subset by4.624277pp mean R1, and the fixed held
range by3.179191–6.840077pp. These are descriptive differences between distinct
query/gallery populations, not a causal source effect. Reducing held gallery size
also changes query composition; do not attribute the entire17.49→27.65 change
to distractor count alone. Original AS-C20 full1375held adequacy remains FAILED.

Within the fixed DEV519 gallery, seen-prefix402 rows have per-query R1
33.333333T/30.845771V; unseen-prefix117 rows35.897436T/29.914530V. T2V uses its
optimistic per-query diagnostic here, not an expanded-tie subgroup denominator.
There is no uniform seen-source advantage in these observed directional rates.
Content/signer/domain imbalance remains; this does not rule out source reliance.

Both equal-gallery populations remain weak, even on official dev. Do not blame
the source partition alone, lower learning thresholds, or use these errors as
if they arose from an adequate strong-model residual generator. No method,
teacher targets, corrective learner, benchmark change, new positive or GO.

## Execution and verification

AS-C26 completed exit0 in2.21s. Sparse set-intersection tests match expected
Jaccard values, nearest indices and source exclusions, including empty-set
behavior. AS-C27 completed exit0 in4.78s, peak allocated GPU1,115,479,552bytes.
Fixed checkpoint SHA256`3d1a135f8a6b93c69989acb821ea719e268fdf147e15fecbaff05fc8ffd1799f`;
partition/input/checkpoint score hashes checked. No training update, PH-fitted
retrieval checkpoint or test input loaded. One new small dev score matrix saved.

Both used preregistered protocols and kept all variants. AS-C27 subset selection
test verifies deterministic membership, original order and distinct salts.
Full focused suite46passed in1.20s. Full result runs were not independently
replicated. Source/rank attribution remains ANALYZED, not a verified causal claim.

## New local code finding and next priority

Post-run read-only inspection identified a concrete recipe difference worth a
separate audit. Local upstream`CLCL/main_task_retrieval.py:58` defaults
`freeze_layer_num=0`; lines709–728 retain all video parameters and text transformer
blocks at/above that layer, but freeze remaining text embedding/position tables
for the default2D path. AS-C20 trained all parameters. The upstream README's
default training command is explicitly a How2Sign example, so this is NOT proof
of the exact PH release recipe or a claim that AS-C20 was falsely labeled a
replication (it was explicitly a different diagnostic regime).

Source hashes: main_task_retrieval.py
`5f6e3606a170dd5c5a130ddadab67ee1d276f36e619b445f51bf1b6087420418`;
modules/optimization.py
`7dab5de2bbd1398e07ddbb18667544caae385d8ef9c17c55e0358303ed28b4b8`.
The default optimizer path also uses local BertAdam, not AS-C20's PyTorch AdamW;
matching beta/epsilon/weight-decay scalars alone does not certify equivalence.
Optimizer update semantics and actual frozen-key effects have NOT yet been
audited numerically in this cycle.

Next priority: bounded training-recipe/initialization audit, including exact
frozen parameter names/counts and upstream-versus-local optimizer update
semantics, before choosing any further clean learning run. Do not call ordinary
freezing, optimizer corrections or a better baseline a novel retrieval method.
No automatic multi-factor retry or wider dev tuning follows from this finding.

## Statistical/integrity scan (11/11, both experiments)

1. Simpson: directions, source slices, fixed support bins and populations separate;
   no pooled trend promoted to a universal effect.
2. Ecological: mean lexical support not a claim about sign-level distinctions.
3. Berkson: source-held construction and hash subsets change population; disclosed.
4. Collider: no source/error-conditioned causal adjustment or learned routing.
5. Base rate: all query/reference sizes, overlap counts and subset roles visible.
6. Regression to mean: no source intervention or selected-error gain claim.
7. Survivorship: all three support regimes and all five fixed subsets reported.
8. Look elsewhere: no winning subset, significance claim or reference-size selection.
9. Forking paths: bins, reference sets, hashes and final checkpoint fixed beforehand;
   recipe finding explicitly post-run, not a silently changed experiment.
10. Correlation/causation: common model/gallery removes two confounds, not all
    content/source/domain differences; descriptive source rates not causal evidence.
11. Reverse causality: no claim lexical/source support caused the observed weakness;
    code differences alone do not identify the cause either.

ARS discipline kept support descriptions and common-model differences separate
from causal/adequacy claims. Progress made; goal active, no global exhaustion.
