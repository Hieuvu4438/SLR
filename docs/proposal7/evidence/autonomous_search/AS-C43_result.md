# AS-C43: fixed gallery-neighborhood filtering does not survive

## Material Passport

- Origin: academic-research-suite / experiment-agent / run + validate.
- Date: 2026-09-15. Status: ANALYZED; baseline score replay and saved-operator
  reconstruction verified within the scopes below, not a verified method.
- AI-assisted, inline phases; no independent agent or human linguistic judgment.
- [Preregistered protocol](AS-C43_protocol.md), [run](AS-C43-GALLERY_run.json),
  [independent numerical validation](AS-C43-VALIDATION_run.json).

## Decision

Neither fixed operator passes in any seed: **0/3 smoothing, 0/3 sharpening**.
Both also fail against the strongest existing uniform-three ensemble. Do not
promote a gallery-filtering method, fit graph parameters, learn a replacement
graph, or try another descriptor/temperature as a rescue of this screen.

This was a competitor-feature diagnostic under user-loop §§13–15, not a new
method proposal. Gallery structure remains scientifically broader than these
two probes; the result is not a representation ceiling or a universal rejection
of structured retrieval. No three candidate derivations are justified by a
surviving new mechanism here. Move to a different causal layer.

## Complete registered results

Official mean bidirectional R1, percentage points. Single-model reference is its
own original checkpoint; ensemble reference is AS-C32 uniform3 (77.263969).
Each row's real and shuffled variant was run; no best-row selection.

| Source | Operator | Real graph R1 | Shuffled graph R1 | Real minus original | Nominal 95% CI vs original |
|---|---|---:|---:|---:|---|
| 42 | Smooth | 68.304432 | 65.895954 | −6.936416 | [−9.183722, −4.766532] |
| 1337 | Smooth | 68.015414 | 64.836224 | −6.743738 | [−8.908140, −4.653465] |
| 2026 | Smooth | 68.208092 | 67.437380 | −6.743738 | [−8.806262, −4.703452] |
| Ensemble | Smooth | 70.712909 | 70.327553 | −6.551060 | [−8.728797, −4.536675] |
| 42 | Sharp | 74.181118 | 73.217726 | −1.059730 | [−2.371659, 0.199203] |
| 1337 | Sharp | 73.988439 | 73.025048 | −0.770713 | [−2.010982, 0.486381] |
| 2026 | Sharp | 74.566474 | 73.699422 | −0.385356 | [−1.694958, 0.900000] |
| Ensemble | Sharp | 76.685934 | 75.818882 | −0.578035 | [−1.679104, 0.496056] |

All16 condition pairs, both directions' R1/R5/R10 and persistent-query ranks are
saved in the run JSON. Point losses for sharpening do not establish statistical
equivalence or prove harm at all possible settings; its CIs include zero.

For the ensemble, real sharpening changes T2V R1 from76.878613 to75.337187
(−1.541426pp), and V2T from77.649326 to78.034682 (+0.385356pp). The direction
nonregression gate fails. Persistent mean-rank changes are +2.521739 T2V and
+1.551724 V2T: worse in both directions. The smoothing ensemble improves mean
rank within the original persistent T2V stratum by0.652174 but worsens V2T by
0.551724, while losing substantial overall R1. A selected stratum is not a rescue.

Post-result arithmetic only, not a new run: even choosing the best of unchanged,
smooth or sharp *separately per direction* for this ensemble would select unchanged
T2V and sharp V2T, yielding only +0.192678pp mean R1—below0.5pp. This is a bound
on that finite menu, NOT on arbitrary continuous mixtures or new methods. No
direction-specific rescue was implemented or selected.

Real graphs beat shuffled graphs in point R1 in all eight real/control contrasts.
Only seed1337 smoothing has nominal real-minus-shuffle CI wholly positive:
+3.179191pp [0.683594,5.714286]. Its Bonferroni16 sensitivity interval includes zero
[−0.568713,7.027977], and it still loses6.743738pp against its original scorer.
Being less damaging than a shuffled perturbation does not establish useful
headroom over the unchanged baseline. Ensemble real-minus-shuffle CIs include
zero for both operators.

## What was intervened on

Per checkpoint, mask-aware mean contextual vectors define separate five-neighbor
video and text galleries. Each query's score vector s is transformed using its
gallery's neighbor-score average m: `s + 0.5*(m-s)` or `s - 0.5*(m-s)`. No labels
or other evaluation queries build the graph or enter the single-query function.
Shuffling relabels the graph nodes, preserving topology and degree distribution,
not each item's degree or a signer/source-specific distribution.

Both directions retain the standard full gallery and original positive mapping.
Different directional matrices are reported as such; there is no fiction of one
shared modified similarity matrix. Original scorer/tie handling is unchanged.
No optimization, new parameters, test access, new caption context, new labels,
SEDS assets, teacher targets or external feature stream. The three modified
checkpoints are not three freshly trained method replications; the uniform-three
output is one ensemble, not another seed.

## Numerical and execution validation

Run: `PYTHONPATH=shared:. timeout 600s /home/haipd/miniconda3/bin/python -m methods.information_probe.gallery_structure_probe`
in `/home/haipd/SLR`. Completed exit0,15.267749s,peak1,589,750,272GPU bytes.
Worker330237 is terminal; no pending training or diagnostic process.

Run SHA256 `6926ee09ae74edc6509f38dbf5377d407da76db4350522fa3cbfe5c9e518bfed`.
Validation SHA256 `e5c6b2155d0f91d50dd610e59948c651fb0d416fa7ed3b7e8f199d5a131714cf`.

- All three fresh encoder/scorer baseline replays equal original full519 matrices
  exactly before graph filtering. Checkpoint/config/manifest parents are hashed.
- Batched filtering and519 separate single-query calls match exactly for every
  single-checkpoint/direction/operator/control condition. No cohort-query coupling.
- Independent validator imports no probe-filter implementation. Explicit loops
  over candidates and neighbors reconstruct all24 single-checkpoint matrices
  exactly; explicit uniform sums reconstruct all8 ensemble matrices exactly.
- All32 saved matrices'96 directional R1/R5/R10 values equal the production
  evaluator; saved diagnostic rank arrays also match. Descriptor-based nearest
  neighbor boundaries/order independently certified to1e−12 cosine tolerance;
  graph adjacency conjugation is exact. This is not a bitwise independent encoder
  rerun or a proof that neighbors are semantically valid.
- Validator command: `PYTHONPATH=shared:. timeout 120s /home/haipd/miniconda3/bin/python -m methods.information_probe.validate_gallery_structure`.
  Exit0,4.814617s. Full probe suite:77tests passed3.71s,exit0.
- No run anomaly or retry. Console truncation was addressed by reading bounded
  JSON summaries; it was not treated as process failure. Saved graphs/descriptors
  and scores remain under `artifacts/proposal7/phase2/AS-C43/`.

## Statistical interpretation: 11/11 fallacy checks

Confidence: CAUTION, exploratory. Paired bootstrap resamples315 inferred
filename-prefix clusters10,000times, fixed full gallery, seed20260915. It preserves
official expanded T2V tie denominators. Sixteen baseline/shuffle contrasts are
reported with nominal95% and Bonferroni16 percentile sensitivity intervals.
The latter has finite Monte Carlo tail resolution and does not account for the
larger historical search on reused DEV. No p-values, effect-size labels borrowed
from unrelated domains, generalization guarantee or untouched-confirmation claim.

| Check | Assessment |
|---|---|
| Simpson's paradox | Both directions and all sources reported; ensemble sharp direction changes differ in sign, not hidden by mean. No exhaustive signer/source-stratified causal analysis. |
| Ecological fallacy | Neighborhood/aggregate effects are not individual linguistic diagnoses. |
| Berkson selection | Historical DEV-selected checkpoints and persistent-error slices are selected populations. |
| Collider bias | No adjusted causal model; graph neighbors and persistent strata are model-conditioned, so no causal semantic inference. |
| Base rates | Full519 gallery and original92T/87V persistent denominators retained. |
| Regression to mean | No worst-case-only efficacy claim; original full-gallery baselines and shuffled controls retained. |
| Survivorship | All16 registered condition pairs completed and reported; no dropped seed. |
| Look-elsewhere | All16 planned contrasts retained, nominal and multiplicity sensitivity shown; historic search uncertainty remains. |
| Forking paths | k5, pooling, signs/strength, permutation and metrics fixed before outputs; no hyperparameter or direction selection. |
| Correlation/causation | Deterministic score changes are attributable to the applied operator, not automatically to linguistic graph meaning. |
| Reverse causality | Graphs were computed without query scores or relevance; learned representation associations still do not reveal a causal semantic mechanism. |

## Prior, closure and remaining question

Graph diffusion is prior work, not novelty: [Iscen et al.](https://arxiv.org/abs/1611.05113)
study image-manifold diffusion, while [Zhang et al.](https://arxiv.org/abs/2012.07620v2)
explicitly connect neighbor-graph construction and message propagation to retrieval
re-ranking. These are primary abstract-level mechanism references, not reproduced
results, exact implementation equivalence or comprehensive novelty clearance.
See the protocol for query strings and unsuccessful direct-page access disclosures.

The off-diagonal operator can depend on this query's scores, unlike a candidate-only
constant offset. Nevertheless, with a fixed gallery it still defines a scalar
function `F_G(query,candidate)`; no violation of scalar-score expressivity follows.
The original cohort-assignment headroom is not transferred by this diagnostic.

No new candidate survives and no B0–B3 controlled method pilot or second-dataset
confirmation is claimed. The ARS workflow contributed the locked intervention,
independent reconstruction and explicit inferential limits. Next: a materially
different causal layer and an open, testable mechanism; no gallery-filter rescue.
The user's method-GO objective remains active and unfulfilled.
