# GCN-R1 residual result — Cycle2

2026-09-22. [M] Read-only CPU replay of completed PH DEV519 score artifacts.
[Protocol](GCN_RESIDUAL_PROTOCOL.md) was written before residual calculations.
[Machine report](gcn_residuals_20260922.json) records all source hashes, ordered
IDs, per-query ranks/margins/confusers and the full descriptive summaries.

## Integrity and comparison scope

All nine score matrices (three seeds × fusion/pose/RGB) reproduce their stored
T2V/V2T R1/5/10/MedianR/MeanR and all per-query ranks exactly with the shared
evaluator. Galleries use identical519 ordered IDs and official pair-ID positives.
Official PH DEV annotations join all519 IDs. No model forward, training,
checkpoint loading, TEST access or active-job check occurred. This does not
verify model checkpoint bytes or restored features.

The analysis uses the previously selected checkpoints:42 at666 updates,1337
and2026 at222. It compares selected outputs descriptively, not equal-step causal
effects. Across these runs, mean bidirectional R1 is78.612717 with sample
SD0.096339 percentage points. This is selected-run variability, not a confidence
interval for a new method or expected generalization performance.

## Persistent errors

Counts below follow seed order42/1337/2026. Rank2–10 is one-based, equivalent
to zero-based1–9. Highest wrong candidate uses deterministic first-maximum
selection in saved candidate order; primary ranks retain the native metric's
direction-specific tie behavior.

| Quantity | T2V | V2T |
|---|---:|---:|
| Errors in selected seeds |115 /117 /115|107 /104 /108|
| Wrong in all three seeds |113|98|
| Persistent errors within rank2–10 |94 /93 /93|75 /73 /74|
| Persistent queries with a common strictly higher-scoring competitor |113|96|
| Persistent queries with same highest wrong candidate across seeds |111|96|
| Persistent errors also wrong in pose and RGB in all seeds |88|74|
| Exact-order candidate availability, entire DEV |5|5|
| Exact-order candidate availability, persistent errors |2|2|
| Strict exact-order confuser among persistent errors, each seed |0 /0 /0|0 /0 /0|

The88/113 (77.88%) T2V and74/98 (75.51%) V2T common branch failures weaken a
simple story that fusion merely discards an already correct branch result for
most persistent queries. They do not prove missing information: branch scoring,
shared priors and representation limitations can all produce common failure.

## Exact-order hypothesis decision

The predeclared contrast requires equal gloss multiset, different ordered
sequence and different exact native translation. Five DEV examples have such
a candidate; two belong to each persistent-error set. Yet none of those
candidates strictly outranks the positive in any seed.

**REJECTED:** the claim that this narrowly defined exact-order confuser slice
accounts for a material share (≥10%) of GCN-R1's persistent errors. No fuzzy
gloss relaxation, new positive mapping or synthetic sign-order labels follows.

This is not a test of trained CTC efficacy and is not proof that all temporal
information or all gloss supervision is useless. It removes the current cheap
evidential support for an order-specific C27 story. A broader gloss-recognition
benefit would be a different claim, with stronger collision risk and a required
same-gloss order-free control.

## Analytic implication for ordinary seed-score mixtures

For fixed query q, positive p and competitor c, if s_m(q,c)>s_m(q,p) for every
constituent m, then for any nonnegative weights w_m with positive sum:

    sum_m w_m [s_m(q,c) - s_m(q,p)] > 0.

Thus ordinary nonnegative mixtures of these three selected fused score matrices
cannot repair any of the113 T2V or96 V2T persistent queries with a common strict
competitor. This applies even to query-dependent nonnegative mixture weights.
It does not apply to newly trained representations, candidate-dependent weights,
nonlinear interactions or mixtures involving additional models. It is a bound,
not permission to implement those closed or untested alternatives.

An optimistic bound that grants perfect ranking to every other query is T2V
R1≤78.227360 and V2T R1≤81.502890 under such mixtures. These are loose mathematical
ceilings, not achieved scores or a claim of a realizable joint weight vector.
No ensemble was tuned or run as a new candidate.

## Length association

The TRAIN-defined threshold is7 gloss tokens. Among293 shorter DEV queries,
seed1337 has89 T2V errors (30.38%) and80 V2T errors (27.30%). Among226 longer
queries it has28 (12.39%) and24 (10.62%). Other seeds show the same direction.
This is descriptive: distinctive content, repeated phrases, duration, information
quantity and annotation ambiguity remain confounded. It does not justify a
length-calibration module or lexical-support revival. Signer counts are retained
in JSON without attributing causal nuisance.

## Decision

Prioritize a positively adequate recoverability/function-class diagnostic
(Q01/Q22), with information absence (Q23) as an alternative explanation. Do not
spend the next GPU budget on an unmotivated CTC/seed-mixture/fusion rescue. The
research program remains active and no primary method is selected.

Verification: three new fixtures pass (direction orientation, exact multiset
and translation constraints, common strict competitor versus ties/seed-specific
competitors). Saved-score parity tests are real-data evaluator checks, not an
independent baseline inference reproduction.
