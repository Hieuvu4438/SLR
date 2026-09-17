## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (identity controls exact; new diagnostics not independently rerun)
- Version Label: AS-C32-C33-v1

## Outcome

AS-C32 provides a stronger3-model control:77.263969% mean R1 versus75.240848%
for the best single model, +2.023121pp, conditional cluster95% CI
[+.465506,+3.619048]. All registered diagnostic criteria pass. This is NOT a
method GO: ordinary ensembling, three-model cost, one fixed ensemble, no new
training or independent dataset confirmation.

AS-C33 uniform parameter average reaches76.011561%, but fails to preserve
the ensemble gain and regresses V2T relative to R0. Its gain over R0 has a CI
including zero. Do not promote it as a uniformly better single-model baseline
or tune its layers/weights to rescue this fixed control.

## Fixed full-gallery metrics

PH DEV519, official singleton positives and asymmetric tie handling unchanged.

| Variant | Models at inference | T2V R1 | V2T R1 | Mean R1 |
|---|---:|---:|---:|---:|
|Single42|1|74.181118|76.300578|75.240848|
|Single1337|1|74.181118|75.337187|74.759152|
|Single2026|1|73.988439|75.915222|74.951830|
|Uniform score average|3|76.878613|77.649326|77.263969|
|Uniform parameter average|1|76.493256|75.529865|76.011561|

| Variant | T2V R5 | V2T R5 | T2V R10 | V2T R10 |
|---|---:|---:|---:|---:|
|Single42|91.136802|91.907514|95.375723|94.990366|
|Uniform score average|92.292871|92.678227|96.531792|95.568401|
|Uniform parameter average|92.678227|92.678227|96.531792|95.953757|

Score ensemble persistent mean rank changes versusR0:−2.489130T/−2.356322V.
Weight average:−3.456522T/−3.632184V. Lower is better; favorable persistent
rank changes do not erase its V2T R1 regression. The weight average has
149,724,163parameters, the same architecture and one-model encoder/scorer cost.
The score ensemble has3models/encoder+scorer evaluations. Cache arithmetic
runtime is not a measured deployed-inference speed benchmark. Training and
checkpoint selection costs precede both diagnostics and are not free.

## Paired conditional uncertainty

Tie-aware official numerator/denominator cluster bootstrap,10000draws,
seed20260915,315inferred filename-prefix clusters; full gallery fixed.

| Comparison | Mean R1 delta pp | Conditional95% CI |
|---|---:|---|
|Score ensemble−single42|+2.023121|[+.465506,+3.619048]|
|Weight average−single42|+.770713|[−1.190531,+2.666700]|
|Weight average−score ensemble|−1.252408|[−2.647059,+.091747]|

These intervals do not cover model training, historical dev checkpoint selection,
exploratory search multiplicity or new-dataset uncertainty. A three-member
ensemble is ONE ensemble realization, not a passing2-of3-method-seed pilot.
AS-C33's prespecified point retention gate fails; the CI including zero for
the ensemble comparison is not evidence of equivalence or noninferiority.

Metadata issue: AS-C32's bootstrap helper copied its fixed descriptive scope
string into AS-C33. Its supplied arrays and named result fields are correct;
the generic string is not. `AS-C33_metadata_annotation.json` records the exact
correct comparison scopes against the immutable run hash. No computed number,
draw, threshold, source artifact or original result was changed/retried.

## What complementarity can and cannot repair

Score averaging recovers1/92 persistent T2V queries(index402) and4/87 V2T
queries(indices9,60,88,133), despite all three single rankings failing them.
No query correct in all three single models becomes wrong under the average.

88/92T and72/87V persistent queries have at least one SAME candidate strictly
above the positive in all three score matrices. A nonnegative weighted average
of those three scores cannot reverse that shared strict ordering. Only4T/15V
persistent queries lack such a common strict confuser; recovering one/four does
not certify that the remaining ones are recoverable by a different convex mix.
Float32 mean casting changes zero ranks in either direction, so these recoveries
are not rounding-induced tie changes. Most net gain is outside the persistent
intersection, not removal of its dominant common-confuser limitation.

## Identity and provenance checks

AS-C32 verifies all original checkpoint hashes, ordered query IDs, shared ID
hashes and published original metrics before mixing. Each of the three
same-model repeat means equals its original matrix exactly. Ensemble metrics
cross-checked with shared official evaluation code. Bootstrap preserves any
T2V tie expansion explicitly rather than assuming one rank per query.

AS-C33 verifies identical305state keys/shapes/dtypes; accumulates floating
tensors in float64 then casts to their original dtype; nonfloating buffers
must agree. Repeating seed42 gives exact identity for ALL305tensors, video/text
token caches/masks/CLS, and original full-gallery bridge scores. Uniform weight
average changes302of305tensors. State hashes are recorded; no large checkpoint
was written. Native precision used, original128-batch shapes retained.
Both saved AS-C33 score matrices additionally pass independent shared official
R1/R5/R10 checks after completion (read-only validation exit0).

## Execution

- AS-C32-ENSEMBLE: exit0,2.711056seconds,CPU score analysis, no encoder execution.
- AS-C33-SOUP: exit0,10.640112seconds,peak allocated GPU1,469,792,256bytes.
- Both protocols registered before their corresponding runs, same files remain.
-57focused tests pass; no training, crash, numerical retry, timeout or test access.
- No source/checkpoint overwrite or deletion. Outputs are one AS-C32 score
  matrix, two AS-C33 matrices, run JSONs and explicit metadata annotation.
- AS-C32run SHA bb16708f342f6b6cba64aa8c2949df4e4034375967b95e6ceaf5ff18767a3f11.
- AS-C33run SHA f3a6e4125d8e8eccc8840420d9ec0e5f08b167c787be4c73847f1b59aeb0fc49.

## Fallacy scan:11/11 checked

1. Simpson: both directions and persistent/full populations reported; weight
   averaging's favorable mean does not hide its V2T regression.
2. Ecological: filename groups not verified signer/recording independence;
   aggregate retrieval effects not semantic disambiguation labels.
3. Berkson: dev-selected checkpoints and persistent-error selection disclosed;
   full gallery primary, no evaluation-conditioned router fitted.
4. Collider: no adjusted causal estimate from selected error subgroups.
5. Base rate: full519gallery, persistent92/87denominators, recovered/common
   confuser counts and compute multiplicity explicit.
6. Regression to mean: exact repeat controls and all single models reported;
   no selection of a favorable subset or averaging coefficient.
7. Survivorship: all7AS-C32variants and both AS-C33variants retained; no failed
   run hidden. The descriptive metadata defect is explicitly annotated.
8. Look-elsewhere: fixed exploratory diagnostics with conditional intervals,
   not independently confirmed discovery or a full-search corrected test.
9. Forking paths: no model/layer/mixture/scale sweep after seeing results;
   AS-C33 was separately registered after AS-C32, with that dependency explicit.
10. Correlation/causation: controlled arithmetic establishes these fixed outputs;
    does not prove flatness, uncertainty calibration or linguistic mechanism.
11. Reverse causality: model identities precede aggregation; improvements do not
    retroactively validate earlier information-loss or nuisance explanations.

## Novelty and next action

Targeted primary-source screen in `AS-C32_candidate_collision_screen.md`:
output consensus is ordinary ensembling; parameter collapse is directly covered
by Model Soups/WiSE-FT; a failure-bank adaptive mixing formulation collides with
VRF. These are three screened mechanism routes, not surviving novel candidates.

Next preregistered question should locate the supported complementarity:
fixed3x3 video/text encoder-source crosses, diagonal score parity, ALL cells and
fixed aggregates reported with compute costs. Do not select the best cross or
fit mixture weights on dev. This tests within-checkpoint co-adaptation versus
marginal encoder variation before proposing a new operator. No AS-C34 launched.
ARS shaped fixed controls, conditional inference, prior-art screen and explicit
scope correction. Goal remains active; no GO, no global exhaustion, no Proposal8.
