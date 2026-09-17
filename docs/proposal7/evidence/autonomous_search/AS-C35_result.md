## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (identity replays and rotation invariance checked; new fitted diagnostic not independently rerun)
- Version Label: AS-C35-v1

## Decision

The fixed TRAIN-fitted shared orthogonal map does NOT explain away AS-C34's
pairing penalty. Aligned mismatched6 mean R1=76.011561 versus original76.107900,
delta−.096339pp, conditional95% CI[−1.183432,+.956023]. The registered
coordinate-explanation lead fails. This is not evidence of zero effect,
equivalence, or absence of all coordinate incompatibility: one specific
pooled-video/EOT alignment was tested, not every token-level alignment.

## Fixed controls and outcomes

Same map applied to BOTH encoders of each checkpoint; seed42 identity,
1337/2026mapped to42 using TRAIN7096only. Each row has one normalized pooled
valid-video anchor and one normalized text EOT anchor,14192anchors/model.
No centering, rescaling, layer selection, dev-fit, optimizer or encoder update.
Shifted-target negative control rolls TRAIN target rows by1jointly for video/text;
it is not a semantic-null dataset and neighboring rows may share content.

| Coordinate condition |Matched3 mean R1|Mismatched6 mean R1|All9 mean R1|
|---|---:|---:|---:|
|Identity|77.263969|76.107900|76.589595|
|TRAIN-aligned|77.263969|76.011561|76.204239|
|Shifted TRAIN targets|77.263969|28.227360|68.689788|

Aligned mismatched6 T2V76.107900/V2T75.915222; original75.915222/76.300578.
Aligned minus original matched3 mean−1.252408pp, conditional95% CI
[−2.233010,−.294695]. The aligned control beats the shifted one, but that is
insufficient: it does not improve the relevant original mismatched control.
No scoring mixture, anchor type, per-tower map or precision sweep follows.

## Alignment actually learned transferable anchor structure

|Source→42|TRAIN cosine before / after|DEV cosine before / after|
|---|---|---|
|1337|.998277 / .999227|.998147 / .999165|
|2026|.762224 / .826091|.760116 / .824945|

These averages mix equal counts of normalized video/text anchors, not all
tokenwise pair relations. Improved anchor agreement does not imply improved
retrieval, aligned linguistic meaning or adequate information recovery.
Shifted-target true-pair TRAIN cosines after fit:.317570/.270377;
DEV:.318731/.272124 for1337/2026. This supports sensitivity to correct
same-input correspondence, not a causal semantic annotation claim.

All maps computed by float64 SVD of X^T Y, Q=UV^T. Maximum Q^TQ−I error
2.442491e-15, within registered1e-10. Smallest singular values recorded,
not suppressed. Maps persisted as float64 arrays; applied in float32 to all
token/CLS slots including padding, with masks/scales unchanged.

## Geometry and score verification

Identity reproduces ALL9AS-C34score matrices bitwise. After shared aligned
rotation, each diagonal's maximum score change<=2.861023e-5; after shifted
rotation<=4.386902e-5. Both below the preregistered5e-5. Every diagonal rank
remains EXACT in both directions, and matched3R1is unchanged across conditions.
Thus the negative control can destroy cross-model compatibility while retaining
each original model's own ranking. This demonstrates why a crossed-tower loss
alone must not be treated as direct evidence of inferior semantic information.

All36saved score matrices receive independent official R1/R5/R10 validation
after completion; original singleton positives/asymmetric tie kernels retained.
Run hashes cover parent, code, protocol, manifests and maps; parent holds the
checked checkpoint/config hashes. No test data or held labels entered map fit.
Original PH-fitted checkpoints remain PH-fitted, NOT clean out-of-fold sources.

## Execution

AS-C35-GAUGE_run.json completed exit0 in25.027145seconds,
peak allocated GPU2,432,737,792bytes. No crash/retry/timeout or tolerance change.
60focused tests pass. One known-rotation recovery/scorer-invariance test added.
Original checkpoint/data files untouched; only4maps and36score matrices written.
Map fitting/extra TRAIN encoding and per-token512x512multiplications are added
costs; three-model encoder resources and nine score cells remain. This is not
a free single-model replacement for AS-C32's ensemble.

The conditional bootstrap uses315inferred source-prefix groups,10000draws,
seed20260915; descriptions name actual comparison arrays. Its uncertainty is
conditional on the fixed gallery/checkpoints and does not include model training,
dev checkpoint selection, anchor choice or exploratory-search multiplicity.

## Prior-art scope

Orthogonal coordinate fitting is standard, explicitly documented by
[SciPy's orthogonal Procrustes reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html).
A targeted primary abstract screen also found
[Multi-Way Representation Alignment, v2](https://arxiv.org/abs/2602.06205v2),
which describes shared orthogonal alignment, its limitations for retrieval,
and a geometry-corrected extension. This is a substantial novelty collision
warning against proposing generic shared-space or geometry-corrected merging.
Only abstract scope checked here; no full-paper implementation equivalence,
SLRet absence claim, human-read status or paper-result replication asserted.

## Fallacy scan:11/11 checked

1. Simpson: mixed-anchor cosines not individual-modality conclusions; both
   retrieval directions reported, including opposite changes versus identity.
2. Ecological: inferred prefixes not signer/recording identities; alignment
   agreement not linguistic feature equivalence.
3. Berkson: historical dev-selected checkpoints and prior-cycle dependence
   explicit; no globally representative population claim.
4. Collider: no error-conditioned fitting/selection or statistical adjustment.
5. Base rate:7096TRAIN/14192anchors,519DEV and315clusters stated; valid slots
   distinguished from SOT/EOT/padding in anchor construction.
6. Regression to mean: fixed maps/aggregates and unchanged identity control,
   no best map or favorable checkpoint selected.
7. Survivorship: all3conditions/27cells/9aggregates and maps retained.
8. Look-elsewhere: exploratory conditional comparisons, no significance claim
   about a novel method or post-hoc rescue of the failed primary improvement.
9. Forking paths: registered anchor/map/shift/precision/threshold unchanged;
   no separate video/text rotations to rescue this specification.
10. Correlation/causation: shared rotations demonstrate coordinate sensitivity;
    failed correction does not prove irreducible semantic co-adaptation.
11. Reverse causality: R1does not determine map fits, and agreement gain does
    not retroactively establish a useful retrieval mechanism.

## Search consequence / next layer

Keep AS-C32matched score ensemble as the explicit3-model strong control.
AS-C34 establishes pairing sensitivity; AS-C35 rejects this one orthogonal
explanation without establishing new novelty. Do not continue with anchor,
layer, CCA or alignment-weight sweeps after the failed fixed control.

Move to data/visual-input distinguishability: inventory exact deployed video
input collisions after official feature fusion/sampling/padding, and separate
within-TRAIN, within-DEV and cross-split identities. Earlier source-prefix
overlap was not actual tensor identity, and caption collisions were not video
collisions. Preregister before measurement; include valid masks in hashes and
confirm any collision with exact tensor equality. This is a bounded information
availability check, NOT changed positives, new benchmark or an information
ceiling inferred from near-neighbor similarity. No AS-C36 launched here.

ARS shaped the conservative control, frozen gates and explicit prior-art and
statistical limits. Goal active, no GO, no global exhaustion, no Proposal8.
