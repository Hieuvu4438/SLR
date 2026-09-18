# Scoring-cost diagnostic, preregistered 2026-09-18

Status: diagnostic/engineering control, NOT an admitted research candidate.
Accuracy remains the primary goal. One failed continuation does not prove that
accuracy cannot improve. This parallel check determines whether the alternative
accuracy–efficiency route has a measurable bottleneck worth investigating.

SOURCE_VERIFIED: SEDS PH evaluator calls video-only fusion inside every video–text
block pair and returns three streams; deployment of the fused model needs only
the fused score. CiCo and SEDS also recompute the same similarity tensor for the
two directions at inference (no augmented text). These are ordinary engineering
controls, not novelty. Preserve inner softmax, masks, outer averaging, temperature,
checkpoint and native tie policy. No inference precision change or scorer repair.

Run one full adapted PH DEV519 encode with native SEDS release checkpoint; capture
all native matrices and cache unchanged normalized fusion/text tokens plus masks.
Compare native score stage with (i) fusion cached once per video batch, all streams;
(ii) fused-only cached exact scorer, block32; (iii) same block128. One warmup and
five synchronized wall-time measurements for each optimized scorer. Report fusion
precompute and encoding separately; raw feature preprocessing remains unchanged
and costs5947.341s TRAIN/390.556s DEV, not erased from end-to-end accounting.

All optimized matrices require maxabs<=1e-4 and exact primary query-rank parity
against native. Full-matrix parity is stronger than merely unchanged R1. Any
failure blocks deployment claims and requires identifying the discrepancy.
This is not a proposal to mask padding differently or change temporal pooling.
Save timing distributions, normalized cache, source/asset hashes, and scores;
reserve256MiB under the existing15GiB cap;<=300s GPU hardtimeout charged to
discovery/control reserve. No optimizer, TRAIN fitting, TEST, or corpus deletion.

Decision: only investigate a new score-preserving acceleration if the optimized
dense fused-only full-gallery stage still exceeds100ms on this hardware and if
a mathematical bound can plausibly avoid computation. Otherwise the apparent
bottleneck is sufficiently explained by routine caching for this benchmark;
do not build a novel-method story around the unoptimized native evaluator.
Before a research card: derive/check bounds and screen strongest relevant prior
art (PLAID, WARP, lossless ColBERT pruning, EigenLI). Exact result preservation is
different from changing pooling, confidence gates, graph calibration or lexical
support. Approximate pruning is NOT silently admitted. An observed speedup over
native alone is neither a paper contribution nor validation of the goal.
