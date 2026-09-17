# AS-C18 — shared video-parameter gradient audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (chain-rule smoke and cache parity verified)
- Version Label: AS-C18-result-v1

## Execution and checks

Protocol preceded execution. Terminal `AS-C18-PARAMETER_run.json` reports
completed,37.47s, peak GPU20848029696bytes. On continuation the original session
handle had expired; no live probe remained and all6 final records were present.
No restart. Exit code was not separately recovered from the expired handle;
completion is supported by the final run record and expected outputs.
`AS-C18-SUMMARY.json` contains source hash, aggregation and11/11 fallacy checks.

No optimizer update, dev/test input or new retrieval metric.86,296,064 declared
visual parameters included, with inactive `conv2_trans.weight` explicitly zero.
Text/logit scale excluded. One R0 checkpoint, three fixed512-row train batches,
two text conditions; not six model replications. Every participating encoder
block matches cached tokens/masks exactly. The8-active-example direct/chain
smoke has loss1.937149534e-7, gradient norm2.576546639e-5, relative discrepancy0
and cosine1. All6 losses and valid-interface cosines match AS-C12 within the
preregistered tolerances. Native mixed precision, scale4096, CPUfloat32 gradient
accumulation/float64 scalar reduction; all returned gradients finite.

## Results

Projection means `<path,total>/||total||²`, NOT probability, fraction of harmful
learning, or additive squared-norm contribution.

|Batch/text|Valid-token cosine|All-token cosine|Parameter cosine|Duplicate-video projection|CLS/pad projection|
|---|---:|---:|---:|---:|---:|
|42 clean|.050566|.047700|.141677|.639071|.081008|
|42 augmented|.259028|.238058|.203030|.698085|.068564|
|1337 clean|.098557|.092442|.161729|.267953|.068171|
|1337 augmented|.155762|.196546|.116255|.156002|.048181|
|2026 clean|.055213|.054483|.139347|.817325|.363366|
|2026 augmented|.089742|.115636|.162134|.650115|.284098|

All6 video-parameter directional cosines positive: clean mean.147585, augmented
mean.160473. A global opposing-direction story is unsupported in this scope;
individual layers, text parameters, training stages and optimizer-preconditioned
updates remain unmeasured. Duplicate rows average2.6042% (range2.1484–3.125%).
Their valid-interface squared-norm mass averages88.2526% clean/77.9002% augmented
in these THREE batches, not AS-C12's12-batch sample. Parameter signed projection
is a different statistic; do not compare them as interchangeable percentages.
Duplicate/other pathway cosines are small positive.000888–.018213. Invalid-slot
projections vary.048181–.363366; omitted paths were not negligible in every batch.

Restricting a VIDEO ROW covector does not restrict duplicate QUERY losses: that
video participates in other queries' negatives. No duplicate-loss attribution,
harm, or authorization to remove/relabeled examples follows.

## Numerical and inference limits

Complementary vector is total minus selected pathway. Squared-norm closure is
an algebraic check, not an independent complementary backward pass. Synthetic
double tests verify chunked chain rule but not exact FP16 additivity. The direct
smoke uses8 active examples, not full512 end-to-end backward. No alternate scale
or float64 encoder validation; fixed scaling cannot prove every tiny derivative
survives. No actual Adam update, validation alignment or retrieval benefit was
measured. Full independent experiment replay not claimed.

Decision: no new supported conflict mechanism; pathway concentration varies and
does not establish harmful training. PMGR/RPCA remain closed. No new candidate
or Proposal8.31 focused tests passed in1.20s. Goal active, no global exhaustion.
