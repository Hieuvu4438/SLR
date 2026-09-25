## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-23
- Verification Status: VERIFIED for fixed CPU diagnostic and exact replay
- Version Label: gcn_temporal_padding_v1

# Valid body-GCN outputs depend on appended padding extent

[M] All four preselected TRAIN samples show the registered boundary signature:
appending zero-pose frames changes the last four valid frame representations
with the model entirely in eval. Earlier valid frames remain unchanged within
1.34e-15, and extending by4 versus8 frames yields exactly equal valid outputs.
Decision: **GO_FOR_ATTRIBUTION_ONLY**, not SUPPORTED-FOR-PILOT.

| TRAIN ID suffix | Valid frames | Maximum absolute change | Relative L2 change | Largest window-mean change |
|---|---:|---:|---:|---:|
| 6694 | 53 | 2.258310 | 7.6706% | 0.205683 |
| 6695 | 90 | 2.304791 | 8.3725% | 0.243980 |
| 6696 | 28 | 4.369314 | 18.2585% | 0.552419 |
| 6700 | 164 | 2.649757 | 6.5671% | 0.299371 |

All IDs have prefix `01April_2010_Thursday_heute-`. Relative L2 is
norm(padded_valid − solo)/norm(solo) over the complete valid body-GCN array,
not retrieval loss, rank change or error rate. Window means are descriptive
averages; the subsequent native sign_conv is not measured by this statistic.

## Mechanism and distinction from previous tests

[V] Native PH TRAIN and evaluation collators pad pose sequences to the batch
maximum with zero coordinates. `modeling.py:get_sign_output` first executes
GCN on that whole sequence, then selects the valid16-frame windows. The GCN
body model applies per-frame spatial processing followed by two temporal
blocks of radius2. Zero coordinates are not a mask carried through those
blocks. External sequence boundaries and additional processed zero frames
therefore need not supply equivalent convolution boundary values.

[I] Last-four-frame localization and4-versus8 saturation agree with the finite
receptive field. This intervention establishes padding-extent dependence at
this body-path output, not the origin of any historical retrieval error.
Frozen BN removes batch-statistics coupling but not temporal boundary effects.
The previous CiCo tail-batch check concerned numerical batch shape, and Q05
concerned score pooling masks. Neither tested this GCN pre-window pathway.

## Execution and reproducibility

The [protocol](GCN_TEMPORAL_PADDING_PROTOCOL.md) was saved before model-output
inspection. Three new fixtures passed before both executions: CPU AST transform
scope, native collator/window behavior, and native body-model shape/BN policy.
Both exact protocol commands exited0 within60s. No diagnostic code correction
or execution anomaly occurred during these two runs.

[First report](gcn_temporal_padding_20260923.json) and
[independent replay](gcn_temporal_padding_20260923_replay.json) are byte-identical,
8183bytes each, SHA256
`e5d6f2944ca22f0a3b16f95b8edb83f81b1f399cadcd887d43898d7a3be740c1`.
All17 source/input hashes were checked again. The fixed selected checkpoint
matches its selection SHA; body state loads strictly, with no missing or
unexpected keys. All BN and other buffers remain exactly unchanged.

The native body class is AST-loaded with only two adjacency `.cuda()` calls
removed for CPU placement. Graph definitions are unchanged. Execution is
float64, one CPU thread, torch2.11.0+cu128; no GPU or deployed-precision parity
is claimed. Pose files match recovery metadata hashes. Recorded retained
indices supply the frame set; native body extraction and window selection run
directly. These are recovered resources, not historical deleted pose bytes.
No raw videos are decoded. No human-read attestation is implied.

## Validation cautions —11/11 checked

1. Simpson: report each sample; no pooled group or signer comparison.
2. Ecological: intermediate-array changes are not video-meaning differences.
3. Berkson: lexicographic IDs avoid retrieval-label selection, but all four
   share a recording prefix and are not representative samples.
4. Collider: no outcome-conditioned subset or adjustment.
5. Base rate:4/4 cannot estimate prevalence across7096 TRAIN or519 DEV.
6. Regression to mean: no error-selected sample, optimization or before/after
   retrieval improvement.
7. Survivorship: all four selected inputs complete; none removed or replaced.
8. Look elsewhere: body path and extensions were registered; no norm,
   padding-length, checkpoint or favorable-example sweep.
9. Forking paths: gate unchanged; replay is not evidence from new subjects
   or independent training seeds.
10. Causality: padding intervention establishes an operator effect, not
    retrieval harm or semantic information loss.
11. Reverse causality: controlled extent change precedes computation;
    no relationship between ranking errors and padding is estimated.

## Decision and next separating test

The consistency failure merits downstream attribution, not a proposed paper
contribution. Carry the same fixed intervention through actual three-part
pose concatenation and native sign_conv; only if it survives, measure fixed-
weight final representations/scores under a predeclared batching comparison.
Keep inputs, checkpoint, gallery and evaluator fixed. No padding optimization,
changed positives or TEST tuning.

If downstream computation eliminates the difference, stop this retrieval-
bottleneck lead. If differences survive but ranks are stable, record bounded
representation sensitivity without efficacy. If ranks change, establish a
batch-independent comparison contract before interpreting candidate gains.
An ordinary implementation correction is not a novel SLRet method. No candidate
has passed novelty/local-collision or retrieval-gain gates here.

ARS supplied pre-execution scope, replay and inference limits. AI-assisted
diagnostic code/source review was used; no human or AI semantic labels.
No training, TEST, GPU, deferred-job polling or external contact. Main method
and SOTA objectives remain active and incomplete.
