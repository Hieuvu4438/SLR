## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-23
- Verification Status: VERIFIED for final-version replay; cross-version auxiliary equality is PARTIAL
- Version Label: pose_padding_propagation_v3
- Repository HEAD: de2f07ecac9ce65563316fde0860f22e4d585acf

# Padding dependence survives native three-part pose window processing

[M] On the same four fixed TRAIN inputs, padding-induced differences survive
the actual native right-hand/left-hand/body GCN concatenation, sign_conv and
16-frame mean. All13 selected boundary windows change; all four first-window
controls and all four invalid-slot controls remain exactly unchanged. Extension4
and extension8 produce exactly equal selected native pose outputs.

| TRAIN ID suffix | Changed boundary windows | Maximum absolute pooled change | Relative L2 over selected valid outputs |
|---|---:|---:|---:|
| 6694 | 4/4 | 0.185278 | 2.9061% |
| 6695 | 3/3 | 0.132484 | 2.4615% |
| 6696 | 4/4 | 0.271090 | 5.0220% |
| 6700 | 2/2 | 0.213865 | 4.2861% |

All IDs use prefix `01April_2010_Thursday_heute-`. These percentages use only
the registered selected valid windows, not all64 slots, the whole frame sequence
or a retrieval score. Do not compare them directly with Cycle20's whole-body-
sequence percentages as an attenuation estimate.

The original propagation gate is met: **GO_FOR_FINAL_SCORE_DIAGNOSTIC**.
This is not SUPPORTED-FOR-PILOT and not evidence of retrieval improvement.

## Executed contract

[Protocol](POSE_PADDING_PROPAGATION_PROTOCOL.md) fixed inputs, checkpoint,
intervention and thresholds before execution. Native `_get_pose` supplies all
three parts from recovered TRAIN pose files. Frame lists and window starts
match recovery metadata. The complete Sign_Bert checkpoint state loads strictly,
including GCN_Conv; all modules remain eval and all buffers remain unchanged.

Native get_sign_output executes unchanged arithmetic. CPU adapters omit only
two adjacency `.cuda()` call sites and replace its two empty_cache calls with
no-ops. RGB is an unused placeholder returned unchanged by this function:
neither an RGB encoder nor fusion is executed. The selected window subset
contains all boundary-affected windows, an interior control, and an invalid slot.
GCN still encodes the entire sequence. Eval BN and independent window processing
permit this subset; a small native temporal-block test matches full-window and
subset outputs to1e-12. This is not a full-model or full-gallery test.

The body slice cross-check agrees with Cycle20's maxima and relative changes
within the original1e-8 tolerance. Concatenated pre-window relative L2 changes
are8.5390%,8.1207%,16.6538%,6.2394%, respectively; these are separate statistics
from the selected pooled-output changes above.

## Execution history and reproducibility qualification

Three initial fixtures pass. Four diagnostic invocations each exit0 before60s:

1. Initial [v1 report](pose_padding_propagation_20260923.json) found the effect
   but captured only body features, missing the requested concatenation summary.
   Its exact script is preserved as [v1 source](pose_padding_propagation_v1_source.txt).
2. [Instrumentation amendment](POSE_PADDING_PROPAGATION_AMENDMENT.md) adds a
   transparent capture around native gcn_emb, without changing its returned
   tensors or downstream arithmetic. [v2 report](pose_padding_propagation_20260923_v2.json)
   and [v2 source](pose_padding_propagation_v2_source.txt) are retained.
3. v3 materializes the body slice contiguously to better match the original
   reduction layout. This does **not** achieve exact equality for every v1
   scalar: three auxiliary `body_parent_relative` values differ by
   1.39e-17,5.55e-17,2.78e-17. The amendment's stronger cross-version exact-field
   check therefore FAILS; do not silently promote it to a pass. No further
   layout/precision rescue is performed. All original primary pooled-output,
   control, saturation and changed-window fields match exactly.
4. Two independently invoked final v3 executions give byte-identical reports.

The residual auxiliary differences arise across changed instrumentation/reduction
layout; their exact floating-point cause was not isolated. They are far inside
the originally registered body cross-check tolerance and do not change its
decision. This is a qualified result, not perfect cross-version reproduction.

Final commands, cwd /home/haipd/SLR:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_pose_padding_propagation.py --output docs/codex_slret_research/evidence/pose_padding_propagation_20260923_v3.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_pose_padding_propagation.py --output docs/codex_slret_research/evidence/pose_padding_propagation_20260923_v3_replay.json
```

[Final report](pose_padding_propagation_20260923_v3.json) and
[replay](pose_padding_propagation_20260923_v3_replay.json):8112bytes each, SHA256
`b65106827c767d8771fc693f4c3bf712e6ce71c3c9f955acfd6fb0b41ee8bf1c`.
All23 source/input hashes rechecked. One CPU thread, float64,
torch2.11.0+cu128 installed; GPU unused. No deployed-precision parity claimed.
Initial source archive hash matches its report; old files were not overwritten.

## Validation cautions —11/11 checked

1. Simpson: per-input results shown; no subgroup aggregate comparison.
2. Ecological: numerical output changes are not linguistic judgments.
3. Berkson: initial IDs were fixed without error labels, but all share a
   recording prefix. No claim of representative coverage.
4. Collider: no relevance/outcome-conditioned sampling or adjustment.
5. Base rate:13/13 selected boundary windows is not a corpus error rate;
   selected-window and full-sequence denominators remain distinct.
6. Regression to mean: no training, selected-error repair or efficacy comparison.
7. Survivorship: all four fixed inputs finish; no replacements or removals.
8. Look elsewhere: no sample, checkpoint or intervention-strength sweep.
9. Forking paths: instrumentation amendment and failed exact cross-version check
   are disclosed; thresholds and primary outputs remain unchanged.
10. Causality: controlled padding extent changes native pose outputs, but neither
    relevance nor ranking harm was measured.
11. Reverse causality: input extent is manipulated before computation; no
    observational inference from retrieval errors is made.

## Consequence for the research goal

The downstream-pose-erasure alternative is rejected on these fixed inputs.
Next test must use final visual encoders/fusion/scoring with fixed weights and
inputs; a representation change alone cannot establish changed preferences.
Use a bounded score-sensitivity check before any full-gallery measurement or
correction. Keep all text/RGB inputs and official relevance rules fixed. If
final scores remain stable, stop this particular deployed-score lead. If scores
change, separately test rank sensitivity; score change is not retrieval harm.

No normalization/padding-policy sweep, automatic correction, new primary method,
SOTA or semantic claim is admitted. Ordinary consistency repairs would remain
engineering controls, not novelty. ARS supplied preregistration, replay and
transparent failure accounting. AI-assisted code/source analysis; no human/AI
semantic annotation, training, GPU, TEST, external contact or deferred-job check.
The original research objective remains active and incomplete.
