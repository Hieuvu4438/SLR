# GCN temporal padding — fixed pre-execution protocol

2026-09-23, Cycle20. Computational consistency diagnostic, not a method pilot.
No new labels, TEST, training, retrieval scoring or GPU. User autonomous guide
authorizes bounded diagnostic code/execution despite ARS default confirmation.

Question: does changing only appended zero-pose extent alter valid-frame body
GCN outputs with fixed weights and eval BatchNorm? Native collators pad raw
pose to the batch maximum; native get_sign_output encodes this entire sequence
before gathering16-frame windows. The two temporal blocks have radius2 each.
This differs from score-slot masking, raw short-video repeat-last padding and
CiCo numerical tail-batch controls previously tested.

Fixed inputs: lexicographically first four IDs in recovered TRAIN labels,
selected without inspecting retrieval outcomes. Use recovered pose and recorded
retained-frame indices, with native body extraction and native window selection.
Verify pose SHA against metadata and checkpoint SHA against the seed42 selected
GCN report. Current source is an explicitly recorded execution contract; recovered
poses are not asserted byte-identical to historical deleted inputs.

Fixed model: body ST_GCN_Model weights from
artifacts/slret_goal_v2/seds-gcn-horizon3-001/best.pt, strict-load its complete
body submodule. CPU execution removes exactly two no-argument adjacency .cuda()
calls from the constructor via AST; no mathematical operations changed. Native
graph definitions are loaded unchanged. Cast model/inputs tofloat64 to separate
structural boundary effects from FP16/GPU kernel effects. This is not deployed
precision parity. Eval all modules; compare buffers before/after.

For each sample compare solo extent L with L+8 zero-pose frames, and L+4 with
L+8. Use native collator to verify that the appended values and original prefix
match this synthetic extension. Retain only original L outputs for comparison.
Record absolute and relative changes, changed-frame indices at1e-6 tolerance,
last4-frame differences, interior differences, and native16-frame window-mean
differences. Window means are a localization descriptor, NOT native sign_conv
or the final retrieval representation.

Predeclared decision: at least one sample with >1e-6 valid-output change AND
unchanged interior (absolute tolerance1e-8 +1e-10 times reference max magnitude)
AND L+4 versus L+8 agreement at that tolerance establishes the predicted local
boundary dependence, GO_FOR_ATTRIBUTION_ONLY. Zero signal is NO_GO for this
four-sample body-path test, not a universal independence proof. Unexpected
interior/extension behavior requires explanation, not an efficacy conclusion.
Any hash/shape/state failure is an execution error, not a scientific NO_GO.

Two independent invocations, CPU threads1, timeout60s, distinct output JSONs;
require byte-identical reports excluding no fields (no timing field). Existing
checkpoint selected on DEV is fixed, but only TRAIN inputs are inspected here.
The four neighboring recording IDs are not a representative independent sample.
No prevalence, effect-on-R1, causality-of-historical-errors or novelty claim.
Even a positive signal does not justify a normalization/padding rescue sweep.

Commands from /home/haipd/SLR:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_gcn_temporal_padding.py --output docs/codex_slret_research/evidence/gcn_temporal_padding_20260923.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_gcn_temporal_padding.py --output docs/codex_slret_research/evidence/gcn_temporal_padding_20260923_replay.json
```
