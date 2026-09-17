# UPRet real-TRAIN reduction sensitivity — locked protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent (inline run)
- Origin Date: 2026-09-15
- Verification Status: PLANNED, before execution
- Authority: autonomous-loop §41; preserve TEST and closed-family restrictions

Question: does replacing the native max reduction by the paper's sum appreciably
change actual encoder gradients, after a frozen amplitude control? This is a
prerequisite diagnostic, not an efficacy experiment, new method or SOTA claim.

Use the existing corrected Method1 UPRet architecture/config and its P14T-aware
feature regime, not the strong CiCo run's H2S-aware stream. Access only the
train-only PH manifest and UPRet TRAIN captions; never invoke Method1Dataset,
which loads its combined all-split manifests before filtering. Reuse its pure
sampling/tokenization/augmentation helpers. Hash all selected feature files.

Fixed selection: NumPy default_rng(20260916) permutes the7096 TRAIN records;
first128 unique indices make four disjoint32-example batches. Batch0 calibrates
one scalar; batches1–3 are the screen. No selection on losses or labels.
Model states: seed42 permitted CLIP initialization and existing
runs/method1/ph/base/seed42/best_dev.pt, labeled PARTIAL historical checkpoint.
Load its exact student state, record step/hash/source identity; never resume its
optimizer, claim completed reproduction, or modify the original checkpoint.

Compute one shared encoder graph per batch/state in training mode. Fix encoder
RNG20260915+batch index and native distribution RNG(seed42, step=batch index).
Extract the existing distribution function AST unchanged except its final return
also exposes the transport-weighted matrix. Both reductions share the exact same
samples, plan and graph. Preserve detached plan, logit scale placement, masks,
augmentation, inner temperature and native balanced four-cross-entropy loss.

Arms: native max (B0), full sum, frozen-scaled sum, no-OT context control.
For each state, calibrate positive c on batch0 only as the ratio of square roots
of the average row-centered and column-centered score energies (max/sum).
Freeze c for batches1–3; no label-fitting or DEV calibration. Report achieved
screen amplitude ratios as well; do not claim exact screen scale equivalence.

Measure B0 scalar parity with untouched baseline_loss_from_local (absolute
error<=1e-6), and selected visual/text parameter gradient parity (max error<=1e-6).
For each arm report loss, auxiliary score magnitude, all trainable encoder
gradient norms, cosines and ||g_arm-g_B0||/||g_B0|| separately for visual/text
encoders. PDE and score-head gradients are secondary. Check finite nonzero
encoder gradients and unchanged model state hash. Save every batch, not best only.

Sensitivity lead threshold (not retrieval gate): median relative difference of
scaled-sum versus max >=.01 in BOTH encoders at the partial state. Below threshold:
deprioritize this reduction as a major mechanism under the sampled regime;
do not infer universal uselessness or an information ceiling. Above threshold:
consider a matched deployment experiment only after baseline adequacy is resolved.
No gradient sign is interpreted as harmful/beneficial without actual deployment.

Execution: PYTHONPATH=methods/sssc:shared:. CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/haipd/miniconda3/bin/python -m methods.information_probe.upret_real_gradient
from repository root, FP32, CUDA, two CPU threads. Budget:1200s hard timeout,
progress printed after each batch/arm; monitor live execution handle. No training
updates, DEV evaluation, TEST, new assets, upstream/package edits or new positives.
Rerun, if used, gets a separate output and requires exact deterministic metrics,
excluding timing/memory metadata; original output remains immutable.
