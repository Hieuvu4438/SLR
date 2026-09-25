# Fixed four-input final-score sensitivity — pre-execution

2026-09-23, Cycle22. Continue Q39 without training, GPU, TEST or new labels.
Same four TRAIN inputs and seed42 checkpoint as Cycles20–21. This is not a
benchmark gallery and no recall/SOTA or semantic-negative claim is allowed.

Execute native Sign_Bert, CLIP pose/text and RGB towers, native fusion and
flip_similarity_softmax. Instantiate from recorded dimensions and strictly load
all four state prefixes. No network downloads or native training initialization.
CPUfloat64 for pose, CPUfloat32 for CLIP/fusion/scoring (native CLIP LayerNorm
casts to float32). This is a computational diagnostic, not deployedFP16 parity.
Use two CPU threads with a120s hard timeout for this bounded four-input job.

Native loader supplies pose, English text tokens and RGB features. Verify pose,
RGB, checkpoint and parent source hashes. Preserve original64 pose/RGB slots,
65-slot visual mask and32 text slots. At pose-window computation only, process
all valid windows plus one representative invalid slot, expanding that identical
eval-BN output back to every invalid position before the visual transformer.
This avoids redundant independent sign_conv work; no scoring tokens are removed.
Check native body values/window starts and preserve part coordinates exactly.

Only intervention: append8 zero-pose frames to all three parts before GCN;
retain the original valid windows/masks. Text and RGB are encoded once and reused.
Run both complete pose policies and both visual/scoring passes at fixed batch4.
Use native dual_mix0.5: mixed video×text score =0.5 I2T+0.5 T2I, separately
for fusion/pose/RGB. Keep the two channels too. Exact same-input score repeat
is mandatory; RGB score changes must be exactly zero.

Primary gate: fusion mixed-score max absolute change>1e-4 with all controls
passing establishes final-score sensitivity for these16 pairs. Below threshold:
NO_GO for this fixed score-level lead, not an information-absence theorem.
Also report row-centered and column-centered delta maxima to distinguish
changes from constant offsets, and strict pair-preference flips within the
four-item pool. These are descriptive, not accuracy or full-gallery rank gains.
No threshold/sampler/precision/checkpoint rescue based on outcomes.

Two independent command invocations must produce byte-identical JSON outputs;
report all source hashes, arrays, score channels, controls and execution errors.
Three structural tests first. The guide authorizes routine diagnostic execution
without ARS default per-command confirmation. Baseline source remains unchanged.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 120s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_padding_scores.py --output docs/codex_slret_research/evidence/padding_scores_20260923.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 120s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_padding_scores.py --output docs/codex_slret_research/evidence/padding_scores_20260923_replay.json
```
