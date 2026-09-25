# Native pose-output padding propagation — before execution

2026-09-23, Cycle21. Follow-up to the fixed Cycle20 body-GCN diagnostic.
No training, GPU, new labels, TEST, text/RGB encoders or retrieval scoring.

Use the same four lexicographically first recovered TRAIN IDs and the same
seed42 selected checkpoint. Verify the parent result's source hashes, current
checkpoint selection hash and each pose hash. Load the complete native
Sign_Bert state strictly, including both GCNs and the frozen GCN_Conv.
Native loader `_get_pose` constructs right/left/body inputs; verify its frame
list and window starts against metadata. This is not synthetic hand geometry.

All modules eval, one CPU thread, float64. As in Cycle20, remove only the two
adjacency `.cuda()` call sites for CPU construction. Execute native Sign_Bert
and native get_sign_output; its two empty_cache side effects are no-ops on CPU.
No tensor arithmetic or trained parameters are changed.

Compare solo extent L with zero extensions4 and8 for every pose part. Preserve
all valid coordinates, clip starts and masks. Use the native collator's zero
extension operation. Record pre-window concatenated features and actual native
sign_conv +16-frame mean outputs, not the previous descriptive means.

Bound cost using a mathematically independent subset of window slots: all native
windows intersecting the last4 valid frames, one first-window interior control,
and one invalid slot (-1 start, mask1). These four samples are all longer than
20 frames. GCN still encodes the entire input sequence. Native sign_conv treats
windows as independent batch items with BN eval, so selecting slots does not
change valid window arithmetic. Unit-test this subset/full-window correspondence
with a small native temporal block. This is not full-gallery/model evaluation.

Predeclared attribution gate: native pooled pose output changes >1e-6 in at
least one affected valid slot, while the interior slot and invalid slot remain
within1e-8 +1e-10 times solo maximum magnitude; extension4/8 must agree within
that tolerance. If true: GO_FOR_FINAL_SCORE_DIAGNOSTIC, not a method pilot.
No surviving output change: NO_GO for this downstream propagation test. Failed
hash, state, subset parity or unexpected-control checks are errors to explain,
not scientific positive/negative outcomes.

Also require the body slice of concatenated features to reproduce Cycle20's
per-sample max/relative changes within1e-8; this cross-check links the stages.
No sample substitution, policy/threshold/checkpoint sweep or inference about
video semantics. Four adjacent recording IDs cannot estimate corpus prevalence.

Three fixture checks before two deterministic independent invocations; output
files must be byte-identical. Each command has60s timeout and refuses overwrite.
The user guide authorizes routine diagnostic generation/execution without the
skill's default per-command confirmation. No baseline behavior is edited.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_pose_padding_propagation.py --output docs/codex_slret_research/evidence/pose_padding_propagation_20260923.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_pose_padding_propagation.py --output docs/codex_slret_research/evidence/pose_padding_propagation_20260923_replay.json
```
