# C23 — frozen hierarchical ST-GCN bottleneck adapters

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline run
- Origin Date: 2026-09-21
- Verification Status: completed valid negative increment; exact recipe deferred
- Version Label: c23_pilot_v1
- Authority: GoalV4 and user authorization to pause/resume UniFormerV2 around jobs

## Evidence and hypothesis

The repeatable positive signal is adaptation of the native pose GCN, but C13/C14/
C15 topology, bone and covariance additions and C22 temporal modulation did not
beat their matched GCN controls. The remaining question is whether updating the
entire pretrained GCN is unnecessarily high-dimensional for one-epoch adaptation.
The CVPR2025 study [Are Spatial-Temporal Graph Convolution Networks for Human
Action Recognition Over-Parameterized?](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Are_Spatial-Temporal_Graph_Convolution_Networks_for_Human_Action_Recognition_Over-Parameterized_CVPR_2025_paper.html)
motivates testing compact adaptation in ST-GCNs; it does not establish retrieval
efficacy. C23 combines that question with zero-initialized residual adapters.

## Intervention and collision check

Freeze all native pose-GCN weights and buffers. After each of the three native and
two pooled graph blocks in each hand/body stack, add GroupNorm → 1x1 down-project
to rank16 → GELU → zero-initialized 1x1 up-project, then residual-add. The hand
stack remains shared by left/right hands; the body stack remains separate. Native
RGB, pose2D inputs, graph adjacency, fusion, scorer and loss are retained.

This differs causally from C11 visual-transformer LoRA, C13 adjacency learning,
C14 bone injection, C15 covariance pooling and C22 post-encoder temporal
modulation. It is a parameter-efficient adaptation test, not yet a novelty or
SOTA claim. Three focused tests verify exact identity, two-stage gradient/update,
adjacency preservation, state roundtrip, frozen backbone and LR partition.

## Pilot contract

Run `seds-graph-bottleneck-adapter-001` under job
`v4-c23-graph-bottleneck-adapter-001`. Release initialization; exact seed42
clean-GCN TRAIN7096 order, B32, one epoch/222 updates; adapter1e-4, fusion1e-5;
full DEV0/111/222; native loss, selector and directional guardrails. Comparator is
exact clean-GCN `seds-masked-control-002` at78.03468208092485; global incumbent is
78.70905587668594. Estimate650s; hard900s/outer950s, allocator16GiB and watchdog
19GB under the user20GB cap. Promote only on matched fused gain; otherwise defer
without rank/LR/seed sweep. UniFormerV2 is PID-validated, stopped only for this
job, and resumed in `finally` on every terminal path.

## Terminal result and decision

The registered run completed222/222 with no early stop. Fused meanR1 at
steps0/111/222 was77.55298651252409/77.64932562620424/77.84200385356455.
Final/selected T2V was76.6859344894027 and V2T78.9980732177264. Against the
exact clean-GCN control this is -.19267822736030pp mean, -.77071290944123pp
T2V and +.38535645472062pp V2T; it trails the global incumbent by
.86705202312139pp. Thus compact adapters learned and improved V2T but did not
preserve the bidirectional objective.

All zero-identity, output/upstream update, frozen native-GCN, fusion/buffer and
batch-order gates passed. Decision: DEFER the exact C23 recipe without rank/LR/
horizon/seed sweep. Retain selected best.pt and provenance. The redundant
optimizer-bearing last.pt was permanently removed only after recording its
size/checksum in `storage-prune-c23-20260921-001`.
