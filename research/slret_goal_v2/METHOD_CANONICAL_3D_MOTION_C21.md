# C21 — anatomical-frame 3D articulation and motion branch

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline run
- Origin Date: 2026-09-21
- Verification Status: implementation/scoped-cache checks passed; retrieval efficacy unverified
- Version Label: c21_pilot_v1
- Authority: GoalV4 and user request to retain pose2D while adding H4W++ 3D

## Evidence, hypothesis and counterargument

C09 already tested raw H4W++ XY/XYZ coordinates and found no explicit depth gain.
Its own registered limitation is that camera-axis normalization does not correct
view rotation. For monocular estimated 3D, the same articulation can occupy very
different coordinates when the palm or signer rotates. C21 tests the narrower
hypothesis that 3D is useful only after expressing articulation in anatomical
frames and exposing local motion, not that another larger fusion module is needed.

Relevant prior art is component-level evidence only. View-Adaptive RNN identifies
view variation as a key 3D-skeleton problem and learns viewpoint regulation
([Zhang et al., ICCV2017](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.html)).
2s-AGCN reports value from joint plus bone direction/length streams
([Shi et al., CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html)).
A hierarchical temporal hand model separates short-term pose cues from longer
action aggregation ([Wen et al., CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wen_Hierarchical_Temporal_Transformer_for_3D_Hand_Pose_Estimation_and_Action_CVPR_2023_paper.html)).
These are action-recognition results, not SLRet evidence or novelty clearance.

Counterargument: deterministic local frames can amplify noisy H4W joints, discard
global orientation that carries signing meaning, and C09/C14 already show that
extra geometry is not automatically useful. Therefore C21 is one bounded input-
representation contrast, not a reopened generic 3D/bone family or module stack.

## Registered representation

Use the corrected retained H4W cache; no extraction/download. Each left/right
hand is wrist-centered, scaled by index-to-pinky MCP distance, and rotated into
a right-handed palm frame: index-to-pinky axis, projected wrist-to-middle-MCP
axis, and their cross product. Body joints are shoulder-center relative, scaled
by shoulder width, and rotated into shoulder/nose axes. Left/right streams remain
separate so handedness is not collapsed. Stable fallbacks cover degenerate frames.

After exact-center aggregation, first differences between adjacent valid clip
tokens are concatenated with canonical positions. Shared hand and body MLPs plus
the existing small temporal convolution produce a zero-initialized512-D residual
before native CGAF. Original pose2D, RGB and all native objectives remain active.
No confidence gate, text conditioning, teacher, score correction, graph topology,
new checkpoint or inference-time gallery state is introduced.

The transform is translation/scale/3D-rotation invariant by construction for
nondegenerate frames. Six pose3D tests plus four geometry-training tests pass,
including invariance, invalid padding, zero-init identity, gradients and compact
checkpoint reconstruction. A read-only check over16 real cache videos produced
144,942 finite canonical values and finite correctly shaped zero output.

## Prepared pilot and decision rule

Attempt001 failed before training because its legacy elementwise score gate
rejected backend floating differences even though every step0 retrieval metric
matched baseline exactly; it is not an efficacy result. Its46.37914991378784s
wall and artifacts remain recorded. Retry002 also failed pre-training because one
pose V2T MeanR shifted by exactly one rank over DEV519 while all primary metrics
were unchanged. Retry003 verifies zero output weights, exact R1/R5/R10/MedianR,
finite scores, and permits at most one aggregate MeanR rank. It passed full-DEV
initialization and reached16 updates under explicit user authorization:
`v4-c21-geometry-canonical-motion-003` ->
`seds-geometry-canonical-motion-003`. Release initialization, exact C09 TRAIN512
IDs/cache/order, seed42, B32, eight epochs/128 updates, native losses, branch1e-4
and fusion1e-5, DEV0/64/128, identical selector/guardrails/early-stop. Historical
raw-XYZ comparator `seds-geometry-xyz-001` selected77.84200385356455; matched
fusion-only subset control selected77.55298651252409. A positive lead first needs
to exceed raw XYZ and then the current78.70905587668594 global incumbent before
full-data extraction/replication is justified.

Launch command (executed through the pause/resume wrapper):

`/home/haipd/miniconda3/envs/seds/bin/python -u research/slret_goal_v2/tools/run_c21_with_pause_v4.py --launch`

Estimate450s only under normal throughput; hard1000/outer1050s,16GiB allocator,
19GB watchdog/user20GB cap. The exact concurrent UniFormerV2 extractor was
PID-scoped SIGSTOP'd without termination (verified `Tl+`) and is automatically SIGCONT'd by the
wrapper after C21 success, failure or timeout. Reserve1050s from remaining
1842.302428469923s; no assistant polling after the one startup check. PROMOTE only on useful fused gain;
DEFER on valid no-gain. At most one representation ablation after a positive lead,
not a blind axis/velocity/capacity sweep.

## Result and decision

Retry003 completed128/128, exit0, without early stop. All zero-output, geometry
update, compact roundtrip, frozen-buffer, data/order and checkpoint gates passed.
Mean fused R1 at0/64/128 was77.55298651252409/76.878612716763/
76.97495183044316. Selector therefore retained the inherited step0 checkpoint;
no best.pt was emitted. Relative to rawXYZ selected77.84200385356455, C21 is
-.28901734104046pp; relative to global incumbent78.70905587668594, it is
-1.15606936416185pp. Final T2V rose to77.64932562620423 but V2T fell to
76.30057803468208, so the trained branch created a harmful directional tradeoff.

Decision: DEFER canonical-motion3D. Do not run the optional representation
ablation, LR/capacity/seed sweep, or claim that monocular3D is generally useless.
The evidence rejects this registered anatomical-frame plus adjacent-motion recipe.
UniFormerV2 was automatically resumed and verified running. Only unselected
last.pt (221789958bytes) was deleted; metrics/logs/source are retained under
`storage-prune-c21-20260921-001`.
