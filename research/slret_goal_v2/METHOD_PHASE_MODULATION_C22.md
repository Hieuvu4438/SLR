# C22 — shared length-normalized phase modulation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline run
- Origin Date: 2026-09-21
- Verification Status: completed valid negative increment; exact recipe deferred
- Version Label: c22_pilot_v1
- Authority: GoalV4 and user authorization to pause/resume UniFormerV2 around jobs

## Evidence and registered hypothesis

Clean pose-GCN adaptation is the only current mechanism with repeated matched
gain, while C12's native clip-temporal fine-tuning tied that GCN and C21's explicit
canonical3D motion damaged V2T. This suggests testing a parameter-efficient
temporal calibration rather than another geometry/motion encoder.

RAP motivates low-rank temporal modulation to reduce video redundancy and reports
that temporal-only decomposition works better than additionally decomposing space
or layers ([Cao et al., arXiv 2024](https://arxiv.org/abs/2405.19465)). C22 borrows
that component-level idea, not its text-conditioned patch selection or reported
efficacy. Multi-scale and long-range temporal context are also established needs
in skeleton recognition ([Liu et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Disentangling_and_Unifying_Graph_Convolutions_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.html)),
but those action-recognition results are not SLRet evidence.

## Intervention and collision check

After native pose2D-GCN and RGB encoders, C22 learns a shared rank3 temporal phase
basis on32 anchors. For each variable-length sample the basis is linearly mapped
to normalized valid-clip phase, so short and long signs share endpoints rather
than raw absolute indices. Pose and RGB use separate rank3-to-512 scale and shift
factors but share the phase basis. CLS and padding are unchanged. Channel factors
start at zero, making inference exactly identity; native fusion/scorer/losses and
both original modalities remain intact.

This is not closed scalar pooling: no scalar clip weights or pooling change. It is
not C06: no adjacent difference, order loss or residual temporal convolution. It
is not reliability gating, text conditioning, teacher supervision, score repair,
new relevance or ensemble merging. Novelty remains unresolved; the specific
shared normalized-phase two-stream adaptation is the experimental contribution.

Four focused tests pass: identity/CLS/padding, normalized phase endpoints, stream
separation, two-stage gradient/state roundtrip, and registered optimizer groups.

## Pilot contract

Run `seds-phase-modulation-001` under job `v4-c22-phase-modulation-001`.
Release initialization; exact clean-GCN TRAIN7096/order, seed42, B32, one epoch/
222 updates; GCN1e-6, fusion1e-5, modulation1e-4; full DEV0/111/222; native
losses, selector and directional guardrails. Comparator is exact clean GCN
`seds-masked-control-002` selected78.03468208092485; C12 is an additional temporal
context and GCN-R1/global incumbent78.70905587668594 is the promotion target.

Estimate650s; hard1150s, outer1200s, allocator16GiB and watchdog19GB under user
20GB cap. Reserve1200s from remaining1608.46139310386s. The registered wrapper
validates exact UniFormerV2 PID/start/cmd, PID-scoped SIGSTOPs it, and always
SIGCONTs it after completion/failure/timeout. PROMOTE only on fused gain over the
matched clean GCN; otherwise DEFER without LR/rank/seed sweep.

## Terminal result and decision

The registered run completed222/222 without early stopping. Fused meanR1 at
steps0/111/222 was77.55298651252409/77.74566473988439/77.86386542166889.
Final/selected T2V was77.3076923076923 and V2T78.42003853564547. This is
+.31087890914480pp over release but -.17081665925596pp versus the exact matched
clean-GCN control and -.84519045501705pp versus the global incumbent. Both
directions are below the matched control. All identity, factor/phase update,
GCN/fusion update, frozen-buffer and exact batch-order gates passed.

Decision: DEFER this exact phase-modulation recipe without LR/rank/seed/horizon
sweep. Retain selected best.pt and full provenance. The redundant terminal
optimizer checkpoint was permanently removed after recording its size/checksum in
`storage-prune-c22-20260921-001`; it was not the selected artifact.
