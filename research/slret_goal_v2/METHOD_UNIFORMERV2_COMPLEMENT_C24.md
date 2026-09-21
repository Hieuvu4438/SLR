# C24 prospective lead — temporally aligned UniFormerV2 RGB complement

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline planning
- Origin Date: 2026-09-21
- Verification Status: prerequisite extraction running; method not implemented or measured
- Version Label: c24_prospective_v0
- Authority: GoalV4; user authorization for public donors and pretrained weights

## Hypothesis and donor

In SEDS, pose-GCN adaptation provides the only replicated gain, while several
small modules change T2V and V2T in opposite directions. A plausible alternative
is that the native cached RGB representation lacks some temporally learned visual
evidence, rather than needing another graph/fusion adapter. The prospective claim
is therefore: under frozen native RGB+pose2D inference, a separately pretrained
video representation may supply complementary motion/context cues if its temporal
support is aligned before fusion.

The donor is the official [UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)
implementation for Li et al., ICCV2023
([paper](https://arxiv.org/abs/2211.09552)), pinned locally at commit
`722a43440fc5b9662cc2a8f23b86caa205e45ebc`. Code is MIT licensed. The public
checkpoint currently used is
`uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics400-rgb.pth`
(1,417,825,259 bytes); a separate weights license has not yet been identified,
so publication redistribution rights remain unresolved. These action-recognition
pretraining results do not establish SLRet benefit.

## Required interface and controls

The extractor produces frozen1024D descriptors for 32-frame, stride1 receptive
fields. Native SEDS RGB inputs are also1024D but are capped at64 tokens and use a
different temporal support; raw file substitution is invalid, and observed DEV
lengths already differ. Before a pilot, define deterministic receptive-field
alignment/downsampling to at most64 valid tokens, preserve padding/CLS semantics,
and test endpoint/short-video behavior.

C24 must retain native RGB and pose2D rather than replacing either by default.
A zero-initialized projection/residual route is allowed only after the data bridge
is exact and must be compared with a native-only matched control under the same
TRAIN subset, exposure, seed, selector and compute. If exploratory complement is
positive, a UniFormer replacement arm separates donor-backbone gain from additive
complementarity. No query-wise gate, score correction, TEST tuning or claim of
novel UniFormer architecture is allowed.

## Admission state

One authorized snapshot found TEST642/642, DEV519/519 and TRAIN3037/7096 paired
feature/meta/temporal outputs. Full TRAIN validation is pending. The remaining
allocated experiment time is429.629545913005s, below the measured566.883s cost of
the exact one-epoch full-TRAIN recipe, so neither a shortened pilot nor its matched
control is admitted. After extraction is terminal: validate counts, finiteness,
dimensions, receptive-field metadata and source IDs; then finalize the bridge,
cost estimate and experiment card before requesting/using adequate compute.
