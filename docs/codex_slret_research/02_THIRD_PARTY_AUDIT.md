# Bounded baseline audit

2026-09-22. Purpose: identify relevant intervention points and fair controls.
No exact-paper reproduction campaign is required or launched.

## CiCo: inspected local scoring path

Root: `third_party/SLRT/CiCo/CLCL`. `third_party/README.md` and
`configs/ph_base.yaml` specify SLRT commit
`38a4f7b00da7a858d59b7fabe5093876a84db8e0`; current full ancestry/diff validation
is pending, so the declared pin is not a fresh pristine-checkout certificate.

`modules/modeling.py::{get_visual_output,get_sequence_output,
flip_similarity_softmax,forward}` implement normalized contextual video/text
token interactions. The shared bridge takes [B,F,1024] video features and maps
to upstream [B,1024,F,1]. Let normalized tokens be v_if and t_jw, and
a_ijfw = v_if dot t_jw. The two returned [video,text] score channels are:

    I_ij = exp(logit_scale) * mean_valid_f sum_w softmax_w(a_ijfw / .07) a_ijfw
    T_ij = exp(logit_scale) * mean_valid_w sum_f softmax_f(a'_ijfw / .07) a'_ijfw

Here a' uses augmented text during training. The outer average applies masks;
native inner softmax is not generally masked before reduction. Preserve this
fact in parity comparisons. With `mix_design=balance`, d=`dual_mix`, and CE
meaning row cross-entropy against the batch diagonal:

    L = .5 * [d CE(I) + (1-d) CE(I.T) + d CE(T.T) + (1-d) CE(T)]

No gloss labels enter this inspected path. Batches define negatives. Local
`ph_base.yaml` uses English captions, BSL5K agnostic plus How2Sign-transfer-aware
features, 64 video slots, 32 text slots, d=.5, AdamW, B512, and DEV mean R1
selection. Those settings must not be attributed to every native CiCo run.
Scoring costs O(Bv*Bt*F*W*D), with O(Bv*Bt*F*W) pair-token storage before
blocking. Native and project continuation optimizer recipes are separate.

## SEDS: inspected local path

`dataloader_ph_retrieval_train_pose.py` → `get_sign_output` →
`get_visual_output` → `Gloss_Fusion_Transformer` →
`flip_similarity_softmax` → `forward`.

`get_sign_output` computes hand/body GCN frame representations, forms windows
using `clips_start`, applies `sign_conv`, and averages each window. RGB comes
from cached 1024-dimensional features. Contextual visual tokens have width
512; masks include CLS and use zero for valid slots. Pose/RGB/fusion each use
CiCo-style directional token interactions. Native `forward` adds fusion,
pose and RGB retrieval losses plus enabled optional matching/KL terms, unless
`freeze_exfusion` selects the fused loss alone. Fusion's name `gloss_atten`
does **not** demonstrate gloss-label supervision; inspected forward has no CTC.

The C27 attachment sees [B,F+1,512] contextual pose tokens and a [B,F+1] mask.
Its classifier has 512*(G+1)+(G+1) parameters (G=1085 implies 557,118), predicts
G gloss classes plus blank, and currently appends an eighth returned loss to
the native seven. A trainer must explicitly weight that term; attachment alone
does not prove it changes retrieval training.

`_get_pose_clips` retains sorted window starts. This supports chronology of
the indexed windows, but does not prove every gloss remains visually recoverable
after frame filtering/window compression. Per-example CTC feasibility must be
checked on actual loaded masks, including repeated labels.

## Remaining baselines and evidence boundaries

Cycle3 completes the targeted mechanism pass below. Exact paths, shapes,
objectives, selection semantics, complexity, source hashes and primary-paper
limits are in the [implementation report](evidence/BASELINE_MECHANISM_AUDIT.md).
Current git queries in all five vendor directories resolve to the parent SLR
repository, not independent upstream checkouts. Historical pins therefore need
source/hash corroboration; they are not current pristine-checkout certificates.

| Baseline | Current local evidence | Next necessary check |
|---|---|---|
| UPRet | Current scorer, training-only sampling/transport, balanced CE and native test selection traced. Partial step767 is not paper strength; reduction-repair lead already closed. | Only use as a quantitative comparator after matching resources and validating a complete trained checkpoint; no repair campaign. |
| C²RL | Primary content/translation pretraining and separate downstream retrieval encoders verified; derivative-source/task gates read fully. | Keep official retrieval reproduction unverified; do not repeat completed SLT-proxy/availability detours. |
| SAN | Current feature/text encoders, negative-table consumption, asymmetric hard loss and test-loader selector traced; primary stress/standard results separated. | Any future comparator needs actual table provenance and explicit DEV adapter; no invented miner. |
| CMCM | Four full component files read; source integration gaps and prior certificates reconciled. | No runnable matched comparator established; do not build a replacement solely for paper parity. |
| Uni-Sign / UniFormerV2 | Local donor code present; C26 weak result, C24 deferred. | Preserve donor supervision/resource distinction; do not restart deferred work. |
| GFSLT / SignCL / SignCLIP / SHuBERT | No dedicated top-level checkout found by this census. | Whole-repo equivalent-code search and paper evidence pending; no claim of global absence. |

The six mandatory named mechanisms now have bounded implementation/paper
coverage sufficient for collision screening, not final resource-equivalent SOTA
certification. Wider donor inventory and complete ancestry remain limited.
Historical detail remains in
[`BASELINE_AUDIT.md`](../../research/slret_goal/BASELINE_AUDIT.md) and
[proposal7 implementation audit](../proposal7/Implementation_Audit.md).
