# C07 — early anatomical interaction, not another late score correction

2026-09-19. Academic-research-suite, focused three-way scan + inline experiment
planning. User-authorized model improvement; no full baseline audit. Latest
user clarification explicitly encourages borrowing/combining strong papers,
including outside SLRet. Prior-art overlap is attribution context, NOT a ban.

## Evidence and choice

PH TRAIN7096 / DEV519, German sentence retrieval; release meanR1 77.552987,
provisional best77.938343. Current LoRA matched3seed effect is small; C03/C03b
and C06 do not demonstrate gains. These results do not prove which semantic
cue is missing. DEV is repeatedly exposed; TEST stays outside selection.

Actual SEDS source `modules/modeling_gcn.py:214` runs left/right/body GCNs
separately. Each reduces joint→part→whole articulator by max pooling before
concatenation. Native later convolutions CAN mix all three pooled streams;
do not claim SEDS has no articulator interaction. C03 operates after this
pooling; C06 after video encoding. C07 asks whether cross-part interaction
BEFORE joint identities are pooled yields a more discriminative representation.
Local adapted pose already supplies21+21+7 real keypoints per frame. No new
RGB extraction, gloss labels, new teacher or dataset download is necessary.
Hand coordinates are crop-normalized/mirrored; do NOT subtract them from body
coordinates as physical distances or claim this recovers contact geometry.

Intervention: at256-channel joint features just before native GCN part pooling,
use rank32 masked dot-product attention between different articulators; retain
anatomical node identity and pass modified nodes through the native pretrained
pooling/tail. New zero-output residual is inside the encoder, not a residual
retrieval score. Preserve native temporal windows, fusion/scorer/objective.
The added capability is source-joint-specific messages before max reduction.

## Rolling shortlist (not an execution queue)

| Hypothesis / model level | Priority and decisive comparison |
|---|---|
| C07 pre-pool joint exchange / spatial encoder | Selected: same tensors, changed access before a lossy reduction; cross vs within-part attention with exactly same parameters. |
| C08 masked-pose reconstruction / representation objective | Viable borrowed idea, not excluded for existing prior art. Needs an explicit supervision target and matched continuation, avoiding old teacher/protection combinations; secondary to directly testing joint interaction. |
| C09 stronger pretrained sign features / input backbone | Potentially larger information gain; dataset/language/protocol compatibility and extraction cost need scoped checks. Do not compare an extra-data system as same-data improvement. |
| C10 relation-aware video–text matcher / scorer | Potentially useful but current local-rival/score-residual proposals collide with closed regimes. Need a genuinely different score path, not renaming an old reranker. |

C07 is exploratory adaptation, not established novelty. If beneficial, combine
with C04 only AFTER isolated contribution; compare baseline, C07, C04, C07+C04.
Do not run all shortlist entries automatically. Stronger methods may be selected
later; there is no requirement to invent all primitives ourselves.

## Focused source record / WHY–HOW–WHAT

Search date2026-09-19; web search + original author/publisher pages, English
queries around sign retrieval, compositional/relational matching, cross-hand
attention, masked pose and multimodal pretraining. Include mechanisms that can
change the local intervention; exclude secondary summaries as claim support.
This is NOT a systematic review or proof of first-ever novelty. No human-read
claims; source pages read via browser, not an acquired local PDF corpus.

- [SEDS](https://arxiv.org/html/2407.16394v1), methods3.1–3.3:
  pose/RGB complementarity, local CGAF, fine-grained matching already exist.
  We retain them and intervene earlier than their clip-level fusion.
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903), original abstract:
  neighborhood-masked attention is established (ICLR2018). C07 borrows this
  principle with dot-product attention, not a faithful GAT reproduction.
- [C²RL](https://arxiv.org/html/2408.09949v1), III-B:
  contrastive learning plus autoregressive context supervision is existing SLRet
  prior art. Useful in principle, but don't reopen user-closed protected-context
  adaptation by renaming it or copy reported gains into this protocol.
- [Scaling up Multimodal Pre-training](https://arxiv.org/html/2408.08544v1),
  pretraining method: masked pose reconstruction plus sign-text contrastive
  learning. C08 borrows an objective direction; reproducing million-scale
  pretraining is not the next bounded local pilot.
- [Uni-Sign](https://arxiv.org/html/2501.15187v1), introduction/related work:
  large-scale generative pretraining, prior-guided RGB/pose fusion are existing
  directions; task/supervision differences matter. Not an apples-to-apples
  sentence-retrieval score reference here.

VSNet CVPR2025 was found by search but original PDF/HTML returned403; no verified
technical claim or exact novelty distinction against it. Other recognition,
dictionary-retrieval and preprint search hits are leads only, not validated
comparators. Full targeted prior-art comparison remains needed for a paper.

Registry comparison: no frozen-text weights, extra raw RGB stream, confidence
gate, teacher support, temporal warp/transport, decoder protection, gallery
graph or local rival scorer. Closest C03 jointly mixes already pooled512-d
articulator vectors; C07 exchanges256-d per-joint features before pooling.
This changes information available to interaction, not merely rank/LR/seed.
Cross-attention itself remains standard and is not the claimed contribution.

## Registered first experiment

Release checkpoint, seed42, B32, one PH TRAIN pass222updates; same native loss
(aux1), FP32 BertAdam moments. Train new joint module LR1e-4 + existing fusion
LR1e-5; all native encoder parameters/BN/dropout frozen. No C04 combination.
DEV0/111/222 full gallery with existing bidirectional R1/R5/R10 evaluator;
initialization eligible, earliest-tie tolerance1e-8pp, per-direction R1 guardrail
release−.5pp; incumbent77.938343 separately compared. Stop if mean falls >2pp.

Smoke2steps: zero-init fullDEV score parity (needed because native GCN forward
is split at a new point), finite loss/gradients, actual new-output updates,
no frozen-weight drift. Two focused CPU tests passed before GPU launch.
Smoke hard240s→pilot hard1800s, chain hard2100s; fail closed, no auto-retry.
Current GPU free48519MiB, disk78GiB, V2artifacts31GiB; remain within36GiB cap
and15GiB disk reserve. If smoke fails or OOM occurs, no efficacy conclusion.

Positive signal → matched within-articulator control, same module/parameter
count/LRs/batch order/horizon; fusion-only anchor alone cannot isolate C07.
If that contrast survives, three seeds then a protocol-compatible second
dataset DEV (CSL-Daily assets available; check required inputs only).
Record all directions, corrected vs newly broken query ranks and wall/VRAM.
No +2pp or SOTA claim from an exploratory DEV endpoint. No fixed novelty gate.
No blind multi-module combination, LR sweep or longer run simply because loss
falls. One motivated refinement may follow actual curves; negative signal
redirects the queue rather than repeating C06.

Training/extraction is background with logs; yield immediately after launch.
User will report completion. No quota-consuming polling loop.

## First result and registered repair

OriginalC07 completed222; selectedinitialization, finalmean77.167630. Source/config
inspection found fusionbaseLR actually1e-4, not1e-5 as this card intended:
native non-CLIP/non-sign optimizer groups inherit sign_lr. Raw result remains
valid for the actual recipe; it is not an efficacy test of the registered recipe.
New explicit group mapping joint1e-4/fusion1e-5, logged and asserted in smoke.
Three targeted CPU tests cover identity/masking/gradients, dependency control,
and actual distinct parameter updates from distinct LRs. Corrected profile
`joint-cross-lrfix`; fresh release seed42, same222updates and all other settings.
Hard2100s smoke→pilot. No claim LR repair itself is a scientific contribution.

User amendment: if C07 has no lead, prioritize their Hand4Whole++ Pose3D idea
over C08 masked reconstruction. Local source/checkpoints are available per user
at /home/haipd/DexAvatar/Hand4Whole-plus-plus_RELEASE. First establish output
convention, joint mapping, estimated-depth quality and local extraction costs;
then compare same detector XY vs XYZ to isolate depth from detector/domain effects.
Do not silently replace a known-good2D stream with an unvalidated3D pipeline.
