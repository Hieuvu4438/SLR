# Rolling model-improvement queue

LATEST C23 completed222: selected/final77.842004 versus exact clean-GCN
78.034682 (-.192678pp). V2T+.385356pp but T2V-.770713pp; all identity/update/
freeze gates passed. DEFER exact frozen hierarchical graph-adapter recipe without
sweep. C22 phase modulation was also below its exact control by.170817pp.

WAITING prerequisite for prospective C24: frozen UniFormerV2-L/14@336 extraction
is running; observed TEST642/642, DEV519/519 and TRAIN3037/7096 complete. C24's
working hypothesis is representation complementarity, not another pose-GCN
adapter: preserve native RGB+pose2D and add temporally aligned donor RGB evidence.
Required controls are native-only under the same subset/exposure and, if positive,
donor replacement versus additive complement. The donor's stride1 32-frame
sequence often exceeds SEDS'64-token interface, so deterministic receptive-field
alignment/downsampling is part of the prospective contract. No training is
admitted before full TRAIN validation; remaining compute429.629546s is below the
measured one-epoch full-TRAIN cost. No C24 efficacy or novelty claim yet.

LATEST C19-R1 retry002 completed160: selected/final78.227360 versus matched
control78.323699 (-.096339pp). All direction-specific activation/gradient/update
gates passed. The preceding attempt001 was only a repaired technical failure.
C19 bidirectional recovery completed160: mean78.323699, an exact matched-
control tie. T2V gains.192678pp while V2T loses.192678pp. No promotion and no
within-stream capacity control. DEFER global-exchange family without LR/rank/slot
sweeps. Budget remaining4145.514179s, no active job/reservation, user20GB.

C20 PARKED incomplete: hard timeout at146/666. Step111 fused77.842004 versus
same-step GCN-R1 control77.938343 (-.096339pp). Activation/update gates passed,
but probability never annealed to0 and no exact optimizer/RNG resume exists. Do
not claim completed rejection, but do not spend remaining budget replaying an
already-negative common-step signal. See METHOD_DOMINANT_STREAM_DROPOUT_C20.md.

NEXT PREPARED C21: reuse corrected H4W cache but replace C09 camera-axis XYZ with
translation/scale/view-invariant palm/body frames plus adjacent-clip local motion;
pose2D and RGB remain. This directly targets C09's recorded rotation weakness and
does not repeat rawXYZ, C14's native2D graph-bone branch or C13 topology learning.
Ten scoped tests and16-video real-cache finite check pass. Do not launch while
unowned GPU utilization is100% at the sole observation; no polling/interference.
Remaining1842.302428s, prospective hard1000s. See METHOD_CANONICAL_3D_MOTION_C21.md.

LATEST decision2026-09-20: C18 B64 completed80, selected78.227360/final78.131021
vsB32control78.323699. DEFER current B64 recipe; C17 remains USER-CLOSED.
CURRENT C19-R1 directed bottleneck adapter before native CGAF. Native
RGB+pose2D/GCN/loss remain, B32/160 matched subset. See METHOD_GLOBAL_EXCHANGE_C19.md.
Prospective shortlist after it: (1) stronger hand/3D input representation,
requiring new hypothesis beyond C09's tested XY/XYZ branch; (2) true GradCache
scaling deferred after C18, not assumed useful; (3) C10 matcher unresolved because
local-rival/score-correction designs collide with closures. No adaptive GPU queue.
Historical entries below superseded.

All entries are hypotheses, not measured gains; novelty unresolved. No TEST
scores/ranks consulted. Priority is current, not a permanent closed shortlist.

LATEST: C12 native clip-local temporal convolution adaptation with GCN deferred
after78.034682, an exact primary-metric tie with matched GCN seed42;
see [METHOD_GCN_CLIP_TEMPORAL_C12.md](METHOD_GCN_CLIP_TEMPORAL_C12.md).
CURRENT: GCN-R1 selected mean78.612717 across3seeds (SD.096339), best1337
78.709056. All selected beat1epoch, but two final666 endpoints are worse than
their1epoch endpoints. GCN-R2 freeze-after222 deferred: selected78.420039,
no increment; training completed but exact pre-freeze comparison failed. Keep
this attribution caveat; see METHOD_GCN_FREEZE_R2.md. No automatic retry/sweep.
Matchedcontrol completed78.323699, beats C16pointwise by.289017pp; no promotion.
C16-R1 completed78.227360 vs noaux78.323699 (-.096339pp); no promotion.
C16-R2 complete78.323699, tie noaux: DEFER direct SignRep pointwise/RKD-D recipes
in this tranche. No more loss-strength/LR/horizon sweep; no collapse diagnosis.
C17 pureDCL completed: selected77.649326/final77.456647, below noaux78.323699.
Clipped160/160 vs control57/160; total gradient median9.314 vs.528. No promotion.
C17-R1 USER-CLOSED after timeout in final evaluation. Training checkpoint160
present, finalDEV missing; intermediate80mean77.649326, no promotion. User says
skip C17/change direction: no evaluation recovery or further DCL experiments.
CURRENT C18 native batch64: change number of in-batch negatives without DCL,
teacher or scorer changes. Same TRAIN512/tenepochs/5120exposures;80updates vs160.
See METHOD_NATIVE_BATCH64_C18.md; engineering baseline headroom, not novel method.
Keep pretrainedhand/3D and true gradient-cache scaling leads queued, not launched.
C18-001 failed at nvidia-smi5s timeout, no saved train update/checkpoint; efficacy
unmeasured. Single infrastructure retry002 keeps recipe, GPU query deadline20s
and failclosed monitoring. Budgetavailable11083.931396s; reserve2450s, max20GB VRAM.
C16 complete160steps:78.034682, +.481696ppinit/-.674374ppbest; no promotion.
Matchednoaux TRAIN512/offload/mathSDPA control now collected; directpointwiserecipe notscaled.
Cachecomplete1031videos/60023clips. Match TRAIN512/160updates, DEV80/160;
native RGB/pose2D inference retained, no genericstreamstack or scoreanchor.
C15-R1complete666 best78.612717, tieR1seed42, -.096339ppglobalbest; deferpoolingfamily.
Native per-channel max can discard joint coactivation; add compact second-order
descriptor before anatomical pooling, retain native max/GCN/RGB/pose2D. Borrowed
bilinear idea, novelty/gain unresolved. C13/C14closed for further tuning.
C15 completed666best78.420039 belowR1seed42by.192678pp/globalbestby.289017pp.
One structural contrast: remove within-part projected mean before moment pooling,
same capacity/LR/horizon. If no increment, deprioritizefamily, no blind sweeps.
User explicitly permits old/recent paper ideas and developing new contributions
from real weaknesses; prioritize measured improvement, do not force novelty claims.
C13 completed666; best77.745665, final77.552987; no increment over GCN-R1.
SlowergraphLR1e-5 completed666 best78.131021 but belowR1seed42 by.481696pp;
defer C13topology family; no extraLR/horizon/seed sweep. C14 explicitbone
descriptors completed666, best78.612717 tiesR1seed42; no promotion.
C14-R1 boneprojectionLR1e-4 completed666 best78.131021, worseby.481696pp;
defercurrentC14family afteronecontrast, no extraLR/horizon/seed sweep.
no full baseline replay, R2 retry or C12 refinement. Keep all GCN-R1 leads.
Clean GCN three-seed lead78.227360mean (best78.516378) retained; matched
fusion-only ablation completed, pairedmeanGCNgain+.642261pp. C11 composition deferred
after77.938343 vs GCN78.034682; no primary-metric increment. C08 masking/reconstruction
deferred after77.745665 vs clean GCN control78.034682 (new provisionalbest).
C07 and current C09
additive geometry are deferred after below-incumbent pilots. C09 corrected cache
and compact weights retained; user-suggested 3D is not universally ruled out.
Rolling hypotheses, ordered: (1) C17 fused-DCL optimization pilot; (2) larger
native contrastive batches via gradient caching, preserves positives/no hard
mining, implementation and matched update/exposure controls needed; (3) rotation/
depth-specific3D encoding using retained H4W cache, must differ from failedXY/XYZ
residual; (4) compatible pretrainedhand-shape representation, needs concrete
checkpoint/layout/cost and a non-generic integration unlike closed C16transfer;
(5) relation-aware token matcher distinct from closed rival-local/scorer-residual
designs, not yet admitted. Literature/donor status in LITERATURE.md/DONORS.md.
Unimplemented hypotheses are not an automatic queue or claims of measured gain.

LATEST: C06 stopped by explicit user request2026-09-19. Current C07 selection
and rolling alternatives are in [METHOD_SELECTION_C07.md](METHOD_SELECTION_C07.md).
C07 changes joint-level interaction before native part pooling; no new dataset,
loss or scorer. Borrowing strong prior-art modules is expressly allowed, and
published overlap alone is not grounds for rejection. Treat C07 as adaptation
pending measured gains, matched controls, replication and generalization.

## C01 — fused-priority supervision (implement now)

- Hypothesis: full-strength branch retrieval objectives constrain fused adaptation.
- Source: `third_party/SEDS/modules/modeling.py:253`, three losses equally weighted.
- Change pose/text and RGB/text losses to .25 each; fusion1, match.4 unchanged.
- Release initialization; native cached-RGB/online-pose model; all native parameters.
- FP32 BertAdam moments, B32, seed42, LR1e-5/sign_lr1e-4, 222 updates.
- Same input order, schedule and evaluation steps0/111/222 as V1 corrected control.
- Anchor77.552987; incumbent/control-selected77.649326; full DEV519, both directions.
- Control: existing `seds-moment-control-001`, not the degraded native optimizer.
- Inference unchanged; ~660s/~22GiB/3.9GiB. Smoke2 updates first.
- Promote a gain; refine with one motivated adjustment if trajectory warrants;
  initialization selection alone does not close the family. Up to3 refinements.
- Collision: [V1 registry](../slret_goal/NO_GO_REGISTRY.md): no teacher, residual,
  gallery risk, augmentation rescue or protected gradient. Ordinary objective
  balancing on SEDS, not the closed CiCo sentence-weighting readout.
- Command: native SEDS torchrun of `tools/train_seds.py --aux-weight .25`;
  output `artifacts/slret_goal_v2/seds-aux025-001`, prospective registration.

## C02 — staged fusion/upper-block adaptation (next architecture/optimization level)

Freeze pretrained lower blocks initially, train existing fusion/projections,
then unfreeze upper visual blocks with smaller LR. Hypothesis: avoiding early
full-model drift permits useful fusion adaptation. No teacher/projection of
gradients or frozen residual scorer, so not RPCA/CICO-REOPEN. Match parameters,
steps and warm-start against a simultaneous-unfreeze control before attribution.
Use train/dev curves to choose one schedule; not the V1 RGB-tail222 recipe.

Registered C02a before training: release checkpoint, aux1 (no C01 combination),
B32 seed42,666updates/3epochs; fusion only222steps then fusion+last visual
transformer block of both streams444steps. Fusion LR1e-5, upper visual LR2e-6,
native cosine/warmup perstage, fresh FP32 moments at transition. Text encoder,
GCNs and lower visual blocks frozen in eval mode, including BN statistics.
This changes trainable subset/recipe; exploratory attribution requires a
simultaneous-upper-training matched control later if promising. Retain original
reference/incumbent and full DEV111/222/444/666; frozen-stage step0 unchanged.
Runtime estimated900–1500s, hard2400s,<=22GiB VRAM,<=2GiB storage. Two-step
stage-transition smoke first (fusion stage2steps, remainingsteps4).

## C01a outcome

seds-aux025-001 completed620.507372s. DEV111mean76.782274, DEV222mean77.456647;
selectedstep0=77.552987 (−.096339 vs incumbent). No gain; defer refinement
while testing C02 instead of raising confidence via repeated audits. First
contrast is valid; weak endpoint recovery does not justify a success claim.

## C03 — checkpoint-compatible cross-articulator interaction

Add a zero-initialized low-rank interaction between hand/body embeddings before
SignBERT temporal convolution; preserve reference at initialization. Hypothesis:
independent early streams omit hand–body relations. Same pose input, no extra
stream/confidence gate/local rival supervision. Compare parameter-matched
independent per-stream transform if lead; inspect actual tensor insertion first.

Implementation prepared while C01 runs: `methods/seds_adaptation/articulator_interaction.py`.
Native insertion confirmed at `Sign_Bert.gcn_emb`, after GCN_Embed concatenates
[left,right,body] into1536 channels and before clip windows/temporal convolution.
Rank32,196608 added parameters; output projection zero-init. Additive control
has exactly the same parameter count. Unit scope: identity, gradient activation,
shape and checkpoint reconstruction, not retrieval evidence.
Generic prior: [Kim et al., Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325),
primary abstract verified2026-09-19; full paper not read. Multiplicative low-rank
fusion is established, not a new invention. This candidate is a specific early
articulator adaptation; novelty remains unresolved and does not block pilot.

Prospective C03a pilot: release initialization, product rank32, native aux1,
all native parameters plus interaction trainable,222steps B32 seed42,
native LR1e-5/sign_lr1e-4 and FP32 moments. Same schedule/order as V1 corrected
control and C01; do NOT combine C01 auxiliary reduction or C02 freezing.
Insertion zero-init requires one initial fullDEV parity check, then111/222
evaluations. Smoke2updates must change interaction output weights and preserve
finite gradients. Estimated650s hard1200s,~22GiBVRAM,<=4GiBdisk. If learned
interaction helps versus initialization/incumbent, run additive matched-capacity
control before attributing gain to multiplicative relations. If drift obscures
learning, one motivated smaller encoder-LR refinement is allowed, with control.

## C04 — upper visual projection adaptation with bounded parameter updates

Factorized trainable changes inside existing upper attention projections,
leaving lower pretrained weights fixed; existing scorer and losses. Hypothesis:
restricted adaptation capacity can learn task detail with less drift. This is
not a frozen-context residual readout: feature generation is trainable. Match
upper-block full fine-tuning control if promising; ordinary adaptation initially.

C04 implementation: rank8/alpha8 FP32 factors on combined QKV and output
projection in the last native visual attention block, both pose/RGB encoders.
Frozen original projection matrices, other encoders/BN frozen; existing fusion
trainable. Prepared `low_rank_attention.py` and `peft_setup.py`. Low-rank factors
can use LR1e-4 separately from fusion1e-5; this candidate is not C02's full
upper-block fine-tuning. Requires matching recipe control before attribution.
Primary prior [Hu et al., LoRA](https://arxiv.org/abs/2106.09685v2), metadata and
abstract verified2026-09-19; full paper not reread. Low-rank adaptation is known;
no novel-component claim. Native torch2.3 setup smoke on CPU passed, GPU pilot
not yet run. Do not interpret unit tests as retrieval performance.

C04a registered before GPU: release initialization, fusion+rank8 LoRA only,
native aux1, B32 seed42,666updates/3epochs; LoRA LR1e-4, fusionLR1e-5,
FP32 factors/moments, frozen lower modules in eval mode. No C01 reduction,
C03 interaction or C05 smoothing. Native schedule spans666updates; initial
zero-init DEV parity then111/222/444/666. Expected~650s hard1800s,<8GiBVRAM,
<=2GiB storage. Two-step GPU smoke must update B factors and no frozen weights.
Control if lead: fusion-only under this same666-step schedule, then simultaneous
full-upper fine-tuning as a stronger recipe comparison. This is not yet attributable
to low-rank parameterization merely by beating the earlier staged recipe.

C04 interim decision atstep444: first exploratory lead77.938343, +.385356
vsreference,+.289017vspreviousincumbent, bothdirectionsup. Complete666 first.
Register C04-control-fusion-001 next: same release, seed42, B32,666updates,
fusionLR1e-5,aux1, frozenencoders/BN, evaluations111/222/444/666, but noLoRA.
This is an absent-component control with less trainable capacity, not a
parameter-matched capacity control. Existing stagedcontrol has different
schedule/parameters and cannot substitute. Expected600s hard1800s,<=2GiBdisk.
If LoRA remains above matchedcontrol and initialization, run seeds1337/2026
before mechanism/confirmation claims. Do not compare selected versuscontrol's
worstendpoint; compare both selected models under the same rule.

Prospective C04 replication plan (before seeds1337/2026): conditional on a
selected seed42 lead over fusion-only, run fusion then LoRA for1337 and2026,
four sequential jobs,666updates each, same settings/selector as seed42. Do not
pick bestseed; report all three paired selected contrasts, sampleSD and
direction/recall trade-offs. These are continuation seeds on one pretrained
initialization and historically exposed DEV, NOT independent confirmation.
Each child hard900s (seed42 pilots~580s), queue hard3800s, estimated2300s;
charge child job walltimes once, do not doublecount driver. Four retained
best/last models need additional<=8GiB; before queue admit a20GiBV2storagecap
with15GiBfree reserve and no historical deletion. This must fit remaining
7200stranche; otherwise reduce admitted jobs explicitly rather than hide costs.
Driver `tools/run_lora_pairs.py` refuses existing childIDs, verifies frozen
source identity, stops on failure without retry. Do not edit runner/modules
while this queue is live. No TEST input or evaluation permitted.

C04 replication COMPLETE: selectedpaired deltas +.385356/0/+.289017pp,
mean+.224791±.200546(sampleSD),2positive/1tie. All3LoRAselected exceedrelease,
but observed effect is small on exposedDEV, not independent/SOTA evidence.

C04b refinement1 registered before GPU: batch32→128 to expose more actual
in-batch negatives, motivated by fusedTRAINloss saturation and LoRAheadroom.
Release checkpoint,seed42,LoRArank8/alpha8,lastvisualblock+fusion,aux1,
LoRALR1e-4/fusion1e-5 unchanged,3epochs=168updates/21288examples (same three
passes as C04a, fewer updates explicitly disclosed). No gradient accumulation
substituted for negatives. Native cosine/warmup scaled to168updates. DEV28/56/
112/168 (halfepoch1 and each epoch), plus step0parity. Compare selected against
release77.552987, currentincumbent77.938343, and batch32seed42; any larger gain
needs batch128fusioncontrol before attribution. Smoke2B128updates hard180s;
pilot hard1200s, expected<1100s pending timing/VRAM. No LR/rank/smoothing change.
Refine/drop based on DEVtrend, not batchloss magnitude. Reserve at most1380s
from currenttranche remaining~1648s; no historical checkpoint deletion.
Runner `tools/train_seds_extended.py` versioned without touching live C03 source.

C04b COMPLETE564.681579s: selected168mean77.745665, +.192678vsrelease,
-.192678vscurrentincumbent; curve77.552987/77.552987/77.649326/77.745665.
Not promoted over B32; rising endpoint alone does not establish a batch benefit.
Register C04b matched control `seds-fusion-b128-001`: remove LoRA only,
release/seed42/B128/3epochs168updates,aux1,fusionLR1e-5,lowerencoders/BN frozen.
Same DEV28/56/112/168 and initialization-eligible guardrail selector.
Compare both selected outcomes and all directional R1/R5/R10; not capacity-matched.
Expected550s, hard900s, <=10GiBVRAM, additional<=2GiBdisk.
Fits tranche remaining1055.626119s. Expand agent-only storage20→24GiB before
launch (88GiB free,15GiB reserve), preserve all artifacts; no compute expansion.
Only runner storage assertion changes; numerical training/evaluation untouched.
Command: launch_bounded.py --name v2-seds-fusion-b128-001 --seconds 900 --
/home/haipd/miniconda3/envs/seds/bin/python -m torch.distributed.run --standalone
--nproc_per_node=1 research/slret_goal_v2/tools/train_seds_extended.py
--run-id seds-fusion-b128-001 --aux-weight 1 --policy fusion --sign-lr 1e-5
--lora-lr 1e-4 --epochs 3 --batch-size 128 --seed 42.
If no paired gain, defer B128 recipe and move C05; if gain, consider one
longer-horizon refinement only with additional tranche admission. No TEST use.
Launch background, report logs and yield; no automatic follow-on job.

## C05 — reduce contrastive overconfidence during pretrained adaptation

TRAIN batches in C01/C02 frequently have fused loss below1e-4, yet no DEV gain.
Hypothesis: saturated paired targets leave little useful adaptation signal.
Try ordinary label smoothingepsilon.05 on existing square paired contrastive
losses, no changed positives/mined negatives/teacher or relevance. This is not
PMGR grouped/gallery risk: same paired minibatches, native scorer and inference.
`methods/seds_adaptation/smoothed_contrastive.py` prepared; epsilon0 recovers
native loss, epsilon.05 matches cross-entropy label-smoothing values/gradients.
No gain/novelty claimed; pilot recipe/card will be fixed if this candidate runs.

C05a prospective contrast: release warmstart, epsilon.05 applied to native
square paired CrossEn calls, aux1, all original trainable parameters,
B32 seed42,222updates LR1e-5/sign_lr1e-4/FP32moments. Same exposure/order and
schedule as V1 corrected control; no C03 module or LoRA. Objective-only change
keeps scorer identical, reuse release step0 scores; two-update smoke checks
finite changed gradients. FullDEV111/222, expected650s hard1200s,~22GiBVRAM,
<=4GiBdisk. If initial gain, confirm lower/higher epsilon through at most one
motivated refinement before seeds; if destructive uniform targets dominate,
reduce epsilon once with stated curve evidence or move family. Target remains
retrieval R1 rather than lower smoothed loss. No TEST accessed.

C05a admitted2026-09-19 afterB128pairedlead only+.096339pp, not incumbent gain.
Jobs seds-smooth005-smoke-001 (2updates,hard180s) then seds-smooth005-001
(222updates,hard1200s), driver c05-smoke-pilot-001 hard1500s.
Runner tools/run_c05_pilot.py uses recipe above; smoke must finish exit0 with
finite loss/gradients and allfour nativeparamgroups updated beforepilot launches.
Sourcehash guard, uniqueIDs, stoponfailure/noautoretry. No morejobs afterpilot.
Expected total~700-1100s, <=22GiBVRAM, <=4GiBretained; hard1500s reserved.
Tranche1used7009.523341/7200s; allocate tranche2 3600s fromremaininglocalbudget,
leaving~10882.476659s unallocated and190.476659s unusedtranche1.
V2storage24→28GiB, current22GiB/86GiBfree, reserve15GiB, no historical deletion.
Only new scientificchangeepsilon.05; selector1e-8pp tolerance fixes numericalties
prospectively (B128means unaffected); compare oldcontrol withsame tie convention.
Novelty unresolved/standard regularizer; skill used for bounded fail-closed chain.

## C04c — depth refinement, upper visual LoRA 1 → 3 blocks

C05a COMPLETE: finalmean60.597303, -16.955684ppvsrelease, selectedinitialization.
Drop epsilon.05/fullmodelrecipe; no extension/repeat or LoRA+smoothing combination.
C04 refinement2 (afterB128): hypothesis one upperblock constrains useful
adaptation; C04B32 has positive paired gain in2/3seeds but small, fullmodelC05
severelydegrades. This motivates moredepth withinPEFT, not proof depth is cause.
Change only last1→last3visualattentionblocks in bothRGB/pose; rank8/alpha8;
fusiontrainable, lowernativeweights/BN frozen. No teacher/gate/mining/newlabels.
Not closedstagedC02/fulltailrecipe: lowrankQKV/outputupdates, nativeweightsfrozen.
Releasewarmstart, B32seed42/3epochs666updates, nativeaux1, LoRALR1e-4/fusion1e-5.
DEV111/222/444/666; initializationeligible, earliestties1e-8pp, directionguard-.5pp.
Newprospective safety stop if fullDEVmean drops >5ppvsrelease; record/savelast,
not classify as runtimefailure. Compare on common observed horizons if triggered.
Controls: existingdepth1sameB32recipe andfusiononly; no baseline rerun. Extra
capacity/compute confounded withdepth; capacitymatched rankcontrol only if lead.
Smoke2updates checks12Bfactors update, frozenweights unchanged, fullDEVzero-init.
Jobs seds-lora-depth3-smoke-001 hard180s → seds-lora-depth3-001 hard1800s;
driver c04-depth3-smoke-pilot-001 hard2100s. Expected~15–25min, <=8GiBVRAM,
<=2GiBdisk, current25GiB+2within28GiBcap,83GiBfree/15GiBreserve.
Tranche2remaining2941.297017s admits2100s; allV2used7668.226324s beforechain.
Command: launch_bounded.py --name v2-c04-depth3-smoke-pilot-001 --seconds 2100
-- /home/haipd/miniconda3/bin/python research/slret_goal_v2/tools/run_c05_pilot.py
--profile lora-depth3. Registeredprofile retains historicalC05default but notrun.
If selectedgain exceedsdepth1, confirmseed/control; if flat/degrades, deferdepth3
and move differentmodellevel. No randomsweep, TEST tuning or automaticnextjob.

## C04d — shared text-side adaptation (third targeted C04 refinement)

Depth3selected77.842004 belowdepth1 77.938343; defer visualdepth expansion.
Hypothesis: fixedtextfeatures constrain video/textalignment. StandardLoRA,
noveltyunresolved; neither scalarpooling nor closedteacher/gating/reranker.
Source modeling.py:get_sequence_output calls clip.encode_text only, notclip_rgb.
Add rank8alpha8 LoRAQKV/output tolastblock clip.transformer.resblocks[-1],
on visualdepth1+fusionrecipe (not depth3/C05combination); originalweightsfrozen.
Same releasewarmstart/B32seed42/666updates/nativeaux1; allLoRALR1e-4,fusion1e-5.
Text freshlyencoded eachbatch/eval, no frozen downstreamtextcache; causal mask
and nativeprojection/scorer unchanged. Zero-init matchesrelease;6Bfactorsupdate.
DEV111/222/444/666, initialeligible, guard-.5pp, ties1e-8pp, stop>5ppmeandrop.
Existingvisualdepth1 control isolates addedtextparameterization (extra capacity
not controlled); iflead, text-only ablation andseeds required before mechanismclaim.
Ifflat/destructive, deferandmove beyondC04; no fourthrefinement inthistranche.
Jobs seds-lora-text-smoke-001 hard180s→seds-lora-text-001 hard1800s,
chain c04-text-smoke-pilot-001 hard2100s; expected~12–20min,<=8GiBVRAM/2GiBdisk.
Tranche2remaining2302.653418s admits2100s; retain15GiBfree reserve, current81GiBfree.
V2storage28→32GiB beforelaunch, current27GiB; no historical deletion.
Command launch_bounded.py --name v2-c04-text-smoke-pilot-001 --seconds 2100 --
/home/haipd/miniconda3/bin/python research/slret_goal_v2/tools/run_c05_pilot.py
--profile lora-text. Sourcehashguard/failclosed/noautoretry, backgroundthenyield.

## C03b — train hand/body interaction with frozen pretrained encoders

C04text COMPLETEselected77.842004, noincumbentgain; deferC04after3refinements.
C03 firstrefinement: priorC03fullmodel222updatesselectedinit, final77.071291.
Hypothesis encoder drift may obscure useful multiplicative representation;
freeze nativeencoders/BN, train only196608interactionparams+existingfusion.
Same rank32products afterGCN [left,right,body] beforetemporalwindows; zerooutput.
Not LoRAcombination, teacher, confidencegate, mining, scalarpooling or reranker.
Differencefromearlierfailure: trainablesubset, longerhorizon, lowernewmoduleLR;
composite recipe refinement, cannot attribute change solely to freezing.
Releasewarmstart/B32seed42/3epochs666updates/nativeaux1; LR1e-5forbothmodules.
Backpropagate through frozen downstreamtail; no stale downstreamfeaturecache.
DEV111/222/444/666, initialeligible, -.5ppdirectionguard, 1e-8ppties, >5ppdropstop.
Existingfusion-only666control matchesrecipe; parameter-matched additiveinteraction
control available but run only iflead; no claim productshelp withoutthatcontrol.
Smokechecksoutputactivation/frozenweightintegrity/zero-initDEV; failurestopschain.
Pilot expected~15–24min,<=22GiBVRAM,<=2GiBdisk; hard1500s+smoke180s, chain1800s.
Pool unusedtranche1 190.476659s withtranche2remaining1630.095417s =1820.572076s;
no totalallowanceincrease. Current29GiB within32GiBcap after<=2GiB,79GiBfree.
Jobs seds-articulator-frozen-smoke-001→seds-articulator-frozen-001;
chain c03-frozen-smoke-pilot-001, using run_c05_pilot.py --profile interaction-frozen.
Launch: launch_bounded.py --name v2-c03-frozen-smoke-pilot-001 --seconds 1800
-- /home/haipd/miniconda3/bin/python research/slret_goal_v2/tools/run_c05_pilot.py
--profile interaction-frozen. No retry/nextjob; backgroundlogs thenyield.
Ifleadvsfusion/initial, additivecontrolnext; ifflat/drops, deferrecipe and pivot
to scorer/temporalrepresentation hypothesis, not fourthLoRArefinement. NoTEST.

## C06 — masked temporal-difference token adapter

Hypothesis: explicit localchange vectors can improve matching beyond existing
contextual tokens; C03pointwiseinteraction has no gain afterfrozen3passrefinement.
Not proof temporal information missing. Nativeget_visual_output yieldsCLS+clips;
insert beforefusion/scoring, separatelyRGB/pose; downstreamtokens freshlycomputed.
z=nonaffineLayerNorm(x); project concat(z[t]-z[t-1],z[t+1]-z[t])→rank32→GELU→D,
add to originaltoken. Outputprojectionzero; adjacentvalidclips only; preserveCLS/pad.
This is a specific video representation change, not frozen contextual score
residual/probe, lexical/rivalteacher, scalarclipweights, orderloss or resampling.
Collision: research/slret_goal/NO_GO_REGISTRY.md residual-readout/temporal-order
rows concern those exactoperators, absenthere. No reopeningclosedmethod.
Nativeencodersfrozen, train temporal_delta+fusion fromrelease; noLoRA/C03/C05combo.
B32seed42/3epochs666updates, LR1e-5fornewmodule/fusion, aux1/FP32moments.
DEV111/222/444/666; baseline/incumbent77.552987/77.938343; sameguard/ties/5ppstop.
Existingfusion-only666control; iflead, equalcapacitypointwisecontrol required
before claiming temporal mechanism. Noveltyunresolved; ordinarydifferenceadapter.
CPUidentity/padding/CLS/singleton/order/gradient/checkpointroundtrip tests passed;
GPUsmoke mustactivatebothoutputprojections andpreserve frozenweights/fullDEVzero-init.
Jobs seds-temporal-delta-smoke-001 hard180s→seds-temporal-delta-001 hard1800s,
chain c06-temporal-smoke-pilot-001 hard2100s,expected12–22min,<=8GiBVRAM/2GiBdisk.
Allocatetranche3 3600s from10692sunallocated; previousallocatedresidual382.616394s;
totalremaining11074.616394s, newunallocated7092s. No totalbudgetincrease.
Storage32→36GiB beforelaunch,current31GiB/78GiBfree; preserve15GiBreserve/allhistory.
Command launch_bounded.py --name v2-c06-temporal-smoke-pilot-001 --seconds 2100
-- /home/haipd/miniconda3/bin/python research/slret_goal_v2/tools/run_c05_pilot.py
--profile temporal-delta. Freeze livecode; noautoretry/nextjob, backgroundthenyield.
Ifgainvsfusion andinitial, capacitycontrolnext; ifflat, inspect update scale once
before one motivatedLRrefinement or pivot. Never useTEST fordecision.
