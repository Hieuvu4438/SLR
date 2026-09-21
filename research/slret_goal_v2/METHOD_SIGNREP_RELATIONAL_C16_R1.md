# C16-R1 — within-video distance transfer

## Material Passport

ARS experiment-agent/run, inline2026-09-20; GoalV4 and task research module8.
Exploratory first structural refinement of C16, not a new invention or SOTA claim.

Evidence: pointwise external SignRep supervision reduces auxiliary loss but loses
to matched noaux by.289017pp at160. Teacher TRAIN64 features do not collapse:
meanpaircos.2115, mean within-video centeredenergyfraction.7627. Student collapse,
modality-conflict and gradient interference are NOT established causes.

Hypothesis: matching individual RGB feature directions imposes modality-specific
coordinates on2D pose. Matching relative distances may transfer clip distinctions
while freeing rotations/translations/scales of the student space. It may still
transfer appearance variation not inferable from pose; teacher geometry may be
irrelevant to sentence retrieval. This pilot distinguishes loss formulations,
not a proven causal diagnosis or universal modality-mismatch theorem.

Borrow RKD-D, Park et al., CVPR2019, Eqs5–8:
https://arxiv.org/html/1904.05068v1 (method and metric-learning experiments read).
Official proceedings confirms venue:
https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html
Official source https://github.com/lenscloth/RKD inspected for provenance, not
cloned/copied (no license shown at root). Implement equations independently.
No new weights/dependencies; existing SignRep checkpoint/code license and IDs
remain as METHOD_SIGNREP_C16.md. Donor metric-learning image benchmarks do not
establish SLRet efficacy. Search2026-09-20: targeted RKD-D distance-wise transfer;
not a comprehensive latest-literature/novelty review.

Operation: same LayerNorm1536->Linear768 head, nativeGCN clip input [B,64,1536],
detached teacher [B,64,768]. Within each video, all valid distinct clip pairs,
Euclidean distance divided by its own valid-pair mean; mean Huber(delta1) over
pairs then equally over eligible videos. Exclude padding/selfpairs, n<2 contributes
zero. FP32 Gram distances, epsilon clamps. Native seven losses +.1*RKD-D;
no pointwise loss combined, no angle loss, no capacity/LR change. Train-only;
native RGB + pose2D inference and scoring unchanged, head discarded.

NO-GO: not baseline score/ranking protection, reference subtraction, gradient
surgery, teacher-selected positives, duplicate relevance, nuisance removal,
gallery graph, OT or temporal order/reversal consistency. No extra inference
stream/readout/gate. Relationships use only external input clip features on TRAIN;
no DEVteacher or query labels. This is a substantive loss change, not reopening
user-closed C06. Common KD ingredient is not itself a causal collision.

Recipe: native release, TRAIN512/fullDEV519, seed42/B32/tenepochs160updates,
encoder1e-6/fusion1e-5/head1e-4, FP32moments, offload/mathSDPA. Select0/80/160
by unchanged bidirectionalR1/guardrails. Fixed160 also reported. Existing noaux
control78.323699 and pointwise78.034682 are reuse comparators, not rerun.
Current best78.709056 remains. Minimal CPU algebra/gradient/mask/teacher-detach
tests plus existing real step2 head/GCN/fusion and frozen-buffer checks; failclosed
on those gates. FullDEV initialization retained for matched recalls, not rawscore
parity; historical cross-process numeric uncertainty remains disclosed.

PROMOTE for fuller-data consideration only if meaningful matched improvement;
weak/tie/lower result with learning deprioritizes direct C16 transfer recipes.
If very weak gradients or instability, first inspect measured loss/gradient scales
before any single further motivated refinement; no automatic sweep/nextjob.
One seed/subset cannot establish generality. No SOTA/fullteacher-value claim.

Run v4-c16-relation-001 -> seds-signrep-relational-offload-001, estimate23–27min,
hard1800s, outer1850s. Reserve1850s from16983.657679s available. User20GB maximum,
allocator16GiB/local safetytrip19GB, corrected descendant tracker. Storage cap38GiB
(agent subcap adjustment allowed V4), freefloor15GiB, saveheadroom1.25GiB.
No deletion/extraction/download. Launch and one startupcheck then WAITING_FOR_USER.

LaunchedUnix1789905444.122684, supervisor3608694/torchrun3608722/trainer3608764;
startup alive, initialDEV entered. Corrected process tree actually observed
2944401408GPUbytes, excluded otheruser3493585. Eight scoped CPU tests passed
(six loss/gradient/mask/invariance tests, two process-tree/failclosed tests).
Matched-report schema test and py_compile passed. These do not establish retrieval
efficacy; real step2/DEV gates remain inside the pilot. WAITING_FOR_USER now.

Collected on actual user return2026-09-20: COMPLETED160steps, all gates pass,
best78.227360 vs noaux78.323699 (-.096339pp). No promotion; RESULTS.md has both
directional metrics, curve and loss scales. Last10 weighted relation loss is
.001485698 vs native.047221867; native-edge gradient comparison unavailable.
R2 challenges possible loss-strength explanation once, not proven underweighting;
same data/optimizer/horizon, only .1->1 coefficient. If no matched gain, defer
these direct transfer recipes. Incumbent and all C16 best/last artifacts retained.
