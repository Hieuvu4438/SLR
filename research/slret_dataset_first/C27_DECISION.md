# C27 — train-only gloss-sequence supervision for contrastive SLRet

## 2026-09-22 recovery and eligibility result

After user `xong`, the recovery is verified complete within its scoped artifact
checks. However,13/7096 TRAIN examples have fewer native visual windows than
CTC requires; all13 failures were confirmed using the native loader. The
unchanged head raises on these examples. See the
[verification report](../../docs/codex_slret_research/evidence/C27_RECOVERY_RESULT.md).
No training was launched and no labels, sampling or head behavior were changed.
C27 remains OPEN, not SUPPORTED-FOR-PILOT. Technical repair alone would not
resolve the existing bottleneck/novelty/control deficiencies. The operational
`xong` dependency below is historical and now resolved.

## 2026-09-22 Cycle 2 scientific update

The exact-multiset order-confusion motivation is **REJECTED in its measured
scope**: among 113 persistent T2V and 98 persistent V2T errors across three
selected GCN seeds, no strictly outranking confuser shares the reference gloss
multiset in a different order. See the
[predeclared diagnostic and result](../../docs/codex_slret_research/evidence/GCN_RESIDUAL_RESULT.md).
This is not a test of CTC efficacy or a rejection of all gloss supervision.
Broader C27 remains **OPEN**, not SUPPORTED-FOR-PILOT. Successful prerequisite
recovery alone must not trigger training; an adequate mechanism diagnostic
and the same-gloss order-free control remain required. The recovery job was
not checked or changed. The older operational text below is retained as history.

## 2026-09-22 evidence/admission addendum

Scientific candidate status: **OPEN**, not yet SUPPORTED-FOR-PILOT under the
current proposal9 guide. The execution handoff below is unchanged; this audit
did not check or modify the feature-recovery job. Feature availability alone
will not establish method admission.

A new read-only TRAIN audit verifies 7,096 gloss sequences/1,085 types and finds
23 exact gloss-multiset groups containing distinct orders (140 rows). Of those,
22 groups/138 rows include a pair with distinct exact translation. This is
annotation structure, not a measured retrieval defect or validated negatives.
The formal CTC length-plus-adjacent-repeat maximum is30, but actual loaded-mask
eligibility remains to be checked after the recovery handoff. Five focused CPU
tests pass (two new annotation fixtures and the three existing CTC tests).

The primary-source collision screen adds CSLR²'s joint sign/sentence retrieval
and its contrasting CE/SignRet auxiliary results. Alongside CVT-SLR, this makes
the broad joint-supervision novelty claim untenable. A future C27 experiment
must compare an order-free auxiliary using the same gloss annotations and head
capacity, in addition to zero-CTC native continuation. Otherwise order-specific
benefit is confounded with extra expert annotation. The order-related bottleneck
and adequate frozen-feature diagnostic are still unverified.

Full evidence, source links, controls and reviewer criticisms:
[`docs/codex_slret_research/00_RESEARCH_STATE.md`](../../docs/codex_slret_research/00_RESEARCH_STATE.md),
[`07_HYPOTHESES_AND_FALSIFICATION.md`](../../docs/codex_slret_research/07_HYPOTHESES_AND_FALSIFICATION.md),
[`08_PRIOR_ART_COLLISIONS.md`](../../docs/codex_slret_research/08_PRIOR_ART_COLLISIONS.md).

## Original operational checkpoint and hypothesis

Status (2026-09-21): **WAITING_FOR_USER on detached feature-recovery prerequisite**.
The C27 retrieval trainer is not yet launched. Job
`c27-ph-feature-recovery-001` began at 22:51 local; see track `STATE.md` and
the job's `status.json`/`run.log`. Its one startup check observed the pose
checkpoint download, before any C27 extraction began.

## Question and hypothesis

PHOENIX-2014T TRAIN contains one nonempty, ordered gloss sequence for each of
its 7,096 sign videos (1,085 TRAIN gloss types; median 7, 95th percentile 14,
maximum 30 tokens). Sentence translations are not sign-by-sign labels. A
contrastive retriever can align whole videos to sentences while underweighting
some order-sensitive sign evidence. Hypothesis: a training-only CTC head on
SEDS's contextual pose-window tokens, alongside its *unchanged* video/text
contrastive matrix loss, improves both full-gallery PH DEV directions over an
otherwise matched continuation. This is a hypothesis, not a measured defect or
novelty claim.

The head predicts the TRAIN gloss sequence without temporal gloss boundaries;
CTC targets are supplied from TRAIN CSV only. At inference it is removed:
video/text encoders, fusion, scorer, gallery and caption relevance are native
SEDS. Video-mask zero means valid and index zero is CLS; only the subsequent
valid contextual windows enter CTC. No hard negatives, teacher, score residual,
new positives, gallery graph, test tuning, or synthetic word-order labels.

The strongest alternative explanation for any gain is *extra expert annotation*,
not the CTC operation itself. Report this as a gloss-supervised setting, not a
gloss-free SOTA comparison. The first matched control has the same starting
checkpoint, batch order, trainable encoder/fusion scope, optimizer exposure and
dev selector but CTC weight zero. Then compare CTC-active and control at equal
steps; report the existing SEDS GCN-R1 DEV 78.709056 separately as practical
incumbent, not causal control. Check CTC validity/finite loss and gradient into
the pose encoder. Predeclare mean bidirectional R@1 as selector, with R@1/5/10
and a −0.5 pp directional guardrail. If the pilot has no positive trend against
its matched control, do not tune a CTC weight sweep. If it does, test seeds and
CSL-Daily with gloss supervision before a research claim.

## Prior and collision

SEDS [source](https://arxiv.org/abs/2407.16394) supplies the strong contrastive
retriever; CTC plus visual/text contrastive learning has been used in sign
*recognition* (CVT-SLR,
[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.html)).
The precise training-only retrieval adaptation is being tested, not presented
as invented CTC. It differs from closed ELSC/SSSC (teacher-chosen lexical
support and margin), C16 SignRep transfer (external feature target), and
closed temporal reversal/OT/hard-negative families: supervision here is the
dataset's independently annotated ordered gloss sequence and its only new
gradient enters pose tokens via CTC. SEDS itself names a `gloss_atten` fusion
module, but source inspection shows no gloss-label CTC loss in retrieval
training. Literature screening is targeted, not exhaustive; novelty unresolved.

## Current asset boundary

At ~22:45 local time on 2026-09-21, `artifacts/slret_goal/` contents changed
outside this C27 work: previously present C26 jobs and adapted SEDS TRAIN/DEV
features are absent. Free disk simultaneously rose from ~116 to ~165 GiB. The
existing raw PH MP4s, official CSVs, SEDS release checkpoint, old C27-independent
incumbent checkpoints, CiCo features, and UniFormerV2 extraction remain.
**Do not use missing historical paths.** Recovery, if admitted, must use the
same registered adapted-feature recipe under a new output root and a detached
job with logs/status/summary. The official RTMPose-L 384 checkpoint also needs
recovery; its prior verified SHA256 is
`13ce77ad08808333e2d4f850c632ac6068c837108e15f1c3902dcbcf30842db9`.
Do not claim C27 efficacy until both TRAIN/DEV inputs are restored and a matched
pilot actually runs.
