# C26 decision — translation-pretrained sign encoder for contrastive retrieval

Status (2026-09-21): **C26-A measured, standalone recipe deferred**. All seven
pinned donor/mT5 assets have been downloaded and hash-verified; real CPU
strict-load/forward passes. The user granted an additional maximum five
GPU-hours with a 20 GB VRAM cap and requested UniFormerV2 pause/resume. The
completed exploratory result is recorded below. C25/CoSign-LI is USER_DEFERRED before
efficacy testing; C24 remains deferred. No claim of gain, novelty, or SOTA is
made here.

## Why this direction

The first controlled question is whether a sign encoder pretrained for *sentence
translation* carries more retrieval-relevant information than the released
retrieval encoder on a matched dataset. Uni-Sign provides a public CSL-Daily
pose-only translation checkpoint and separate body, face, left-hand and
right-hand streams. Its mT5-based encoder gives a natural text-side initialization.
This changes the visual representation and its pretraining, not a small residual
module on SEDS. It retains the core paired video/text contrastive training.

The candidate C26-A is a **global dual encoder**: frozen Uni-Sign pose encoder,
frozen mT5 text encoder, trainable shallow projections, L2 normalization,
and symmetric in-batch InfoNCE. Use one score matrix for T2V and V2T. No window
max, local text-token matching, hard-negative mining, duplicate equivalence,
score ensemble, or test-set tuning. C26-A is an engineering transfer baseline,
not a novel method by itself. C²RL already pairs pretrained visual features and
an mBART text encoder with cross-lingual contrastive retrieval. A distinct paper
claim must be formulated from measured C26-A failure/success modes and compared
directly to C²RL; changing the backbone alone is insufficient.

Primary sources: Uni-Sign ICLR 2025 official code
<https://github.com/ZechengLi19/Uni-Sign>, weights
<https://huggingface.co/ZechengLi19/Uni-Sign/tree/main>, paper
<https://openreview.net/pdf?id=0Xt7uT04cQ>; C²RL
<https://arxiv.org/html/2408.09949v1>; CiCo
<https://openaccess.thecvf.com/content/CVPR2023/html/Bao_CiCo_Domain-Aware_Sign_Language_Retrieval_via_Cross-Lingual_Contrastive_Learning_CVPR_2023_paper.html>.

## Admission gates before a large job

1. Work on CSL-Daily first because the public Uni-Sign donor is trained for
   CSL-Daily; no PH checkpoint was found in the inspected official weights.
   Freeze DEV as selector and do not use TEST for method selection (historical
   PH/CSL TEST scores were already exposed). Obtain a
   dataset-level SEDS CSL DEV reference under exactly the same gallery/protocol.
2. Use the existing `keypoint/` CSL poses, not
   `keypoint_focus_hand_w_face/`: inspected TRAIN sample
   `S000000_P0000_T00.pkl` has 168 frames with `keypoints` `(1,133,2)` and
   `scores` `(1,133)`, matching Uni-Sign's loader keys and index range. The
   focus-hand copy has only 77 points and is incompatible. This passes the
   structural gate for one sample, not the semantic/coverage gate. Spot-check
   left/right hands, confidence scale, missing points, and TRAIN/DEV coverage.
   Uni-Sign's source randomly subsamples clips longer than `max_length` even
   in its generic `load_pose`; use deterministic DEV sampling for retrieval.
   Follow-up CPU check: bundled Uni-Sign TRAIN/DEV label files are byte-identical
   to local CSL labels (SHA256 `e7fb4bba...`/`b96b62b7...`); all 18,401 TRAIN
   and 1,077 DEV named `keypoint/` pose files exist. Three fixed DEV samples
   have 133 keypoints and 133 scores per frame. This proves path coverage, not
   every file's semantic/keypoint validity.
3. Pin the donor checkpoint and exact source commit, prove strict/expected
   loading, output finiteness, and measured peak owned VRAM below 20 GB decimal.
   The inspected HF weights are CC-BY-NC-4.0; the cloned source had no root
   LICENSE file. Keep the donor checkout locally ignored and do not redistribute
   its code/weights in this repository without permission review.
4. Predeclare a fair comparison: C26-A versus (a) **zero-shot** cosine between
   frozen Uni-Sign visual and mT5 text encoder outputs *before* trainable
   projections, and (b) a matched contrastive control using the same data,
   mT5 text encoder, batch order, updates and selector but an existing frozen
   CiCo/SEDS visual feature (only if the exact CSL sample mapping is available).
   C26-A itself already trains linear projections; calling that a separate
   control would be a duplicate arm. Report parameter and pretraining/data
   differences explicitly. Also report the native SEDS CSL release checkpoint
   separately, without attributing donor-pretraining gains to the new loss.
   Evaluate full-gallery directional R@1/5/10 and mean R@1.
5. A small, fixed TRAIN subset may establish pipeline viability, not efficacy.
   Promote to full-TRAIN only if it is finite, trains both projections, fits
   VRAM, and shows nontrivial full-DEV retrieval signal beyond its zero-shot
   starting point. After full training, require a practically useful mean
   R@1 gain over the matched control without a severe directional regression.

## Resource / job boundary

The official pose-only checkpoint is about 1.2 GB; validating and encoding
CSL TRAIN/DEV plus a fair control/candidate pair is substantially more than
the formerly remaining registered ~430 GPU seconds. The user granted an
additional five GPU-hours on 2026-09-21, still <=20 GB owned VRAM. C26-A's
first fixed pilot is capped at 18,000 seconds wall (conservatively charged as
GPU occupancy), leaving the previous ~430-second balance untouched. This is
a cap, not a claim that the entire full-TRAIN/control study will fit. Use the
V4 detached-job protocol:
bounded queue, run.log/status.json/summary.json, one startup check, then
WAITING_FOR_USER. The wrapper must stop the *exact verified* UniFormerV2 process
group only while C26 owns the GPU, and resume it in a `finally`/exit trap on
success, failure, timeout, or interruption. No polling by the assistant.

## Rejected adjacent shortcuts

- C25 late interaction: user-deferred, not tested.
- Plain Uni-Sign + InfoNCE as a paper novelty claim: overlaps C²RL.
- SignCLIP transfer as first choice: its dictionary/short-sign pretraining and
  MediaPipe input are a domain and pose-format mismatch to sentence CSL/PH.
- Score-blend consistency on SEDS: the release scores are already strong and
  a convex mixture of separate CE losses does not supply an identifiable new
  capability; insufficient causal case for scarce GPU time.

The useful next scientific question is *whether translation pretraining improves
retrieval under a controlled contrastive objective*. If yes, investigate the
specific error slices and a genuinely distinct mechanism. If no, drop this donor
without checkpoint/temperature sweeps and revisit a different visual prior.

CPU implementation prep: `methods/translation_retrieval/bridge.py` extracts the
four-part pose sequence before Uni-Sign's translation decoder, encodes it and
captions through the frozen mT5 encoder, and trains only two global projections
under symmetric InfoNCE. The donor is supplied from the ignored local checkout;
no code/weights are vendored. `methods/translation_retrieval/csl_data.py` reads
the existing 133-point pose files, uses reproducible TRAIN subsampling and
fixed uniform DEV subsampling, and returns the four anatomical parts with masks.
For one real DEV clip (`S000020_P0000_T00`), its normalized part tensors agree
with the donor's `load_part_kp` to max absolute error below 2e-7; that is an
input-numerics check on one clip, not a corpus validation. Scoped CPU tests
cover shape, masking, gradient flow and frozen donor parameters.
The real checkpoint strict-loads on CPU with 627 state tensors and zero
missing/unexpected keys. A two-example real CSL DEV forward yields finite
video/text features [2,768]. This surfaced a pose-normalization float64
promotion, now fixed with a float32 regression assertion; eight focused CPU
tests pass. This is compatibility evidence, not a retrieval efficacy result.

The projection-only path now has `methods/translation_retrieval/cached.py`:
cache frozen pre-projection video/text features once, compare their zero-shot
cosine matrix, then train a separate symmetric-InfoNCE projection head from
those same features. This prevents repeated 1.2 GB donor forward passes during
the projection pilot. Two new CPU tests cover cache IDs/shapes, zero-shot
scores, finite loss and gradients; all 16 track tests pass. This correction also
removes the earlier duplicate "linear-projection control" from the plan.

Environment note: the existing `seds` environment has PyTorch 2.3.1 while
its installed Transformers 5.5.4 disables PyTorch model support (requires
>=2.5). The repository's base Python has PyTorch 2.11.0/Transformers 5.5.4 and
successfully imports Uni-Sign's `models.py`; use an isolated compatible donor
environment or this base environment for the C26 checkpoint smoke, not the
SEDS environment. Uni-Sign's generic `datasets.py` import additionally pulls
in `deepspeed`, which is unnecessary because the independent pose loader now
matches its normalization on the inspected clip. Neither the 1.19 GB CSL
checkpoint nor mT5-base initialization assets were present at the inspected
donor checkout/cache paths at that time. Both are now downloaded under
`artifacts/pretrained/c26_unisign_csl/`.

Download-only amendment (2026-09-21): after byte-identical CSL labels, full
named-pose coverage, one-sample pose numerical parity and 123 GiB free disk,
admit only the pinned `csl_daily_pose_only_slt.pth` plus the six files needed
for `google/mt5-base` initialization. The HF API reports 3,520,971,333 bytes
total; reserve at most 4,000,000,000 bytes and retain a 20 GiB free-space
floor. This is network/disk work, not a GPU compute top-up; it does not alter
the ~430-second GPU balance. The downloader in `tools/download_c26_assets.py`
uses exact HF commits, checks expected size and SHA256, avoids the rest of the
80.2 GB Uni-Sign repository, and runs detached under the V4 log/status rule.

## Fixed first GPU run after the user's five-hour grant

Run `c26-unisign-csl-pilot-001` with
`tools/run_c26_with_pause.py --launch`; its worker is
`tools/run_c26_pilot.py`. Exact UniFormerV2 worker PID/start-ticks/command are
checked before SIGSTOP, and SIGCONT is issued in `finally` after C26 succeeds,
fails, times out or is interrupted; a separate local watchdog covers supervisor
death. Stop only the GPU worker, not its foreground shell/tee process group.
Supervisor wall timeout is 18,000 seconds and C26-owned peak reserved VRAM is
checked against 20,000,000,000 bytes. Free-space floor is 20 GiB.

Frozen-feature extraction: CSL DEV 1,077 first, then TRAIN 18,401, pose input
up to 256 frames, inference batch 8, atomic 128-example cache shards. Fail on
nonfinite features, wrong 768-dimensional shape, resource violation or missing
data. No TEST or donor gradients. Then compute full 1,077-video/797-Chinese-
caption-group zero-shot DEV retrieval using the existing multi-positive CiCo
metric contract. Train only 256-dimensional video/text linear projections and
logit scale with symmetric InfoNCE on cached TRAIN features: five epochs,
AdamW 1e-3, batch 256 with no duplicate caption group within a batch, seed 42.
Evaluate full-gallery DEV after every epoch; retain the highest mean
bidirectional R@1 head and all epoch metrics/scores. This run is exploratory:
it compares zero-shot and trained C26-A but cannot attribute any gain to a new
mechanism or replace the matched CiCo/SEDS visual-control arm. That control
must be decided from the results in a later user turn, not silently queued.

## Measured result and technical repair

`c26-unisign-csl-pilot-001` extracted all 1,077 DEV and 18,401 TRAIN frozen
features in atomic shards and measured zero-shot mean DEV R@1 5.785507, then
FAILED at epoch evaluation because `score_matrix` assumed a paired square
batch; CSL retrieval needs a rectangular 1,077-video × 797-caption-group
gallery. The error was in the C26 evaluator, not the donor or extraction.
UniFormerV2 was automatically resumed. `ProjectionRetrievalModel.score_gallery`
now scores unmatched gallery sizes; a regression test covers this contract.
The 17 track CPU tests pass. No feature extraction was repeated.

Cache-only technical retry `c26-unisign-csl-pilot-002` COMPLETED, exit 0,
five epochs of 73 updates each. Full DEV mean bidirectional R@1 rose by epoch
from 36.710610 to 41.023091, 42.908935, **43.700670** (selected epoch 4),
then 43.154867. Selected T2V R@1=44.040151 and V2T R@1=43.361188.
All 1,077 videos and 797 caption groups were evaluated using the existing
multi-positive CiCo metric contract; no TEST was loaded. Existing CiCo CSL DEV
reference mean R@1 is 67.403588, a 23.702918-point gap, but CiCo has
different pretrained features and English-rendered captions, so this is a
practical reference, not a matched causal estimate of donor quality. The
plain frozen Uni-Sign/mT5 global-projection recipe is therefore not an
accuracy lead; do not claim novelty/SOTA or spend the remaining grant on an
unmotivated LR/epoch sweep. Both supervisors verified UniFormerV2 resumed.
Combined wall charge was 123.087 seconds; extraction peak C26-reserved VRAM
was 3,210,739,712 bytes. Retain the feature shards, best head, metrics, logs
and failed attempt record for reproducibility; no cleanup is needed at present.
