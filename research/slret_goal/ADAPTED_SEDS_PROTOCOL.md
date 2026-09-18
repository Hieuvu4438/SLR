# Adapted SEDS PH feature pipeline — registered before extraction

2026-09-17. Purpose: make a controlled train/dev feature path available despite
unresolved release preprocessing. This is not B_release, a new method, or a
claim that SEDS originally used these extractors. Do not compare adapted dev
metrics with published test recall.

Fixed recipe (no retrieval tuning): decode user-provided PH MP4 frames with
OpenCV, preserve frame count/rate; RTMPose-L 384x288 official 2023 checkpoint,
whole-image bbox, no flip test; native SEDS hand/body confidence filtering,
300-frame limit/interval2, 16-frame windows stride1, uniform maximum64 windows.
For the RGB stream use the existing BSL5K checkpoint and shared deterministic
CiCo I3D preprocessing on exactly the same retained frame-index sequences.
384x288 is the first preregistered extractor, not selected by dev recall or
pose-proximity ranking. Original video rate is retained, not silently called
the paper's 24fps. Record this preprocessing difference explicitly.

Store all original-frame poses, selected RGB features, original frame indices
per selected window, raw/checkpoint/source/config hashes and output hashes.
Require 210x260 PH frames; reject empty native hand support rather than invent
fallback frames. Metadata must distinguish the native filtering rule from
unknown release RGB provenance. Dataset objects for retrieval must refer to
these adapted paths explicitly; never overwrite/symlink over release assets.

First smoke: first two release-TRAIN videos, <=5 minutes, batch8 I3D. Verify
real frame support and window count, output finiteness, end-to-end native SEDS
loader compatibility. If viable, extract canonical DEV519 with same pipeline,
<=30 minutes, then evaluate the released retrieval checkpoint as input-transfer
diagnostic only. A controlled trained adapted baseline must also use this
pipeline on its training split; never train native/evaluate adapted silently.
Full TRAIN extraction is deferred until throughput/storage justify it within
the 2h baseline budget. No test access, no method/hyperparameter selection.

## Registered full-gallery branch diagnostic

After adapted DEV evaluation, use the three saved native score matrices and
shared official tie-aware evaluator. Report RGB, pose and fused recall plus
per-query correct-at1 overlap; oracle union is diagnostic only, never a usable
retrieval score. If fused meanR1 falls at least0.5pp below the stronger branch,
prioritize tracing fusion computation before any new method; this is not proof
that fusion training hurts (branches share training). If fusion improves both
and the majority of fused errors are common branch errors, prioritize input/
representation analysis. No automatic admission of gating, ensemble, or other
NO-GO mechanism follows from complementary branch errors. This adapted-input
study cannot attribute differences solely to the release architecture.

## Full TRAIN extraction admission and resume validation

2026-09-17 header census:7096 videos827,354frames,23 videos>300frames,max475;
none missing/empty. Frame-scaled DEV throughput projects5793s, not a guarantee.
Allow6300s hard timeout for full TRAIN within the original2h baseline tranche;
no extension to method or confirmation budget. Expected output around3GiB.

Before long launch, run first2 TRAIN with a clean pause after1, then resume to2;
verify no recomputation of first video and numeric parity against earlier smoke
(FP32 maxabs<=1e-4, identical frame/window indices). CPU tests cover exclusive
lock, checksum/recipe mismatch rejection and preservation of orphaned partial
files. Resume requires unchanged code/model/manifest/batch/recipe contract,
checks raw and output hashes for each complete video, and never replaces release
assets. Files from an interrupted item without committed metadata are moved to
the run's `orphaned/` directory before recomputation. This is recoverable local
artifact handling, not raw-data deletion. No extractor/model recipe changes.
