# RTMPose reconstruction diagnostic — registered before inference

2026-09-17. Baseline preprocessing diagnostic, not a method pilot. Use the first
eight SEDS PH TRAIN IDs and each video's first/middle/last original PNG frame.
No dev or test content is used. Compare to corresponding release keypoints.

Two bounded extractor hypotheses: official RTMPose-L COCO-WholeBody 384x288 and
256x192 checkpoints from the local OpenMMLab model registry. Full-image bbox,
native MMPose affine preprocessing, no flip augmentation (single-image recipe).
Do not fit coordinates or confidence scales to release outputs. Record model,
configuration, input-frame and reference hashes, environment and timings.

Exact reconstruction gate: all coordinate errors <=0.01 px and confidence
errors <=1e-4 over every compared joint/frame. A near match is descriptive only,
not release parity. Also report SEDS-relevant body/hand errors and confidence
threshold decision disagreements. If neither passes, stop extractor guessing:
provenance remains unresolved and any reconstructed dev must be labeled adapted.
If one passes, validate additional held-out TRAIN videos before extracting dev.
This does not establish RGB/I3D preprocessing parity by itself.

## Follow-on RGB cache comparison (registered before comparison)

Fixed same eight TRAIN videos, no new GPU inference. Derive retained frame names
and clip starts with the native SEDS loader. Compare release RGB windows only
when the complete 16-frame sequence is contiguous in the local CiCo raw-frame
cache. Report missing alignment separately. Feature parity gate max absolute
error <=1e-4; cosine is descriptive, not an equivalence gate. Failure means the
existing CiCo cache cannot silently stand in for SEDS RGB. No parameter search.

Budget: two public checkpoints, <=500 MiB download; <=10 GPU minutes total;
hard timeout 15 minutes for diagnostic including downloads. No training or
retrieval-based selection. Retain all outputs and failed attempts in ledger.
