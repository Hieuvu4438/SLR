# User-priority next direction: Hand4Whole++ estimated 3D

2026-09-19, academic-research-suite / scoped experiment preparation, inline.
Explicit user request: after unsuccessful C07, investigate/apply pose3D from
`/home/haipd/DexAvatar/Hand4Whole-plus-plus_RELEASE`; other methods remain
available if it has no promise. This supersedes C08 priority, not budget or
TEST restrictions. Do not automatically queue extraction after current C07.

## What is actually established

Update2026-09-19 (supersedes historical runtime bullets below): user completed
h4wpp installation. Existing sample001 completed24 finite frames in70.350850s;
inference1.542257s, peak4509579264bytes. No runtime reinstall or repeat of that
same sample. Current scope is checking the observed crop-dependent placement
issue, then extracting geometry for added-XY/XYZ retrieval experiments.

- Local README/main/model.py/config.py/run_inference.py and joint definitions
  inspected read-only. No edits in DexAvatar, no GPU model load yet.
- Main checkpoint `output/model_dump/snapshot_6.pth` exists (~3.0G).
  WiLoR `pretrained_models/wilor_final.ckpt` and `detector.pt` paths exist.
  This does NOT certify all dependencies, checkpoint-key coverage, or inference.
- `main/model.py` test returns `smplx_kpt_cam` AFTER pelvis subtraction, while
  `smplx_kpt_proj` is projected BEFORE subtraction, after hand/body combination.
  Do not project the root-relative output directly using camera intrinsics.
- Keypoint names support wrist +20 finger joints on each hand and SEDS body
  order Nose,L_Shoulder,L_Elbow,L_Wrist,R_Shoulder,R_Elbow,R_Wrist. Use names,
  not presumed COCO indices. MANO wrist is separate from the20 finger range.
- Projection is in cfg.vit_output_shape coordinates (16x12), not original
  image pixels. Convert through body/crop scale + bb2img affine if needed.
- Input512x384; cfg uses virtual intrinsics. Treat 3D as monocular estimates,
  not metric ground truth or an independent sensor.
- Existing run_inference renders meshes and writes OBJ; retrieval extraction
  needs only aligned joints/projections/frame IDs, not per-frame mesh storage.
  Implement an SLR-owned lightweight wrapper, preserving the other repo.
- The documented conda name h4wpp is not in the inspected miniconda env list;
  existing dexavatar/wilor/etc may be the configured runtime. Resolve actual
  working interpreter before imports; don't reinstall/alter working environments.

Original [paper](https://arxiv.org/abs/2603.14726), abstract checked2026-09-19:
Hand4Whole++ combines pretrained body/hand estimation with conditional hand
modulation and alignment. No retrieval improvement is established by this paper
or by the local source inspection. No human-read/full-paper-review claim.

## Smallest useful experiment before full extraction

After C07 result/user handoff, resolve runtime and explicitly load local weights
with missing/unexpected trainable-key checks (existing demo uses strict=False).
Read PH TRAIN raw-video paths from existing dataset guide/manifests; use an
outcome-independent fixed small TRAIN sample. Measure seconds/frame, VRAM,
finite outputs, hand identity, temporal jitter and projected overlay quality.
No TEST samples. Visual checks are AI inspection, not linguistic ground truth.
Keep native temporal frame mapping; don't interpolate sparse predictions and
claim exact framewise3D. Register any stride/interpolation in all control arms.
Estimate total extraction cost before committing remaining~10092s compute or
storage. No full-corpus extraction until this bounded feasibility run is usable.

## Retrieval contrast to register

Retain original RGB+pose2D baseline and add a checkpoint-compatible geometric
input pathway, initially zero-output so reference behavior remains available.
Actual architecture will be fixed after sample output inspection, not claimed
implemented here. No generic confidence gate or pseudo-label semantic claims.

Compare: (A) original SEDS; (B) same-capacity geometry pathway fed H4W XY with
Z zeroed; (C) identical pathway fed H4W XYZ. Same H4W samples/joint identities,
initialization, trainable groups, loss, schedule and selection. Use the SAME
XY-derived normalization scale in B/C; don't let 3D norm leak Z into the XY
control. Preserve original2D inputs in both arms. H4W's inferred XY itself
already reflects its3D prior: B/C isolates use of explicit Z, not all benefits
of3D pretraining. Projected2D is a further detector comparison, not identical
to camera-relative XY. Report these distinctions.

Use hand wrist-relative geometry plus body-relative wrist placement so local
hand normalization does not discard articulation location. Avoid silently
canonicalizing away palm orientation. Mapping, units and missing-frame policy
must be explicit. No use of TEST error patterns to choose representation.

Judge full DEV519 mean bidirectionalR1 + both directionsR1/5/10 against original
77.552987 and provisional incumbent77.938343; compare B/C before attributing
gain to depth. Pilot improvement is not SOTA: replication and second-dataset
generalization follow only if promising. Background/log/yield user rule applies.

## Implementation / current bounded run

User2026-09-19 reiterated: DO NOT remove pose2D; additional3D architecture is
delegated to agent. Implemented independent hand/body geometric MLP + temporal
convolution branch in methods/seds_adaptation/pose3d_branch.py, with a zero
output projection added to the native pose representation before CGAF. The
original2D encoder/input and RGB stay present; no frozen baseline tensor is
replaced. Geometry windows must match native clip windows exactly. This is not
yet connected to a real3D dataset loader/trainer; extraction is prerequisite.
Three CPU tests passed0.88s; no GPU retrieval improvement claimed.

User explicitly confirmed deleted h4wpp and requested recreation/downloads.
Existing dexavatar matches historical torch2.1.1+cu121, torchvision0.16.1,
PyTorch3D0.7.5, numpy1.26.3, smplx0.1.28 and timm1.0.25. Clone with --copy into
new h4wpp (refuse existing prefix), then install missing H4W dependencies and
MMCV2.1.0 official cu121/torch2.1 wheel. Do not modify dexavatar or seds.
Pins derive from saved successful h4wpp environment; this is a focused runtime
reconstruction, not exact restoration of every old unrelated simulation package.
Official MMCV wheel listing was checked2026-09-19; actual binary compatibility
must pass imports. No expensive source build fallback or automatic retries.

Bounded bootstrap_h4w.py runs setup then extract_h4w_sample.py, only first3
canonical TRAIN videos ×8 consecutive center frames =24 frames, no TEST.
Full-frame person crop is explicit (PH single-signer video), same H4W hand
detector and native forward. Save joints/projections/frame IDs and3overlays;
no mesh OBJ export. Native model still computes mesh smoothing in this initial
speed measurement. Model load fails on unexplained missing core/unexpected keys.
Report all missing external-prefix keys separately; local WiLoR/DWPose models
initialize from their own checkpoints. No full checkpoint checksum rerun needed.
Total hard2000s; persistent reports/logs; no follow-on training/extraction queue.

## Native person-crop sample — registered2026-09-19

Material Passport: academic-research-suite / experiment-agent / run;
date2026-09-19; verification status UNVERIFIED for new run; version crop_sample_v1.

- ID h4w-ph-train-person-sample-002, ETL feasibility, not retrieval training.
- Observation: sample001 projections visibly offset from hands/shoulders on
  first-frame overlays; first video has some joints outside the image. These
  are AI visual observations, not annotated pose error or ground truth.
- Projection formula matches native project_coord plus inverse crop transform.
  Native demo uses highest-confidence YOLO person, while sample001 used full
  image. Test that specific preprocessing difference before scaling extraction.
- Same first3 canonical TRAIN IDs, same8 centered consecutive frames per video,
  same H4W model/weights/joint mapping; no DEV/TEST lookup or result selection.
- New crop policy person: local demo/yolo11n.pt, highest-confidence person;
  explicit logged full-frame fallback if absent, native aspect padding1.25.
- Record package versions/interpreter, script hash, boxes, detection time,
  inference time, projections and XYZ. Existing sample001 remains untouched.
- Five focused CPU tests pass: crop policy/fallback, projection transform,
  zero-init original-stream preservation, geometry masks/gradient/name mapping.
- Expected~75–100s, hard240s; GPU estimate<6GiB, output<10MiB. Free disk60GiB,
  GPU idle before launch. Native-crop runtime is not estimated from model-only
  timing; detection/decode/load are accounted separately.
- Command: `/home/haipd/miniconda3/envs/h4wpp/bin/python -u research/slret_goal_v2/tools/extract_h4w_sample.py --run-id h4w-ph-train-person-sample-002 --crop-policy person`.
- Job logs: artifacts/slret_goal/jobs/v2-h4w-person-sample-002/console.log.
- Output: artifacts/slret_goal_v2/h4w-ph-train-person-sample-002/run.json.
- Stop at completion/timeout and yield, no auto queue. Inspect on user return;
  if placement remains imperfect, document limitation and select a bounded
  useful geometry pilot (e.g. local hand articulation), not endless pose audits.
- Resource reconciliation: failed runtime002 wall62.155916s and later sample001
  wall70.350850s are distinct; charge both conservatively, no double-count of
  failed parent's original child. Used12141.873517s/21492s; remaining9350.126483s.
  Reserve240s from tranche3remaining1875.510088s. No retrieval/SOTA claim.
