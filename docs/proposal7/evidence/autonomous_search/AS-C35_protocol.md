## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C35-v1

## Question / scope

AS-C34 mismatched6 is1.156069pp below matched3 (conditional95% CI
[−2.190476,−.188324]). Is coordinate misalignment sufficient to explain a
material part of this penalty? A common orthogonal map applied to BOTH towers
of one model preserves its internal dot products/cosines, but can change scores
with another checkpoint's tower. This is a standard diagnostic, NOT novel
retrieval or teacher fitting. [Orthogonal Procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html)
fits such a map without scaling or translation. Contemporary
[Multi-Way Representation Alignment](https://arxiv.org/abs/2602.06205v2)
also explicitly investigates shared orthogonal spaces and retrieval geometry;
its abstract is a collision warning, not a reproduced method or full-paper audit.

## Locked anchors / fitting

Original selected models42/1337/2026, original native-precision TRAIN7096
representations. Each sample supplies TWO normalized512D anchors: mean of
unit-normalized valid video token vectors (then renormalized), and normalized
text EOT/CLS. Equal numbers of video/text anchors; no centering/scaling, no
alternative anchor/layer sweep. Reference is seed42, maps for1337and2026.
Fit Q=U V^T from SVD of X^T Y in float64 CPU; Q^T Q error<1e-10.
Fit14192anchors/model, checkpoint-native outputs,128batches. No DEV labels or
DEV vectors in map fitting. Maps fit same-input cross-model correspondences,
NOT video-to-caption positive/negative optimization. Historical checkpoints
were PH-fitted: this is NOT a clean out-of-fold model.

Three fixed conditions: identity(no rotation), aligned TRAIN targets, and
targets shifted by ONE TRAIN row jointly for the two modalities. Seed42 stays
identity in all conditions. Shift is a correspondence-negative control, not
a semantic null: neighbors may share captions/content. Same fitting operation
and dimensions. Report true-pair TRAIN/dev anchor cosines before/after.

Apply Q cast tofloat32 to every token and CLS of BOTH towers, including invalid
slots. Keep original masks/scales/score functions and source ownership from
AS-C34. Evaluate all9cells, fixed matched3/mismatched6/all9 means per condition.
No best-cell/weight/layer selection. Identity all9score matrices EXACT toAS-C34.
For rotated diagonals require max score error<=5e-5 and all ranks exact. These
checks protect the supposed internal-geometry control; failure is preserved,
not silently repaired by score rounding, mask/precision changes or replacement.

## Decision and costs

Coordinate-explanation lead: aligned mismatched6 improves>=.5pp over original
mismatched6, its conditional95% lower>0, and mean gain over shifted-target
mismatched6>=.5pp. This is not superiority to the strong matched ensemble,
novelty, GO, or an information-ceiling result. Failure closes this single
pooled-video/EOT orthogonal explanation, not all coordinate incompatibility.
Preserve the complete grid and margins; no alignment rescue sweep.

Scorer/encoder model counts stay3; each9-cell grid has9scoring passes, optional
maps add512x512multiplications per token. Additional TRAIN encoding and map
fitting costs explicit, not free single-model inference. Persist small maps and
36score matrices; no new checkpoint or large frozen cache. Original data intact.

## Execution / validation

    PYTHONPATH=shared:. timeout 300 /home/haipd/miniconda3/bin/python -m methods.information_probe.encoder_gauge_probe

Output AS-C35-GAUGE_run.json; CPU/GPU finite checks and process/output monitoring,
300s timeout, expected tens of seconds and<5GBGPU. Unit test known rotation
recovery and scorer invariance before run. Bootstrap helper uses explicitly
correct per-comparison scope;315inferred prefixes/10000draws/seed20260915/full
DEV519gallery fixed. No test access. No method candidate promoted by this alone.
