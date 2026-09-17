## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C36-v1

## Exact visual-input availability question

Do distinct rows become identical visual inputs under deployed preprocessing?
Source-prefix overlap is not input identity; repeated captions are not repeated
videos. Audit original PH7096TRAIN/519DEV ONLY. No new positives, deleted rows,
benchmark, near-neighbor threshold, model fitting or method claim.

Use original CiCoFeatureDataset(feature_len64,alpha.9) for both splits: float32
.1aware+.9agnostic fusion, uniform sampling, zero padding and bool validity mask.
Hash the complete64x1024h AND64valid array, including dtype/shape delimiters.
Canonicalize signed zero so equal-valued finite tensors have equal hashes.
Check all features finite, some frames valid, and describe valid-length
histogram/all-zero valid input/nonzero padding. Require finite after native cast.

Audit two boundaries: FP32loader output and FP16visual input. Read pinned R0
checkpoint's conv1 dtype and verify its hash; local encode_image explicitly
casts to visual dtype. No encoder forward or approximation of its output.
FP16quantization can in principle merge unequal FP32inputs; report separately.

Record one digest pair per row and rehash both aware/agnostic source files
(15230file reads). Same-file paths also inventoried separately. Original
manifests and loader/view/source hashes retained. These are snapshots of current
files, not proof their raw RGB extraction is correct or cross-corpus disjoint.

Group hashes across all7615rows, separate withinTRAIN, withinDEV and cross-split
pair counts. For EVERY collision, reload rows and require exact torch.equal
of h at the relevant dtype AND valid mask. If no collisions, do not run vacuous
pair checks and call them a reproduced encoding: report zero such checks.

Any verified input collision is a diagnostic lead requiring caption/feature
provenance follow-up; zero collisions rejects this exact-collapse explanation,
not near-duplicate video overlap, semantic ambiguity, lost fine information,
or sufficiency of frozen features. No aggregate information ceiling is inferred.

## Execution

    PYTHONPATH=shared:. timeout 300 /home/haipd/miniconda3/bin/python -m methods.information_probe.visual_input_inventory

Output AS-C36-VISUAL-INPUT_run.json with all row digests and scoped collision
groups. No overwrite. Expected seconds to tens of seconds,CPU only,300s timeout,
progress every1000rows plus process/output monitoring30–60s. Unit tests include
mask-sensitive hashes,signed zero,dtypes,nonfinite rejection and split counts.
No test split or external upload; all data untouched. Full11/11 fallacy scan
and limitations after completion; goal/GO requirements unchanged.
