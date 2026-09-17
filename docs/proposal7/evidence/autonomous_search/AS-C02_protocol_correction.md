# Raw provenance correction, before retrieval outcomes

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run/validate
- Origin Date: 2026-09-14
- Verification Status: ANALYZED
- Version Label: raw-batch-parity-v1

The first raw smoke (`AS-C02-RAW-SMOKE_run.json`) failed the fixed relative
feature-parity threshold 1e-4. Its traceback and original source hash are retained.
There was no raw retrieval training or dev selection before the repair.

`raw_parity_diagnostic.py` re-extracted the same permitted RGB with the same I3D
checkpoint and image preprocessing. Selecting eight windows before execution
changed batch shapes from historical batches of 32. Full historical batches
reproduced the stored pooled vectors exactly; subset batches and TF32-off did not.
This isolates a numerical extraction dependency, not a visual-information finding.

Correction: execute all historical windows in batches of 32, then retain eight
uniformly selected windows' 2x2 spatial maps and corresponding pooled vectors.
The feature-parity gate was NOT lowered. The corrected three-video smoke passed
with zero pooled error for all three videos. Both raw spatial and pooled controls
share this full extraction cost, which must be included in resource reporting.

Pretraining remains the exact baseline agnostic I3D checkpoint. No SEDS asset,
stronger encoder, test split, changed positive relation, or new dataset is involved.
The numerical finding is an implementation/provenance repair, not a candidate.
