# CiCo raw-input padding exposure gate

2026-09-16. Bounded source-contract check, not a candidate or efficacy experiment.
Academic-research-suite / deep-research, inline source verification.

Hypothesis under inspection: short-input repeat-last padding or shifted final
windows alter the raw temporal inputs used by the current PH baseline. This is
different from downstream feature-slot masking, adaptation pseudo-clip merging,
and nearby-view sampling, but does not reopen those closed method families.

Inspect pinned upstream VideoDataset and the local deterministic I3D extractor.
Census only existing TRAIN temporal metadata, exactly 7,096 files. Do not follow
source-video paths, load feature arrays, checkpoints, DEV or TEST. Report source
hashes, complete metadata digest, frame-count minimum/maximum, counts below/equal
to 16 frames, recipe values, and discrepancies against stride-one 16-frame window
placement and clipped half-open receptive-field intervals.

Decision: zero short-input exposure and exact stride-one windows reject this
specific raw-input-padding explanation for this TRAIN corpus. Nonzero exposure
would establish input behavior only, not harm, semantic loss or novelty. Do not
choose a threshold/subgroup after inspection, change extraction, re-extract,
score retrieval, or tune sampling/padding. Internal convolution zero padding and
decoder failures are separate questions, not measured by this metadata census.
No inference about historical pretraining/adaptation inputs or author features.
