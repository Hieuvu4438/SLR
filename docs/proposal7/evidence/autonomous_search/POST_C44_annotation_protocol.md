# Post-C44 annotation compatibility audit

2026-09-15; academic-research-suite fact-check, not a method pilot or AS-C45.

Decision: can the published manual handshape/mouthing evaluation resources be
directly joined to permitted PH2014T TRAIN/DEV clips for a joint-cue diagnostic?
The local release README has now been read: it explicitly excludes the ECCV2014
manually annotated mouthing sequences from all sets, and documents changed
boundaries. This is a source statement, not a recomputed footage-overlap test.

Before further resource access, inventory the exact local release files:

- Root README and evaluation README (documentation only).
- Standard TRAIN and DEV corpus CSVs, and TRAIN complex-annotation CSV.
- Existing TRAIN/DEV manifests, solely for identity/translation consistency.

Record full-file hashes, schemas, row/identity counts, start/end value counts,
video-path naming agreement and manifest agreement. Cross-check CSV parsing with
independent line/field splitting if the bytes contain no quoted or embedded
delimiter/newline ambiguity. Do not infer a frame mapping from a filename.

No TEST CSV/pickle/gzip/video/features/labels, archive acquisition, model loading,
score computation, new annotation, parameter fitting or split alteration. Directory
entry names may be inspected to locate permitted files, but are not test evidence.
The two local annotation-only gzip files are not needed and will not be opened.

Output: POST_C44-ANNOTATION-COMPATIBILITY.json, refuse overwrite. No inferential
statistics or retrieval efficacy claim. Absent coordinates in inspected files do
not establish global resource absence. If direct manual joint supervision is
unsupported, retain the wider question but stop this specific resource-join route.
