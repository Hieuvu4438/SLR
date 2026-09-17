# Existing human TRAIN annotation join: preregistered resource check

2026-09-16. New lead, not a model experiment. DFKI-SignLanguage/sacre-bird-phoenix
commit012b11c22b1c64b79325ec6b004b81693db86187 advertises307human-reviewed TRAIN
segments. Verify whether this resource joins the existing permitted TRAIN IDs.

Fetch exactly `train_annotations_sacrebirdphoenix.csv` at that pin. Never fetch
either TEST CSV, repository archive or back-translations. Read existing PH TRAIN
and DEV manifest identifiers/split fields only; do not inspect model scores.
Record schema, row/unique-ID counts, duplicate IDs, exact TRAIN matches, DEV
overlap, non-TRAIN IDs, category value counts, and missing values. No fuzzy join,
ID correction, field recoding, semantic interpretation of free-text comments,
new annotation, changed positive, filter, weight or training update.

Do not print or retain source comment text; the fetched file is hashed and used
in memory. Reviewers' flags concern gloss/German-text consistency per README,
not necessarily signed-video content or English model-caption fidelity. No
population prevalence or DEV-error burden may be inferred from this selected
sample. A successful join changes resource feasibility, not model GO.

Failure policy: retain actual schema/count discrepancy; no silently dropping
duplicates/unmatched rows. Before any subsequent semantic or score analysis,
verify TRAIN sampling and annotation procedure from primary methodology. Existing
soft-positive, noise/reliability, lexical repair and group-risk closures remain.
