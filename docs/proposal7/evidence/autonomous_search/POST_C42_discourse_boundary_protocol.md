# Post-C42 discourse-context boundary audit

Material Passport: academic-research-suite / deep-research / three-way-scan;
2026-09-15; ANALYZED; AI-assisted, inline roles, no human-read attestation.

This is a pre-candidate feasibility/closure audit, not AS-C43 and not a pilot.
Question: can the current PH sentence-level inputs identify external discourse
context well enough to justify a new, independent-query mechanism?

Before running the metadata census, lock these boundaries:

- Read PH train/dev manifests, corresponding forensic JSONL, and every referenced
  sentence temporal JSON. No test, video, feature, checkpoint or score access.
- Record all top-level schemas, input hashes, temporal coordinate scopes, source
  path identity, inferred filename-prefix counts and numeric-predecessor presence.
- Independently enumerate predecessor counts by direct string membership. A
  suffix predecessor is NOT a verified temporally adjacent sentence; a missing
  predecessor is only absent from inspected train/dev, not absent from the corpus.
- Decision consequence: if verified recording/time linkage is not present in this
  metadata, do not concatenate inferred neighbors, their captions or cross-split
  rows into a scorer. First require task-compatible lineage; no linkage rescue
  sweep or dataset replacement follows from this audit.
- Even if linkage exists, require a local linguistic/retrieval signature and an
  internal/external novelty screen before any method candidate. Generic context
  concatenation, RPCA generation/protection, nuisance removal, and global
  composition/order remain unavailable as fresh central contributions.

Literature scan, begun before census: general English queries on discourse,
coreference, cross-sentence context, PH segmentation, and sign-language retrieval;
no private corpus text sent externally. Include primary papers/documentation that
establish context dependence, existing mechanisms, contrary evidence, or the data
contract. Exclude secondary summaries as evidence. This is a targeted collision
scan, not PRISMA/systematic coverage and not proof of novelty by search absence.
