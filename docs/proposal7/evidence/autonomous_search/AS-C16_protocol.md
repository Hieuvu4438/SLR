# AS-C16 — natural same-lexicon contrast availability

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / supervision-adequacy audit
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C16-v1

Q09: do the existing PH train/dev captions supply natural pairs with the same
lexical multiset but different order, suitable for distinguishing lexical from
relational information? Availability is necessary, not sufficient: paraphrases,
enumeration order, translation artifacts and truncation can invalidate a pair.
No new benchmark, generated caption, new positive, external model or training.

Two fixed keys, separately reported:

- Lowercase Unicode word sequences (regex word runs, internal apostrophes kept;
  punctuation excluded), grouped by SORTED MULTISET, preserving repetitions.
  Keep only groups with different ordered word sequences.
- Exact deployed32-slot BPE ID sequences, grouped by sorted valid CONTENT IDs
  (exclude SOT/EOT/padding), keeping groups with distinct ordered content IDs.
  Uses actual uniform truncation, not first30. BPE bags can merge different
  word segmentations; do not equate them with whole-word lexical equality.

All groups and every within-group unordered pair with different defining
ordered sequences are retained. Mark whether deployed inputs differ. Count
rows/groups/pairs and source-prefix diversity. Grouping uses only existing
captions, never retrieval labels. Train and dev processed separately.

For the union of collected pairs with different deployed inputs, inspect frozen
R0 scores for both positives and crossed pairs. Record four directional paired
margins with strict threshold1e-4 (the established train-support numerical band).
These are pairwise diagnostics, NOT full-gallery R1 or a method pilot. For dev,
reuse full-gallery saved matrices and attach original ranks. Train uses the
existing independently checked paired scoring kernel. No sample exclusion by
score and no learned probe from these candidates before validity review.

Review candidate text pairs after the inventory, explicitly separate textual
interpretation from signed-video validity; neither raw sign annotation nor
expert minimal-pair validation is supplied by this automated audit.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.lexical_contrast_audit`
from `/home/haipd/SLR`, timeout30min, monitor session≤60s. Run JSON and inventory
are the outputs; preserve failures and refuse overwriting a prior run. No testsplit.
