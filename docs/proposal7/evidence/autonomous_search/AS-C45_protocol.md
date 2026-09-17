# AS-C45: exact-gloss collision and persistent-confuser screen

2026-09-15. academic-research-suite, annotation/score audit. Registered before
the census. Not a new method, relevance annotation or gloss-supervised pilot.

Question: are exact official gloss sequences insufficient to distinguish a
material share of existing persistent DEV retrieval confusers?

Use standard official TRAIN/DEV CSVs and corresponding existing manifests.
Canonical gloss is the exact tuple of whitespace-separated tokens: no synonym,
edit-distance, morphology, lemma, case or manually inferred equivalence mapping.
Inventory repeated gloss sequences and pairs whose native translations differ;
separately require model-input translations to differ. Never call either pair a
linguistic minimal contrast or an equivalent-positive pair.

Load only the three existing historical DEV score matrices, not models/features
or new encodings. Preserve row order by exact CSV-to-manifest identity joins.
For each direction and original persistent mask, measure: availability of an
exact-gloss/different-native-and-model-text candidate, a strictly higher-scored
such candidate at each seed, higher-scored candidates at all seeds, and the same
strict competitor shared across seeds. Report scores/ties as diagnostics, not
replacement official recalls. Enumerate all unordered exact-gloss pairs via both
grouping and an independent all-pairs comparison.

Registered material-burden lead: at least10% of the original persistent queries
in BOTH directions have a strict same-gloss/different-text confuser in ALL three
seeds. This is a descriptive prioritization threshold, not a significance test,
novelty verdict or GO gate. No tuning/relaxing this signature after seeing it.

If supported, require source-verified interpretation of gloss omissions and
distinct open candidate consequences before a pilot. If unsupported, stop the
exact-gloss persistent-confuser route: no fuzzy-gloss/miner/group-positive rescue.
The native pixel pipeline is not gloss-only; a gloss collision cannot establish
its information ceiling. Existing equivalence/soft-positive and lexical-support
families remain closed regardless of this result. No changed positives or labels.

Output AS-C45-GLOSS_run.json; refuse overwrite; hash all inputs, protocol and
implementation. No TEST access, checkpoint loading, training, acquisition or
annotation creation. All work local and read-only except the audit artifact.
