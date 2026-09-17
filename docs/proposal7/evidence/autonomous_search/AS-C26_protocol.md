# AS-C26 — common-reference lexical/source support

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C26-v1

After two failed scorer cycles, inspect generalization/data support. Does the
AS-C20 source-held population have substantially different observable lexical
support from official dev? Compare1375held TRAIN rows and519DEV rows against
the SAME5721fit TRAIN reference. Secondary dev/all7096TRAIN reference separates
reference-size effects. No new split, fitting, scorer change, targets or test.

Input-only descriptors: complete BPE-token-set Jaccard nearest neighbor;
complete adjacent-BPE-bigram-set Jaccard nearest neighbor; exact deployed32-token
sequence match; count of reference rows sharing inferred video-ID source prefix;
complete token length. Also nearest Jaccard after excluding same-source references
for both ngram types. Prefix identity is inferred, not independently verified.
No weighting, semantic model, threshold tuning or labels used in nearest search.
Neighbor ties pick first reference row. Save every row, neighbor and descriptor.

Compare distributions descriptively (mean and0/.25/.5/.75/1quantiles). No causal
inference, p-value or equivalence claim. For support/error association, use fixed
unigram-neighbor bins [0,.25),[.25,.5),[.5,.75),[.75,1),{1}; report counts and
per-query R1 for AS-C20 final-held and historical R0dev WITHIN each separate
regime. These cross-regime accuracies are not matched: models, initialization,
training data and gallery sizes differ. T2V uses optimistic per-query ranks,
not primary tie-expanded aggregate. Do not explain R0's advantage from this
descriptive comparison or lower AS-C20's learning-adequacy threshold.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.lexical_support_probe`
Cwd `/home/haipd/SLR`; timeout300s; process plus`AS-C26-SUPPORT_run.json`.
Sparse binary incidence multiplication computes exact set intersections; small
synthetic test compares with Python sets and checks same-source exclusion.
No large cache/model saved. Failure must be preserved, not silently retried.
