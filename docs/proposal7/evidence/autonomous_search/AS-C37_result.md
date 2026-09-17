## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (complete inventory, no independent rerun)
- Version Label: AS-C37-v1

## Finding

No valid temporal vector repeats within any of the7615original TRAIN/DEV
clips, at either FP32loader or nativeFP16input precision. Whole-clip identity
was tested separately in AS-C36; this census tests within-clip exact equality.
The exact temporal-collapse mechanism is unsupported in these artifacts.
Neither result establishes preservation of linguistic distinctions.

|Split|Rows|Rows with at least2valid slots|Valid slots|Eligible unordered within-clip pairs|Repeated pairs FP32/FP16|
|---|---:|---:|---:|---:|---:|
|TRAIN|7096|7083|417323|12577248|0/0|
|DEV|519|518|29816|888354|0/0|

Unique counts equal valid counts within each clip, summed in this table. This
is NOT a cross-clip slot-uniqueness claim. Padding excluded;13TRAIN and1DEV
single-slot clips remain in the row census but contribute no eligible pairs.
No excess slots, adjacent repeated pairs or FP16-only mergers. Grouping checks
equality without explicitly materializing all13,465,602eligible pairs.

## Provenance and source attribution

AS-C36 run pinned by SHA256
`81e3765c0daec4c73d0d8f548c39487c507bfa2c547896b591ec3c64d1bdf185`.
Both manifest hashes, original loader/view/visual-cast source hashes, every row
identity/view/length, both complete-input digests, and all15230stream-file hash
reads match AS-C36. The same feature_len64,alpha.9 canonical preprocessing and
FP16cast were used. No inference, optimizer step, test access or data changes.

There were zero repeat-bearing rows. Thus zero conditional dense-stream
reload/fusion checks and zero repeated-pair torch.equal verifications occurred.
The cause categories are empty, not demonstrated extraction correctness.
Routine loader reads and fresh file hashing DID occur for every row. Source
attribution code is covered by synthetic fixtures, not real positive examples.
Signed-zero canonicalization makes byte grouping consistent with finite numeric
equality. No near-duplicate, normalized-direction or semantic similarity claim.

## Execution and anomalies

Command registered in AS-C37_protocol.md; output AS-C37-TEMPORAL-SLOTS_run.json.
Completed exit0,18.840844seconds, CPU only; progress observed every1000TRAIN rows.
No crash, timeout or experiment retry. Original artifacts unchanged.

One PRE-RUN synthetic-test failure was preserved in the execution history:
`.1*9` and `.9*1` appeared identical when printed but differed in float32.
The fixture was replaced with exactly cancelling commutative products;
all3new tests passed before the experiment. This did not change the hypothesis,
grouping algorithm or data. Tests cover padding, signed zero, adjacent/nonadjacent
repeats, precision merger, dense-index reuse, source-fusion cancellation and
nonfinite rejection. No independent full-inventory rerun claimed.

Final validation:59tests passed in the complete methods/information_probe
directory (1.59s); compileall and git diff --check passed. Saved JSON row/group/
denominator consistency checks passed for both splits and precisions, including
uniqueness of selected dense indices. These are record checks, not a second
tensor census. Run JSON size is recorded by the validation command.

## Fallacy scan:11/11 checked

1. Simpson: TRAIN/DEV and two precisions reported separately.
2. Ecological: within-clip vector uniqueness is not semantic or raw-RGB uniqueness.
3. Berkson: complete declared manifests, not error-selected clips.
4. Collider: no outcome conditioning, fitted adjustment or causal inference.
5. Base rate: rows, single-slot exclusions from pair eligibility, valid slots
   and all eligible pair denominators explicit.
6. Regression to mean: no retrieval pre/post improvement claimed.
7. Survivorship: all7615rows checked; no silent drops or failed reads.
8. Look-elsewhere: two fixed precision boundaries, no similarity threshold sweep.
9. Forking paths: registered before run; synthetic fixture repair disclosed.
10. Correlation/causation: no exact equality does not prove semantic sufficiency;
    repetition, had it existed, would not itself prove retrieval damage.
11. Reverse causality: ranks/captions were not used to determine vector identity.

## Research consequence

AS-C36/C37 are two bounded negative acquisition checks. Switch to another layer;
do not rescue this branch by near-duplicate thresholds, sampler variants or
precision tuning. This is no novel candidate and no GO, not global exhaustion.
ARS influenced the preregistration, source pinning, anomaly disclosure and
the separation between artifact equality and linguistic interpretation.

Next selected question is layerJ numerical independent-query batching
stability: AS-C31 measured batch-shape-dependent text-token variation, while
retrieval-rank impact and AS-C32ensemble robustness remain unmeasured. Register
a fixed unchanged-precision tail-padding control against original replay;
no numerical sweep or novel-method claim. No AS-C38 launched in this turn.
