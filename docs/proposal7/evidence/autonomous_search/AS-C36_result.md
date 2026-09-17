## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (complete deterministic inventory; no independent rerun)
- Version Label: AS-C36-v1

## Finding

All7615original TRAIN/DEV rows have distinct deployed visual inputs at BOTH
FP32loader and nativeFP16encoder boundaries. No within-TRAIN, within-DEV or
cross-split exact input collision. This rejects exact whole-input collapse
as an explanation in these artifacts, not broader information loss or semantic
ambiguity. No new method or GO follows from a negative identity inventory.

| Boundary |Distinct inputs|Within TRAIN collision pairs|Within DEV pairs|Cross-split pairs|
|---|---:|---:|---:|---:|
|FP32fused/sampled h + valid|7615|0|0|0|
|FP16cast h + valid|7615|0|0|0|

No repeated aware-file path or agnostic-file path in any of those scopes either.
Different paths alone would not establish different contents; the actual
preprocessed tensors were hashed. Both stream files for every row were freshly
hashed,15230file-read operations. Those per-file digests are provenance
snapshots, not a separate claim that every dense stream file is unique or that
the underlying raw footage is disjoint.

## Exact scope and integrity

Original7096TRAIN/519DEV manifests, feature_len64,alpha.9 (90%agnostic),
original uniform temporal selection and zero padding. Every full64x1024h tensor
and64bool validity mask included in dtype/shape-delimited SHA256. Signed zeros
canonicalized to match numerical equality. Dense indexes and paths NOT included
in the input hash because the encoder does not receive them. Valid masks ARE
included because equal padded tensors can otherwise represent different inputs.

R0checkpoint hash checked, native conv1 dtype confirmedFP16. The local
encode_image function casts visual inputs to that dtype; the inventory therefore
checks the cast, not just file storage precision. All rows finite in both
precisions, each with at least one valid frame. Zero all-zero-valid-input rows,
zero rows with nonzero padding, both splits. Hash groups had no collisions,
so zero reload/equality comparisons were needed; this is not presented as an
independent encoder replay or a successful nonvacuous collision check.

| Split |Rows|Full64valid slots|Fewer than64|Exactly1valid slot|
|---|---:|---:|---:|---:|
|TRAIN|7096|5524|1572|13|
|DEV|519|381|138|1|

Full valid-length histograms retained in AS-C36-VISUAL-INPUT_run.json. Short
inputs are not automatically erroneous extractions or semantically inadequate:
source duration/extraction provenance would need separate examination.

## Execution

- AS-C36-VISUAL-INPUT_run.json completed exit0,10.355849seconds,CPU only.
- Protocol and two unit tests preceded execution.62focused tests pass.
- Hash tests cover mask changes, dtype changes, finite rejection, signed zero;
  grouping tests cover within/cross-split pair-count semantics.
- No crash, retry, timeout, external upload, encoder inference/training or test access.
- One run JSON with7615row records, original source/file hashes, no large cache.
- No source file, manifest, checkpoint or label changed/deleted.

## Fallacy scan:11/11 checked

1. Simpson: TRAIN/DEV/cross-split identities counted separately, no pooled
   rate hides a subgroup collision.
2. Ecological: tensor uniqueness not signer, semantic or raw-footage uniqueness.
3. Berkson: complete declared TRAIN/DEV census, not an error-selected subset;
   conclusions limited to those manifests, not outside populations.
4. Collider: no outcome-conditioned fitting or adjusted causal inference.
5. Base rate: all7615rows, two precision boundaries and split counts explicit.
6. Regression to mean: no improvement or pre/post retrieval claim.
7. Survivorship: all rows successfully read and checked; none silently dropped.
8. Look-elsewhere: no similarity thresholds, favorable hash variant or new
   group definition selected after results.
9. Forking paths: original loader/precision/masks fixed; no near-duplicate
   rescue sweep after zero exact collisions.
10. Correlation/causation: no exact collisions does not prove inputs contain
    the needed linguistic distinctions or explain persistent retrieval errors.
11. Reverse causality: tensor equality determined independently of ranks;
    failures were not used to declare distinct inputs equivalent.

## Consequence and next question

Exact whole-video input collision is unsupported. The earlier filename-prefix
overlap cannot now be described as exact fused-input duplication, and neither
audit establishes raw-data leakage absence. No deduplication or changed positives.

The next bounded acquisition question is WITHIN-video temporal distinguishability:
do valid sampled feature slots repeat exactly or collapse at nativeFP16 precision,
despite whole clips being distinct? Exclude padded slots, separate true dense
length from retained length, and check any identified repeat against source
dense indexes/values before attributing it to sampling or extraction. Register
the inventory first; temporal repetition may reflect static content and is not
automatically a defect. No temporal sampler or closed alignment family is
reopened by this plan. No AS-C37 launched. After this second bounded check in
the layer, switch if no attributable mechanism is supported.

ARS influenced the preregistered exact boundaries and restrained interpretation.
Goal active: no GO, no global exhaustion, no Proposal8.
