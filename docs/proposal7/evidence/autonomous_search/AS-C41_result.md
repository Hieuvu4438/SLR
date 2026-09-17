## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (independent inventory enumeration agrees; scores not rerun)
- Version Label: AS-C41-v1

## Result: no supported natural one-word contrast failure

The registered bilingual one-word-substitution inventory contains 396 TRAIN
pairs and no DEV pairs. Every TRAIN pair has all four directional margins
strictly positive. These observations do not establish a retrieval bottleneck,
a method candidate, or GO. In particular, the empty DEV inventory is an absence
of eligible observations, not evidence that the model generalizes to these
contrasts successfully.

| Split | Rows | English one-substitution pairs | German pairs | Bilingual intersection | Distinct involved rows | Joint templates |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 7096 | 554 | 1399 | 396 | 386 | 270 |
| DEV | 519 | 0 | 9 | 0 | 0 | 0 |

All 396 bilingual TRAIN pairs have different complete deployed text inputs;
none was excluded by that rule. There are 0 same-source and 78 same-signer
pairs. These metadata are descriptive, not verified causal or linguistic
controls. Pairs share clips and templates, so 396 is not an independent sample
size. The largest joint template contains 10 pairs.

## What the scores establish

The pinned PH-fitted seed42 baseline gives 1584 positive directional margins,
0 near-zero and 0 negative margins, using the preregistered 1e-4 tolerance.
The minimum is 1.2682914734 and maximum 18.0315093994. These are four-cell
pairwise comparisons, not a full TRAIN-gallery retrieval evaluation. The
baseline has already been fitted on PH; TRAIN separation is not held-out
linguistic evidence.

The weakest pair illustrates a validity limitation: English “begins” versus
“starts,” corroborated by German “beginnt” versus “startet,” in otherwise
matching weather sentences. A one-word edit in each language can be a
near-synonym, not a change in meaning. Caption edits do not establish different
signed realizations of a controlled semantic opposition. No expert minimal-pair
labels were created or inferred.

For each of the three original DEV models and the fixed AS-C32 ensemble,
inventory coverage of persistent strongest-confuser queries is 0/92 T2V and
0/87 V2T. This follows from having no eligible DEV pairs. Paired-kernel numerical
validation also has 0 pairs and 0 scalars, with max error null: **no numerical
kernel comparison was possible**, rather than a passing parity test.

## Independent inventory validation

The main algorithm uses masked-word templates. A separate validator enumerates
all same-length unordered sentence pairs and counts differing positions,
stopping at the second difference. It uses the same registered word
normalization, not an independently validated linguistic tokenizer.

It inspected 1,191,132 English and 1,335,028 German same-length TRAIN pairs,
and 7,073 English and 8,324 German DEV pairs. Counts, bilingual pair IDs and
both positional indices agree exactly with the main inventory. The validator
checks manifest hashes and pins the parent run. It does not rerun encoders,
pair scores, metadata interpretation, or downstream retrieval metrics.

- Experiment: `AS-C41-SINGLE-WORD_run.json`, completed, exit 0;
  2.457207 seconds, peak 61,294,080 GPU bytes.
- Parent result SHA256:
  `608c17b0ddc1cc7ed1f20c55638e48756f3f96614c9475f47685e92a84e5639b`.
- Independent census: `AS-C41-INVENTORY-VALIDATION.json`, completed,
  0.598492 seconds. Process is terminal; no restart is needed.
- Protocol: `AS-C41_protocol.md`, written before outcome counts.
- Full `methods/information_probe` test suite: 67 passed in 1.76 seconds at
  finalization. Parent-result digest rechecked against the independent census.
  Tests validate implementation behavior, not linguistic validity or method GO.
- No training, test access, new captions, new labels, uploads, or changed
  evaluation. Source/cache/manifest/forensics/checkpoint provenance is retained
  in the run. No experiment failure or retry.

## Statistical scope: 11/11 fallacies checked

This is a finite descriptive census under a fixed rule, not a population
significance test. No p-values or confidence intervals are claimed.

| Check | Assessment |
|---|---|
| Simpson's paradox | TRAIN and DEV are separate; no pooled generalization rate. |
| Ecological fallacy | Pair/template counts do not imply individual linguistic capability. |
| Berkson selection | Exact edit and bilingual filters select a narrow subset, explicitly retained as a limitation. |
| Collider bias | Matching captions is not causal adjustment; no signer/content causal estimate. |
| Base-rate neglect | Full row counts and original persistent denominators reported; empty coverage not disguised as accuracy. |
| Regression to mean | No selected-error pre/post improvement or trained intervention. |
| Survivorship bias | All registered pairs and zero-pair DEV results retained; no score-dependent exclusions. |
| Look-elsewhere | Exploratory autonomous search context retained; no confirmatory or search-wide inference. |
| Forking paths | Fixed normalization and edit rule; no post-result edit-distance, synonym, or native-only rescue. |
| Correlation versus causation | Wording corroboration does not establish signed semantic contrast or a causal bottleneck. |
| Reverse causality | No directional causal claim from metadata or pair separation. |

## Consequence

Close this narrow explanatory lead. Do not loosen the edit rule to manufacture
DEV support, promote TRAIN separation as generalization, or revive ELSC/PLEL/
SSSC or native-caption preservation. Broader linguistic sensitivity remains
unresolved. Move to a materially different question; no global exhaustion or
Proposal 8 is established.
