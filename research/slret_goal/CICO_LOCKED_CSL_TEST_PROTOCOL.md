# CSL held-out-video numerical confirmation: fixed models and analysis

Registered 2026-09-19 before any CSL TEST retrieval scores. This extension was
chosen after the PH TEST result; it is NOT part of the original PH preregistration.
No method/training change is permitted. Input preparation is still running;
evaluation is conditional on its complete provenance/finiteness/parity checks.

Question: does the fixed240-step FP32-versus-native continuation effect replicate
on the official held-out CSL videos, and what happens under DEV model selection?
All models are already trained and frozen. No further fitting, caption repair,
filtering, relevance modification or tuning may follow this TEST for this claim.

## Models and estimands

Score all ten distinct models once: shared historical selected initialization;
native and FP32 final240 checkpoints for seeds42/1337/2026; native42 selected36,
FP32 1337 selected228 and FP32 2026 selected216. Other selected models map to
initialization (FP32 42, native1337/2026). No best-TEST checkpoint or seed.

Primary estimand: within each seed, FP32minusnative fixed-endpoint mean of T2V
and V2T R1; report each direction, mean±sampleSD over three continuation seeds.
Mechanism replication gate: mean>=1pp,>=2/3 positive seeds, both directional
means>=0, conditional paired cluster95% interval excludes0. This gate does not
establish a new best model or publication novelty. Selected-checkpoint contrasts
and each model versus initialization are mandatory descriptive secondary results;
no selective significance tests. Earlier selected-accuracy pilot gates failed.
All models' R1/R5/R10/MedianR/MeanR must be reported, including trade-offs.

## Data and relevance

Use only completed `csl-test-assets-001/csl_test.jsonl`:1176 unique videos,
798 caption groups, English query translations with the pinned DEV translator,
agnosticBSL5K/H2S-transfer-aware features, alpha.8, len64, maxwords32, evalbatch128,
scoreblocks128. Feature roots change only to the verified TEST inputs. Preserve
all other training config and native dtype/scorer. Hash assets, input-preparation
report, evaluator/model/tokenizer/feature code, initial/final/selected checkpoints,
source training/audit reports and this protocol before scoring.

Unchanged shared grouped evaluator `id_multi_positive_best_rank`: one text per
caption_id, all associated videos positive; best-positive optimistic rank in
both directions (not PH singleton tie expansion). Original first-seen video/text
order retained. Do not merge different caption IDs whose English translations
happen to match. Audit score/rank recomputation and ID/positive-mapping hashes.

795/798 TEST caption groups overlap DEV; zero TRAIN caption groups and zero
TRAIN/DEV video IDs overlap. This is held-out-video evidence, not unseen-caption,
source-disjoint or signer-disjoint evidence. Historical pretraining/checkpoint
selection exposure is incompletely known. Do not claim global clean TEST.

## Paired uncertainty

Bootstrap798 caption-ID clusters,10000 multinomial draws, analysisseed20260919.
The same cluster weights apply to every seed and arm and both directions.
Within each sampled group T2V contributes one query, V2T contributes all its
videos; compute each direction's success/denominator ratio then average directions
and seed contrasts. Gallery remains fixed; no candidate resampling/re-ranking.
Report95% percentile interval and cluster sizes. This is conditional on trained
models and fixed gallery, not a population CI for seeds, signers or novel captions.
Caption clustering captures repeated-content groups, not all signer dependence.
Seed SD and query-cluster uncertainty remain distinct. No pooled PH/CSL score.

## Gates and retention

Lock creation must fail if any preparation/source/model evidence is missing or
drifts, model count differs, step/selector differs, or config changes outside
TEST manifest/feature location and provenance descriptions. Evaluation requires
the explicit expected lockSHA and a fresh output directory. On any failure keep
all results but do not use a survivor-only summary; no automatic retries.

Evaluation<=300s GPU,<=128MiB outputs,44GiB campaigncap/15GiBfree reserve;
charge confirmation6h reserve after actual input-preparation runtime is known.
CPU analysis<=180s. No score launch while extraction is active. This protocol
registers a conditional evaluation, not a successful confirmation.
