# CSL exact text-input collision census — registered before counts

2026-09-23, Cycle18. Scope: existing CiCo CSL TRAIN/DEV group index and its
source manifests. No TEST, model weights/forward, features, translation calls,
human review, artificial semantic labels or changes to relevance.

Question: does exact equality of deployed text inputs merge a substantial
fraction of distinct existing CSL sentence groups? This is an input-identity
question, not whether their videos have different meanings. Earlier PH-only
collision results do not answer the CSL question. The initially considered
within-group performance decomposition already exists in PMGR's
`weak_performance.json`; do not repeat it or reopen population-risk training.

Hypothesis: exact text collapse is a material obstacle for the current CSL
representation contract. Prediction: at least10% of DEV groups share their
complete deployed text arrays with another protocol group. This is a
preregistered investigation threshold, not a significance test or performance
ceiling. Below10%: NO_GO for a dominant exact-input-collapse explanation.
At/above10%: GO_FOR_ATTRIBUTION_ONLY; no remedy or pilot automatically follows.
All failed provenance/parity checks yield an error, not a NO_GO result.

Inputs: `artifacts/pmgr/indexes/csl_{train,dev}.json`,6598/797 groups;
`artifacts/manifests/csl_{train,dev}.jsonl`,18401/1077 rows. Verify the
index's stored source SHA256, exact group membership, both text fields and
unique video IDs. Counts describe local protocol exports; raw official-source
ancestry and feature bytes are inherited, not revalidated here.

Stages, each counted within split only:

1. Exact original string equality (descriptive; not semantic equivalence).
2. Exact model English string equality.
3. Complete, unbounded native CLIP token-ID equality after native cleaning/BPE.
4. Deployed32-slot IDs, mask and segments, including SOT/EOT and uniform
   linspace selection of30 content positions when needed. No prefix truncation.

Record class counts, participating groups/videos, duplicate group-pair counts,
and group-ID witnesses; do not join TRAIN with DEV into one gallery. Differences
between stages2–4 quantify newly introduced identical pairs. Original-text
equality is not assumed to be nested under translation. Inspect all groups,
not only retrieval errors. Report truncated groups separately.

An independent token-array implementation must equal the exact native
`csl_DataLoader._get_text` body on every group, extracted through Python AST
without importing its unrelated video/dependency initialization. Bind only
NumPy and an object supplying the actual tokenizer/special tokens/texts.
The NumPy `long` name is bound to int64 for compatibility; no vendor edits.
`runs/csl_base_b512_s42/resolved_config.yaml` specifies max_words32.

Identical text arrays force identical deterministic query representations
under a shared encoder, but do NOT establish a numerical ceiling for the
actual evaluator: positive-tie accounting and gallery aggregation matter.
No embeddings, scores, semantic harms or encoder-gradient effects are measured.
The closure on generic equivalence positives/native-caption rescue remains.

Commands, cwd `/home/haipd/SLR`: scoped pytest first; then two independent
invocations of `timeout 60s /home/haipd/miniconda3/bin/python
docs/codex_slret_research/tools/diagnose_csl_text_collisions.py --output <path>`
using `evidence/csl_text_collisions_20260923.json` and `_replay.json` under this
directory. Refuse overwrite; require exit0 and byte-identical reports. The
user's autonomous guide authorizes diagnostic code and short synchronous runs,
overriding the skill's routine command-confirmation/default no-generation rule.
