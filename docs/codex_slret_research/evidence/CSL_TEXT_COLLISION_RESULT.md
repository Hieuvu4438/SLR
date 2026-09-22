# CSL exact text-input collision result

2026-09-23, Cycle18. **NO_GO for a dominant exact-input-collapse explanation
on CSL DEV.** This is an input-identity census, not a video-meaning judgment,
translation-quality evaluation or retrieval experiment.

## Registered question and result

[Protocol](CSL_TEXT_COLLISION_PROTOCOL.md) preceded both executions. The locked
primary gate required at least10% of DEV groups to share complete deployed
text arrays with another existing group. Observed: **0/797 groups**, so no
attribution follow-up or method pilot is admitted by this result.

| Split / stage | Collision classes | Participating groups | Participating videos | Group pairs |
|---|---:|---:|---:|---:|
| TRAIN original string | 20 | 40/6598 | 114/18401 | 20 |
| TRAIN English string | 25 | 50/6598 | 142/18401 | 25 |
| TRAIN full token IDs | 26 | 52/6598 | 148/18401 | 26 |
| TRAIN deployed arrays | 26 | 52/6598 | 148/18401 | 26 |
| DEV, each of the four stages | 0 | 0/797 | 0/1077 | 0 |

TRAIN deployed collision participation is0.788118% of groups. Native cleaning
and BPE introduce one additional identical TRAIN group pair relative to exact
English strings. Uniform selection introduces zero new identical pairs in
either split. There are15 TRAIN and3 DEV groups with more than30 content tokens.
Original-to-English collision classes are not assumed nested: the difference
between their counts is not a count of translation-induced losses.

## Execution, provenance and material passport

Six scoped synthetic tests passed before execution (exit0). They cover padding,
uniform selection, an induced selection collision, collision accounting and
gate boundaries, duplicate IDs and invalid manifest mappings. This is not a
full-repository test run or a natural-language sensitivity validation.

Both registered commands completed with exit0 within their60s limits:

```bash
timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_csl_text_collisions.py --output docs/codex_slret_research/evidence/csl_text_collisions_20260923.json
timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_csl_text_collisions.py --output docs/codex_slret_research/evidence/csl_text_collisions_20260923_replay.json
```

The [first report](csl_text_collisions_20260923.json) and
[replay](csl_text_collisions_20260923_replay.json) are byte-identical, SHA256
`bcb4e1d474148a30d35bd7c3df023112e15461294c30481d6eb83712805c3c49`.
All10 recorded source hashes were rechecked against current files. Native
loader-array parity passes for all7395 groups. Both manifest hashes match the
indexes' stored source hashes; video membership, both text fields, language,
dataset and split match exactly. Runtime versions and group-ID collision
witnesses are in the machine-readable reports.

Materials read: local TRAIN/DEV indexes and source manifests in full; native
tokenizer source, vocabulary and loader text construction; baseline max_words
configuration. No human-read attestation is implied. Raw official-source
ancestry, feature hashes and deployed model execution were not revalidated.
No TEST data, scores, model weights, videos or feature arrays were loaded.

## Validation cautions —11/11 checked

1. Simpson: splits remain separate; no pooled population or signer claim.
2. Ecological: group-string identity says nothing about individual video meaning.
3. Berkson: the full local split population is counted, not selected confusers.
4. Collider: there is no conditioning on retrieval outcome or causal estimate.
5. Base rate: group and video denominators are explicit; pairs are not videos.
6. Regression to mean: no pre/post performance comparison or error-selected fit.
7. Survivorship: all7395 groups pass provenance and parity checks; none dropped.
8. Look elsewhere: one locked DEV gate; descriptive TRAIN counts cannot rescue it.
9. Forking paths: no fuzzy matching, alternate threshold or semantic relabeling.
10. Correlation/causation: exact input collisions do not measure retrieval harm;
    no collisions do not prove faithful translations or good representations.
11. Reverse causality: no direction between input identity and retrieval failure
    is estimated; there are no retrieval outcomes in this census.

## Consequence

Stop this specific widespread DEV input-collision lead. Distinct inputs may
still yield similar embeddings or omit meaningful distinctions; these remain
unmeasured, not established failure mechanisms. Identical query arrays alone
also do not give an evaluator-independent retrieval ceiling because ties and
group aggregation matter. Do not reopen native-caption, multilingual-swap,
equivalence-positive or PMGR rescues from these counts.

ARS discipline supplied preregistration, replay and claim boundaries. Bounded
empirical progress, not a new method, efficacy gain or SOTA. The main research
goal remains active and incomplete; no human/AI semantic annotation was added.
