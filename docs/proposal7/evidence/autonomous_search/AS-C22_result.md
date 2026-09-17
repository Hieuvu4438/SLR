# AS-C22 — omissions are real but narrowly supported in dev errors

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (full tokenizer parity; no independent replication)
- Version Label: AS-C22-result-v1

All7,615 train/dev rows match the shared deployed tokenizer EXACTLY under an
explicit reconstruction of uniform content selection. No test access or model
training. `AS-C22-OMISSIONS_run.json` completed exit0 in.955s.

| Split | Rows | Omission-affected | Omitted BPE tokens | Maximum full content length | New exact collisions |
|---|---:|---:|---:|---:|---:|
| TRAIN | 7096 | 230 | 1185 | 57 | 0 |
| DEV | 519 | 10 | 38 | 36 | 0 |

Median content lengths15train/14dev. Every complete caption fits within the
generic77-position text encoder's75content slots. The audit does not assume all
those positions were trained for this retrieval task. No distinct full token
sequences become identical due to uniform truncation; this does not establish
semantic preservation. Per-row omissions and token strings retained locally.

Three of92 historical persistent T2V errors have affected query text(indices
134,140,492); two of87 persistent V2T errors have affected positive text(134,492).
Two in each direction are persistent seed42 top10 errors. No persistent V2T
seed42 strongest confuser has affected text. Other affected candidates could
still alter V2T ranks, so this pair inventory is not a bound on all effects.
T2V subgroup counts use one optimistic rank/query, not its official tie-expanded
aggregate. These are five directional error occurrences, only three unique pairs.

This meets the prespecified feasibility rule for one frozen restoration/control
diagnostic, not any method gate. AS-C23 separately preregistered and executed that
test. No native-caption substitution, alternate translation or longer-context
method proposed. A length omission is a deterministic fact; its linguistic
importance and causal contribution to errors remained unknown at this stage.

## Statistical/integrity scan (11/11)

1. Simpson: separate splits/directions, no subgroup causality or pooled trend.
2. Ecological: token counts not individual sign-level loss claims.
3. Berkson: persistent-error slice selected by rank; descriptive only.
4. Collider: no controlled causal estimate from this slice.
5. Base rate:230/7096 and10/519, three unique persistent pairs disclosed.
6. Regression to mean: no correction/improvement measured by inventory.
7. Survivorship: every train/dev caption accounted for; no examples dropped.
8. Look elsewhere: fixed support rule, no significant token-category search.
9. Forking paths: protocol before run; complete per-row omission records.
10. Correlation/causation: error co-occurrence does not show omitted content caused it.
11. Reverse causality: no claim that failure generated truncation; deterministic
    preprocessing is established but causal error attribution requires intervention.

Two selection/collision tests pass. No GO, new benchmark, positives or global
exhaustion claim. The skill kept measured omission distinct from semantic loss.
