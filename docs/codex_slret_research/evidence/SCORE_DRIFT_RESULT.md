# Score drift is not a rank-harm label

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate (inline, autonomous diagnostic authority from proposal9)
- Origin Date: 2026-09-22
- Verification Status: VERIFIED for deterministic saved-score analysis only
- Version Label: score_drift_v1
- AI-assisted analysis; no independent linguistic or human relevance review.

## Result and scope

[M] The proposed sufficiency claim fails in all three seeds and both directions:
an unpaired item's score can increase while its margin against the official
paired item improves, or decrease while the margin deteriorates. This is not
evidence that an unpaired item is semantically correct or incorrect.

Fixed before measurement: step111→666, fusion only, seeds42/1337/2026,
519 official PH DEV pairs, 1e-7 numerical guard. Six matrices replay their stored
metrics and per-query ranks exactly. No selected-step substitution, fitting,
GPU, TEST, corpus acquisition or new annotation. See the
[protocol](SCORE_DRIFT_PROTOCOL.md), [data](score_drift_20260922.json) and
[byte-identical second execution](score_drift_20260922_replay.json).

For each query, hold its step111 highest-scoring unpaired competitor fixed:

| Seed | Direction | Competitor score rises | Of these, paired margin improves | Competitor score falls but margin worsens | Official query ranks better / same / worse |
|---|---|---:|---:|---:|---|
| 42 | T2V | 355 | 162 | 21 | 25 / 475 / 19 |
| 42 | V2T | 338 | 138 | 16 | 30 / 472 / 17 |
| 1337 | T2V | 330 | 160 | 38 | 21 / 478 / 20 |
| 1337 | V2T | 311 | 128 | 30 | 27 / 472 / 20 |
| 2026 | T2V | 361 | 167 | 23 | 22 / 476 / 21 |
| 2026 | V2T | 353 | 144 | 15 | 30 / 465 / 24 |

Each row contains519 queries; columns are not disjoint categories. Margin
improvement is not necessarily a rank change or a successful retrieval. The
complete nine-quadrant counts include small/zero changes in the JSON.
Among all268,842 unpaired entries per direction, score-up/margin-up counts are
78,522/71,219 (seed42 T2V/V2T),68,033/57,321 (1337),83,366/75,477 (2026).
These pairs are highly dependent; they are not independent observations.

## Why the diagnostic separates the quantities

[V, algebra] Let a query's paired score be p and a fixed competitor's score n.
Its margin change is Δm=Δp−Δn. The sign of Δn does not determine Δm. Adding
a global scalar c to a score matrix leaves all row/column softmax probabilities,
paired margins and rankings unchanged. A time-varying c changes absolute score
trajectories. This is a score-level counterexample, not a claim that a specific
fixed normalized embedding model can implement every such shift. Unit tests
check both directions and strict rank crossings on exact synthetic arrays;
synthetic arrays are not presented as signed-language examples.

[I] Consequently, raw score drift alone cannot label a negative as harmful,
informative, semantically invalid, or responsible for forgetting. Relative
margin is more directly relevant to ordering, but even its decline does not
identify a training example's causal gradient contribution.

The bounded literature check rediscovered
[SCL-SLT](https://aclanthology.org/2026.acl-long.2116/), already recorded in
proposal5/7. Its abstract/intro discusses negative-similarity trajectories and
curriculum selection for translation. We did not reproduce its method, inspect
all selection equations, or challenge its reported translation results. A
methods-page fetch timed out. This diagnostic tests the generic inference above,
not whether the paper's actual selector is invariant to a common score shift.
Primary publisher record verifies publication; author explanations are [A], not
locally established causal evidence. No bibliographic API or independent source
corroboration was run; venue/author conflicts beyond the record were not audited.

## Decision and limits

Close raw-score sign as a sufficient diagnostic of rank harm. Do not promote a
trajectory-based miner, calibration, new relevance label, or protected update.
Those mechanisms still face the existing blacklist. No claim that all-negative
pressure is benign follows. This changes what constitutes valid evidence for
Q15/Q16; it does **not** admit a pilot or produce a primary method.

The next independent scientific requirement remains a non-colliding causal
mechanism with a separating test. Existing official pair labels suffice for
ranking measurements, while semantic attribution remains unresolved. MY DGS
permission is not needed for this completed diagnostic and remains a separate
conditional branch. The overall SOTA objective is incomplete.

## Reproducibility and statistical validation

Command from `/home/haipd/SLR`:

```sh
/home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_score_drift.py --output docs/codex_slret_research/evidence/score_drift_20260922.json
```

Second execution changes only the output filename to
`score_drift_20260922_replay.json`; `cmp` exits0. Both executions exit0. Reports
record20 input/code/protocol hashes, Python/NumPy versions and unchanged base
HEAD. Neither execution modifies source matrices, annotations or model files.
Six dedicated tests cover discordant movements, global-shift invariance,
fixed-rival selection, direction/numerical guard and invalid inputs.
The combined research-audit and C27 fixture suite passes22/22 tests; all20
recorded input/code/protocol hashes still match after execution.

Overall confidence: CAUTION for generalization/causation; exact verified counts
for these files. No p-values, confidence intervals, effect-size significance or
population independence are asserted. Fallacy scan:11/11 checked.

| Check | Finding / guard |
|---|---|
| Simpson | Seed and direction reported separately; no pooled trend inference |
| Ecological | Pair margins are not interpreted as query successes or semantic labels |
| Berkson | Fixed initial maximum is selected; all-pair counts also reported |
| Collider | No adjusted causal regression; rival selection is not causal control |
| Base rate |519 queries and268,842 unpaired entries explicitly distinguished |
| Regression to mean | Initial maxima can regress; no causal improvement claim |
| Survivorship | All six planned matrices present; conclusions restricted to these completed runs |
| Look elsewhere | Both directions/all seeds reported; no significance search |
| Forking paths | Protocol precedes counts; fixed endpoints and tolerance; exploratory historical data |
| Correlation/causation | Endpoint changes do not isolate training-example effects |
| Reverse causality | Chronology verified, but temporal precedence alone is insufficient |
