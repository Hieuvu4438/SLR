# AS-C21 — separate-stream evidence does not yield a fixed-fusion gain

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (identity replay checked; not an independent replication)
- Version Label: AS-C21-result-v1

## Outcome

The fixed late-fusion diagnostic fails its preregistered lead gate. It improves
persistent-error mean ranks, but does not improve full-gallery mean R1 over the
strong canonical baseline. No fusion candidate, active-acquisition revival,
novelty claim, or GO follows.

| Variant | T2V R1 | V2T R1 | Mean R1 | Persistent mean-rank delta T/V |
|---|---:|---:|---:|---|
| Early fusion, two-pass control | 74.1811 | 76.3006 | 75.2408 | 0 / 0 |
| Aware only | 35.4528 | 25.8189 | 30.6358 | +25.2826 / +33.4253 |
| Agnostic only | 73.9884 | 75.9152 | 74.9518 | −.1304 / +.1954 |
| Fixed late .1aware+.9agnostic | 73.0250 | 76.3006 | 74.6628 | −.3804 / −.6207 |
| Late with aware video rows shifted | 72.8324 | 75.3372 | 74.0848 | +.3804 / +.3448 |

Late fusion loses.578035pp mean R1 and1.156069pp T2V R1 against canonical.
It beats the shifted control by.578035pp and improves both persistent mean ranks,
but cannot pass by cherry-picking these diagnostics. Late R5=90.944123/91.907514,
R10=95.953757/94.990366(T/V); full per-query metrics for every variant retained.

Correct-set overlap: aware/agnostic/both/union are184/384/173/395 T2V and
134/394/123/405 V2T. Either component is correct on20 T2V and22 V2T queries
where early fusion is wrong. This is a nondeployable label-informed union and
NOT incremental net R1 gain: components also lose early-correct queries.
T2V overlap uses one optimistic rank/query, distinct from its primary expanded
tie denominator. No oracle router, dev-selected weight, or new targets created.

## Evidence and limitations

Protocol `AS-C21_protocol.md`; terminal `AS-C21-STREAMS_run.json` completed
exit0 in5.32s, peak allocated GPU673,064,448bytes. One seed42 R0 checkpoint,
519-row official PH dev gallery, native encoder precision, zero optimizer
updates and no test access. Fixed two-pass control matches the encoder/scorer
pass count of the late mixture; timing is not a throughput benchmark.

All canonical/repeat contextual tokens equal the existing frozen cache exactly;
their score channels equal one another. Maximum channel difference against
the historical scorer1.525879e-5 (<2e-5) with exact directional rank parity.
Raw .9 fusion equals .1aware+.9agnostic exactly; masks, IDs and sampled indices
match across streams. R0/cache/manifest hashes retained. Focused suite36passed
in1.25s, including fixed-mix orientation/control test; git diff --check passed.

The frozen transformer was trained on early-fused features. Aware-only collapse
can reflect input distribution shift; it does NOT show that H2S-aware I3D lacks
linguistic information. Agnostic dominance at this fixed .9 mixture is not a
causal certificate that the aware stream is unnecessary. Ordinary dynamic
fusion, weight tuning and extra passes are not new mechanisms by themselves.
No extra stream encoding caches saved, only small score/metric artifacts.

## Decision / next research layer

Reject this exact frozen fixed late-fusion specification. Do not conduct a
dev-weight sweep to rescue it or infer a universal fusion/acquisition barrier.
Move to the text-input contract: inventory actual token omissions under the
deployed uniform32-token limit, exact resulting collisions, and their error
support before any longer-context intervention. AS-C16 used truncated keys
for an order audit; it did not measure this omission/error-support question.
This is distinct from the closed native-caption method: no replacement language,
translation resource, new positives or novelty claim is authorized by an
inventory. If coverage is insufficient, record that and switch again.

## Statistical/integrity scan (11/11)

1. Simpson: both directions and persistent/global outcomes retained; source-wise
   heterogeneity untested, no aggregate-to-every-source extrapolation.
2. Ecological: average stream score not a claim about individual sign cues.
3. Berkson: persistent slice and oracle union selected by errors; descriptive.
4. Collider: no error-conditioned causal inference or proposed oracle routing.
5. Base rate:519 queries, component intersections/unions and lost ranks visible.
6. Regression to mean: persistent improvement is not accepted as overall efficacy.
7. Survivorship: all fixed variants reported, no best-weight/seed selection.
8. Look elsewhere: one predefined screen, no significance or three-seed claim.
9. Forking paths: weights/control/gates preregistered and unchanged.
10. Correlation/causation: stream ablations confounded by training input regime;
    row shift is an alignment-negative control, not semantic causal evidence.
11. Reverse causality: component complementarity does not explain why baseline
    fusion succeeded or failed on an example.

The ARS workflow kept the favorable persistent diagnostic separate from the
failed primary gate. No method currently survives; goal remains active.
