# AS-C23 — restoration changes ranks but fails the lead gate

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (identity verified; no independent replication)
- Version Label: AS-C23-result-v1

## Result

| Condition | T2V R1 | V2T R1 | Mean R1 | Persistent mean-rank delta T/V |
|---|---:|---:|---:|---|
| Canonical32 | 74.181118 | 76.300578 | 75.240848 | 0 / 0 |
| Full original tokens | 74.181118 | 76.493256 | 75.337187 | −.021739 / 0 |
| Same-length retained-token repetition | 74.181118 | 76.300578 | 75.240848 | −.010870 / −.011494 |

Restoration improves mean R1 by only.096339pp versus BOTH controls, below the
fixed.5pp gate; persistent V2T mean rank does not improve. No method candidate.
Restored T2V R5/R10 unchanged91.136802/95.375723; V2T R5 falls.192678pp to
91.714836, R10 unchanged94.990366. All full-gallery metrics and ranks retained.
Full restoration changes only one T2V query's rank(index134, zero-based13→11);
repetition also improves that query13→12. No affected persistent query becomes
R1-correct. V2T has12 changed query ranks, one net extra R1 success. These rank
movements are not three-seed evidence or a demonstrated information bottleneck.

## Controlled scope and execution

One fixed R0 seed42 checkpoint; all519 canonical video candidates and text
queries; no labels for routing. The length>30 input-only rule changes exactly
10 score columns. Other509 columns remain BITWISE canonical. Both conditions
use the same original sequence lengths, positional slots, SOT/EOT and scoring
dimensions. Repetition preserves every retained token at its original position
and fills omitted positions using nearest retained tokens, without supplying
omitted identities. This synthetic control is not semantic equivalence.

Text encoding uses77slots in both arms, then affected columns score only actual
SOT/content/EOT tokens, excluding newly introduced PAD slots. Canonical affected
inputs had all32slots valid; therefore no original padding slots were removed.
The unchanged columns keep historical padding policy. No repository tokenizer,
production model, encoder weights, positives or official splits modified.

`AS-C23-RESTORE_run.json` completed exit0 in4.15s, peak GPU696,940,032bytes.
Canonical32 re-encoding matches every cached text token/mask exactly. Independent
channel kernel differs by at most1.525879e-5 and has exact directional rank parity.
No numerical failures or retries. One control-position test passes; full focused
suite39passed in1.24s after AS-C22/C23. Source/protocol/checkpoint/cache/manifest
hashes and full metric files saved. This is not a full independent run replication.

## Consequence

Reject this exact frozen restoration as a useful candidate, without reducing
thresholds or selecting another context length. The broader contextual regime
remains untested, but these ten affected dev captions provide weak direct support
for further text-limit tuning. No native-caption preservation revival. Generic
extension of an input limit would not supply method novelty even if it helped.

Next change layer from text input to the cross-modal score functional. The
existing AS-C01 pooled diagnostic changed representations, whereas AS-C14
changed masks and AS-C03 changed channel mixture. None isolates whether the
inner token competition's soft expectation versus hard maximum suppresses
rank-critical distinctions while holding representations/masks/outer averages
fixed. A fixed analytic endpoint comparison is a distinct cheap question—not
permission for a dev-temperature sweep, token-weighting method or generic loss.

## Statistical/integrity scan (11/11)

1. Simpson: both directional/global/persistent outcomes reported; no every-source claim.
2. Ecological: one net retrieval correction not sign-level semantic validation.
3. Berkson: persistent subgroup selected historically; global gate retained.
4. Collider: no conditioned causal model or dev-trained routing.
5. Base rate:10changed columns/519; three unique affected persistent pairs.
6. Regression to mean: small persistent movement not treated as mechanism success.
7. Survivorship: both fixed arms and unchanged baseline reported.
8. Look elsewhere: one fixed screen, no significance claim from the small gain.
9. Forking paths: lengths/control/gates fixed beforehand; no rescue sweep.
10. Correlation/causation: full versus repetition changes lexical identities and
    downstream context; does not identify a specific linguistic contrast.
11. Reverse causality: measured score differences do not establish why historical
    learning produced errors or that longer-context training would fix them.

ARS discipline kept execution success and the favorable tiny gain separate from
the failed gate. Goal remains active; no GO or global exhaustion claim.
