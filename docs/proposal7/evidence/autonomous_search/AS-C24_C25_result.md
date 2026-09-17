# AS-C24/C25 — token competition matters, but proposed analytic replacements fail

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (identity/algebra checked; not independent replication)
- Version Label: AS-C24-C25-result-v1

## Outcome

All six fixed hard/mean endpoint replacements and all three entropy-term
interventions underperform the unchanged strong R0. No candidate is supported.
The two-cycle scorer limit is reached; do not continue with a temperature,
entropy-coefficient, masking or pooling sweep to rescue these findings.

| Intervention | T2V R1 | V2T R1 | Mean R1 | Persistent mean-rank delta T/V |
|---|---:|---:|---:|---|
| Soft expectation baseline | 74.1811 | 76.3006 | 75.2408 | 0 / 0 |
| Hard maximum, both channels | 74.1811 | 75.1445 | 74.6628 | +.1522 / +.7816 |
| Hard A, soft B | 73.4104 | 75.5299 | 74.4701 | +.0435 / +.2184 |
| Soft A, hard B | 72.8324 | 75.7225 | 74.2775 | +.1304 / +.5287 |
| Mean, both | 33.9114 | 32.7553 | 33.3333 | +15.9674 / +13.2759 |
| Mean A, soft B | 68.2081 | 64.7399 | 66.4740 | +1.1957 / +1.7701 |
| Soft A, mean B | 64.5472 | 66.0886 | 65.3179 | +4.7174 / +4.0460 |
| Log-mean-exp, both | 71.8690 | 70.7129 | 71.2909 | +1.4891 / +.8966 |
| Log-mean-exp A, soft B | 72.8324 | 73.7958 | 73.3141 | +.4783 / +.0115 |
| Soft A, log-mean-exp B | 71.6763 | 74.3738 | 73.0250 | +.8043 / +.5632 |

All nonidentity variants worsen persistent mean ranks in both directions and
fail their lead gates. Full R5/R10 and per-query ranks/fixed-confuser margins are
retained in each `AS-C24-*`/`AS-C25-*` metric JSON. Some individual queries improve;
those selected successes are not evidence of overall efficacy or an oracle router.
For example hard-both improves38T/21V ranks but worsens39T/46V, with mean fixed
confuser margins declining.288634/.337217(T/V) score units.

## What changed—and what did not

Same frozen519-row R0seed42 video/text representations, normalized dot products,
legacy CLS/PAD inner competitors, outer validity masks and .5 channel mixture.
No encoder, pooled feature representation, data, labels, positives, temperature,
trainable parameters or official protocol changed. Mean endpoint is NOT the
AS-C01 normalized-pooled-representation test: here all original normalized token
vectors and outer masks remain, but inner competitor weighting becomes uniform.

AS-C24 changes the inner expectation to its analytic tau→0 hard-max and
tau→infinity mean endpoints, both and one channel at a time. Pointwise channel
ordering mean<=soft<=hard holds within tolerance, but increasing all similarities
does not preserve or improve their relative ranking. These endpoints were fixed
before execution, not chosen from a temperature sweep.

## AS-C25 exact entropy identity

For p=softmax(a/tau), E=sum(p*a), H=−sum(p*log(p)), and fixed tau=.07:

`E = tau*logsumexp(a/tau) − tau*H`.

Log-mean-exp differs from E+tau*H by tau*log(N), a fixed constant for each
channel because every inner pool has32text or65video slots. The maximum scaled
algebra error is1.907349e-5 (<5e-5). Direct log-mean-exp and entropy-added
formulations yield identical per-channel T2V/V2T ranks (zero discrepancies in
all four comparisons). Direct log-mean-exp was the prespecified primary score.

The local derivative `dE/da_i = p_i*(1+(a_i−E)/tau)` is negative for many active
coordinates. Tests verify the expression against autograd and verify positive
log-mean-exp derivatives. Observed fractions, pooling all active outer rows:

| Pair population (519pairs each) | Channel A negative fraction | Channel B negative fraction |
|---|---:|---:|
| Official positive | .659675 | .735693 |
| Fixed strongest T2V confuser | .628845 | .709217 |
| Fixed strongest V2T confuser | .632647 | .714254 |

These are unweighted coordinate COUNTS, not negative gradient mass, parameter
pressure, or feasible independent perturbations of all similarities. Positive
and confuser pair populations differ; no matched causal inference follows.
Monotonicity is a property of the aggregation function, not automatic retrieval
quality. Removing the implicit entropy subtraction reduces mean R1 by3.949904pp;
it therefore does not support a claim that the observed negative derivatives are
a harmful bottleneck in this fixed model. No gradient surgery/regularizer follows.

## Execution and verification

AS-C24 completed exit0 in1.000s, peak GPU249,609,728bytes; AS-C25 completed exit0
in1.158s, peak GPU328,836,608bytes. No numeric/parity failure or retry. No test
access or optimization. Both soft replays have maximum channel error1.525879e-5
and exact historical directional ranks. Cache/checkpoint/manifest/protocol/code
hashes recorded in terminal run JSONs; score/channel matrices saved locally.

Focused suite43passed in1.29s. Added tests cover endpoint ordering, actual
analytic limits, legacy inner PAD treatment, constant-tensor shape invariance,
entropy identity and gradient sign/formula. The full research runs have not
been independently replicated; identity checks are not three-seed evidence.

## Decision / next causal layer

Reject these nine fixed interventions, not all score functionals or information
in the representations. Ordinary pooling and entropy changes supply no novelty
claim even if individual examples improve. No full candidate review or expensive
three-seed campaign is justified by these failed lead gates.

Switch to generalization/data support rather than another scorer variant. The
AS-C20 source-held model strongly fits but generalizes poorly; AS-C19 found
exact-text overlap, not the broader lexical-support distribution. Next bounded
question: is that source-held calibration population substantially different
from official dev in TRAIN lexical/source support under a COMMON fit reference?
Compare both against the same5721fit rows, also record dev support under all7096
TRAIN rows. Use input-only descriptors and fixed strata, no dev-trained model or
inference that this explains R0's advantage. This can refine whether the failed
calibration is a useful research substrate without lowering its adequacy gate.

## Statistical/integrity scan (11/11, both experiments)

1. Simpson: directions and global/persistent outcomes separate; source-wise
   reversals not ruled out.
2. Ecological: coordinate fractions not signer or individual-sign claims.
3. Berkson: strongest-confuser/persistent populations are selected; descriptive.
4. Collider: no error-conditioned causal estimate or learned routing.
5. Base rate:519pairs/category; coordinate denominators recorded, not gradients
   weighted by magnitude or independently sampled observations.
6. Regression to mean: selected improved queries not accepted as method success.
7. Survivorship: all six endpoint and three entropy variants retained.
8. Look elsewhere: no significance inference, winning variant or coefficient sweep.
9. Forking paths: each separate protocol precedes its run; thresholds unchanged.
10. Correlation/causation: algebraic nonmonotonicity does not demonstrate harmful
    encoder updates or a linguistic mechanism; tested correction harms ranking.
11. Reverse causality: no assertion that entropy/derivative statistics caused the
    historical generalization gap.

ARS discipline distinguished mathematical properties, numeric execution and
research efficacy. Goal remains active; no candidate, GO or global exhaustion.
