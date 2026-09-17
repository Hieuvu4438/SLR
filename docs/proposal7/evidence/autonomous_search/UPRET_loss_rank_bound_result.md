# UPRet: auxiliary-score capacity and saved deterministic-loss bound

2026-09-16. academic-research-suite, inline source verification. AI-assisted;
ANALYZED, not independent review. Posthoc analysis of an existing run, not a
preregistered experiment, new rank evaluation, or method proposal.

## Finding

[V] UPRet trains the deterministic retrieval channels: the released code adds
the stochastic transport score to their scaled logits before contrastive CE.
Removing transport at inference is intentional in the
[original paper, §3.6](https://arxiv.org/html/2405.19689v1#S3.SS6).
It does not imply that only the auxiliary heads receive retrieval supervision.

[I] With the source defaults, the auxiliary term can change a positive–negative
logit margin by at most **0.75**, subject to the finite/unit-mass assumptions
below. Independently, the saved no-OT losses imply strict deterministic-channel
separation on all three inspected B32 TRAIN batch states at partial step767.
Thus auxiliary-head-only success is not an explanation of those batch losses.
Neither result establishes good full-gallery retrieval or generalization.

## Source and derivation

Pinned UPRet revision `046366227417e1d8ec14145965403462df345984`:
`modules/modeling.py`, Sinkhorn:370–384 and flip_similarity_softmax:573–695.
The committed version, not historical local repairs, defines the source claim.
Modeling source SHA256:
`20889071772f679cab763da7d6c5e54aef6c0d770713ca6c301dceff1cf2a385`.

For S normalized samples, let similarities s_ij be in [-1,1], and let T be a
nonnegative transport matrix of total mass M. The source reduction is

`q = (sum_i max_j(T_ij s_ij) + sum_j max_i(T_ij s_ij)) / (2S)`.

Each row maximum is at least its row average, and at most its row absolute sum.
The same holds for columns. Consequently **-M/S² ≤ q ≤ M/S**. This does not
require converged/equal row marginals. In real arithmetic, the final Sinkhorn
column normalization sets column sums to the requested v, so M=1 when the
kernel/scalings are finite positive and at least one iteration runs. Underflow,
nonfinite intermediates and floating-point deviations are not covered by an
exact-arithmetic certificate.

For S=2, q∈[-.25,.5]. For nonnegative coefficient λ, any margin perturbation is
at most .75λ. The default λ=1 is outside the deterministic CLIP scale in source.
For deterministic competitor advantage δ≥0, the combined per-query CE is at
least `softplus(δ-.75)`; at δ=0 this is approximately .386871. If all B
deterministic logits are equal, CE is at least `log(1+(B-1)exp(-.75))`:
approximately 2.750047 for B32 and 5.490504 for B512. These are bounds, not
assertions that every extremum can be realized jointly by the shared network.

A bounded logit perturbation does **not** bound encoder-parameter gradients:
normalization and shared parameterization can amplify them. The earlier
[real-gradient screen](UPRET_real_gradient_result.md) remains the relevant
measured sensitivity result; this analysis does not override it.

## Saved no-OT evidence

[Source run](UPRET-REAL-GRADIENT_run.json),
[calculation](../../../../methods/information_probe/upret_loss_rank_bound.py),
[all six state/batch results](UPRET-LOSS-RANK-BOUND.json).

The old probe's no_ot arm uses zeros in place of transport on the same encoded
v/t tensors. Its objective matches `methods/sssc/method1/baseline.py:340` with
dual_mix=.5 and balance: L is the average of four B32 directional CEs. There
are128 nonnegative per-query terms. Thus every term is ≤128L. For every
positive–competitor logit margin m, `CE ≥ log(1+exp(-m))`, giving

`m ≥ -log(expm1(128L))`.

Any nonpositive margin requires CE≥log2. Even one such query would require
total loss≥log2/128 = .005415212348124573.

| Saved partial-step batch | No-OT loss | Lower bound on every deterministic margin |
|---|---:|---:|
| 1 | .0000848869968 | 4.516722 |
| 2 | .0003121728660 | 3.199878 |
| 3 | .0000439586511 | 5.177416 |

All three are well below the one-wrong-query loss floor. Initialization losses
3.782370/3.724959/3.666627 yield uninformative bounds; they do not imply that
all initialization queries are wrong. All outputs are retained, not selected.

The calculation verifies hashes of the saved run, baseline source and config,
and checks the four32-row batch metadata records (one calibration, three screen).
Executed with Python3.13.5, exit0. No models imported or new forward/backward,
GPU, training, data, DEV or TEST access. Run SHA256:
`9a9d34bafd5fcbd7702b098dc77e83e2c40bc5517b92a73346ce1973c54be25f`.
Probe SHA256:
`abc302a501c384d6e10313acf9092a3cf3013ab039a6b7902fe0d6e7934d3969`.

## Limits and decision

These are real-arithmetic implications of recorded FP32 loss values, not
interval-certified arithmetic or newly measured Recall@1. The batch tokens
include training-mode augmentation/stochasticity. They are not clean evaluation
tokens, all TRAIN examples, full-gallery rivals, or a completed competitive
UPRet model. No semantic interpretation of batch correctness is certified.

The original paper was checked at §3.6 and the inference-efficiency paragraph;
published performance tables surfaced in browser context but were not used.
No new full-paper/final-version parity review is claimed. Primary source supports
the implementation contract, local algebra the bounds, and the saved record the
conditional batch inference. No SOTA claim follows from any of these.

Reject auxiliary-only batch success as the proposed failure explanation. Do not
launch an uncertainty-weight/reduction/temperature/long-training rescue. General
train-to-gallery transfer remains unresolved, but a generic negative-pool or
distillation reformulation would collide with the closed families. No GO,
Proposal8 or global barrier. Continue with the cross-repository gap overview.

Handoff validation: the saved six-record JSON exactly matches a fresh execution
of the arithmetic-only script (exit0). Both evidence JSON files parse; the
overview passport has one claim-intent manifest. All30 local links across this
note and the overview resolve. No trailing whitespace found in the new prose
and script; tracked git diff --check passes (it does not cover untracked files).
The initial jq validation command could not run because jq is unavailable;
JSON parsing/comparison was instead performed in orchestration JavaScript.
