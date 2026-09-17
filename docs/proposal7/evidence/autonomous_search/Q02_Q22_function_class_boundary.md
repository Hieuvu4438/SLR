# Q02/Q22: a shared scalar score is not itself an expressivity obstruction

## Scope

2026-09-16; academic-research-suite, inline analytical follow-up to the
[37-question audit](Q01_Q37_feasibility_audit.md). AI-assisted derivation, not an
external theorem attribution, novelty claim, numerical experiment or linguistic
validation. No dataset, score array or checkpoint was loaded for this argument.

## Result

Two broad explanations can be rejected **as logical implications**:

1. Using one scalar matrix for both directions does not by itself force a
   retrieval trade-off under paired singleton positives.
2. Having more gallery items than embedding dimensions does not by itself make
   perfect bidirectional R@1 unrepresentable.

Neither statement says the trained CiCo model can solve the actual gallery,
that its encoders can learn a desired representation, or that all semantic input
distinctions are available. Those are separate empirical questions.

## 1. Joint row/column ranking constraints are satisfiable

Use the existing paired-singleton convention and reorder identities so the
positive pairs are diagonal. Perfect strict bidirectional top-1 asks only for

`S[i,i] > S[i,j]` and `S[j,j] > S[i,j]` for every `i != j`.

These constraints do not contradict each other: choose diagonal entries 1 and
off-diagonal entries 0. Each off-diagonal variable is required to be below two
positive variables; no constraint requires one of those positive variables to
be below that off-diagonal variable. This is an existence argument about an
unrestricted finite score table, **not a permissible inference algorithm**:
constructing that table by reading evaluation identities would be label leakage.

Consequently, different measured directional recalls cannot on their own prove
that sharing a scalar score is the bottleneck. A claimed obstruction must specify
a narrower function class or an input-identifiability restriction.

## 2. Even rank-two dot products can realize all strict diagonal maxima

For any finite `n >= 2`, take `n` distinct unit vectors on a circle:

`z_i = (cos(2*pi*i/n), sin(2*pi*i/n))`, for `i = 0,...,n-1`.

Assign matching modalities the same vector and define `S[i,j] = z_i · z_j`.
Then `S[i,i] = 1`, while every off-diagonal entry is strictly less than 1.
All diagonal entries are strict maxima in both directions, although the score
matrix has rank at most two. Matching an identity score matrix exactly would
have a different rank requirement; reproducing its top-1 ranking does not.

The minimum diagonal gap in this construction is `1 - cos(2*pi/n)`, positive
for finite n but shrinking with n. Thus this is not a fixed-margin robustness
guarantee, a floating-point stability result, a semantic embedding or a learned
generalization result. It refutes only the proposed implication from item count
versus dimension to inevitable top-1 failure.

If different official IDs have identical deterministic model inputs, an encoder
cannot arbitrarily assign them distinct vectors. The construction explicitly
assumes distinguishable inputs and available mappings; it does not bypass the
actual duplicate-text constraints documented in the research state. It also
does not change official positives or tie handling.

## 3. What the actual source restricts

Source inspected for this follow-up:

- [CiCo `flip_similarity_softmax`](../../../../third_party/SLRT/CiCo/CLCL/modules/modeling.py:468), including normalization, inner softmax, outer masks and channel outputs.
- [bridge `mixed_score`](../../../../shared/slr_common/upstream/cico_bridge.py:93) and paired scoring path through line 142.
- [diagnostic scorer](../../../../methods/information_probe/scoring.py:7), including the bounded readout.

The deployed score is not an arbitrary table. It uses normalized token affinities,
fixed-temperature softmax expectations along two token axes, outer validity
averages, and a shared channel mixture. This identifies actual restrictions worth
reasoning about, but does not establish which causes a persistent error.

For instance, permuting contextual tokens together with their masks leaves the
analytic aggregate unchanged. That is a **post-encoding** invariance, not absence
of order in the entire pipeline. [C06](AS-C06-TEMPORAL_run.json) already shows
pre-encoding order sensitivity. The raw-readout variant has a related scope
limitation in [C44](AS-C44_result.md). Repeating either observation is not a new
bottleneck. Numerical summation-order effects are outside this algebraic claim.

Moreover, the softmax pooling makes the full CiCo score a nonlinear function of
token affinities. A rank bound for a single pooled dot-product matrix must not
be silently assigned to this late-interaction score matrix.

## Decision consequence

Do not propose independent directional heads, a higher-rank embedding or a
non-scalar/cohort mechanism merely because the two recalls differ or the gallery
has 519 items. The above counterexamples remove those justifications; they do not
show that such architectures could never help for another measured reason.

Q02/Q22 remain **unresolved at the implemented-function/learnability level**.
No new admissible restriction with an untested decision consequence survived this
bounded formulation check. Existing permutation, channel-mixture, entropy,
cohort and gallery-filter routes are already tested or collide with closed
families. No new run, Q38, candidate or SOTA claim follows.

Next move to another existing unresolved question, rather than restating this
function-class argument or launching a generic scorer. A useful next analytical
target is Q01's diagnostic adequacy: state what a positive control could establish
without restarting the prohibited clean-model recipe campaign. If no permissible
positive control can be justified, record the exact missing prerequisite; do not
turn failure to formulate it into universal information insufficiency.

## Adversarial limitations

The strongest counterargument is that an existence construction ignores learning,
semantic generalization, robust margins and duplicate inputs. Agreed; those are
explicitly outside the claim. This corrects an overbroad *necessity* inference,
not a measured explanation of PH errors. The result does not pass any method GO
gate and does not establish a global research barrier.
