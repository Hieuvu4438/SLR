# Paired TRAIN-anchor mechanism screen — Cycle14

2026-09-22. Academic-research-suite inline adversarial mechanism review.
AI-assisted; no independent reviewer, linguistic judgments or new annotation.
Status: **REJECTED as a generic method proposal**, not an empirical negative.

## Hypothesis and permitted inputs

Observation: persistent official-pair errors are measured in Cycle2. No
neighborhood-correspondence defect has been measured. Hypothesis: video/text
similarities to corresponding TRAIN examples preserve rank-relevant structure
that the deployed cross-modal comparison misses. This is a hypothesis about
official pairing, not an assertion about confusers' video meaning.

For fixed paired anchors `(a_i,b_i)`, define

`r_v(v)_i = k_v(v,a_i)` and `r_t(t)_i = k_t(t,b_i)`;
`s_A(v,t) = r_v(v)^T r_t(t)` (or cosine between those vectors).

Input: one query, candidate, fixed TRAIN pairs and frozen encoders. Output:
one compatibility score. No other evaluation queries, relevance relabeling,
assignment capacity or new backbone. A naive implementation needs O(K)
anchor-coordinate comparisons per pair after encodings; bank preparation and
storage must be charged. No parameters change in this diagnostic specification.

## Algebraic boundary — derived here, not empirical evidence

For normalized single-vector encodings x,y and normalized anchor-row matrices
A,B, linear-kernel coordinates give

`r_v = A x; r_t = B y; s_A = x^T A^T B y`.

Thus the dot-coordinate version is a fixed bilinear map, not inherently a new
scoring family. Cosine adds the denominator `||Ax|| ||By||`; it is still a
fixed transformed-vector cosine. Nonlinear kernels do not obey this particular
bilinear reduction; changing the kernel alone does not establish novelty.

This is **not universally a candidate-only offset**. With A=B=I and two query
and two candidate vectors e1,e2, the score matrix is I. Its cross-difference
`s11-s12-s21+s22=2`, whereas any additive `u(query)+c(candidate)` has zero
cross-difference. This disproves universal additive reduction, not a reduction
of the actual SEDS-relative residual for particular learned features.

The [AS-C07 forced-assignment reduction](../../proposal7/evidence/autonomous_search/AS-C07_candidate_screen.md)
depends on its own forced-edge construction. It cannot be extended to every
fixed-bank method. Likewise [AS-C43](../../proposal7/evidence/autonomous_search/AS-C43_result.md)
tests within-gallery score filtering, not the exact paired-anchor formula.
Neither distinction supplies evidence that anchors improve retrieval.

## Direct prior collision

Chen, Li, Sun and Liu (ACL2023),
[*Weakly Supervised Vision-and-Language Pre-training with Relative Representations*](https://aclanthology.org/2023.acl-long.464/).
Read publisher metadata/abstract and original PDF sections3.1–3.2 opening:
paired image/text anchors define coordinates using within-modality cosine
similarities; the resulting representations support cross-modal retrieval.
This directly anticipates the proposed information path. Their broader RELIT
pipeline constructs weakly paired pretraining data, not sentence-level SLRet.
No external effectiveness claim is transferred here and no pseudo pairs are
generated. [Original method text](https://aclanthology.org/2023.acl-long.464.pdf).

Search-bounded, not exhaustive: queries covered cross-modal retrieval/relative
representations, paired anchors/similarity, and relative latent spaces.
Other search hits are leads, not verified additional contributions. Publisher
identity and inspected mechanism are verified; code, complete experiments,
COI/retraction and bibliographic API checks were not performed. The source is
adequate primary evidence of the mechanism's prior existence, not replicated
evidence of PH gains.

## Four sequential perspectives and decision

- Novelty: generic paired-anchor retrieval is already explicit prior. Applying
  it to PH or changing anchor sampling is not yet a distinct mechanism.
- Experiments: exact pair shuffling could test dependence on anchor linkage,
  but beating a shuffled bank would not establish benefit over the incumbent.
  TRAIN-fitted checkpoints also prevent calling a held-out probe pretraining-clean.
- Signing: geometric neighborhood agreement is not sign meaning. No assumptions
  that a video's nearest captions are interchangeable or valid new positives.
- Retrieval: independent-query deployment is possible; no assignment leakage
  is inherent. A frozen bilinear/cosine control is essential, and both directions
  must improve under the official gallery if efficacy is eventually tested.

Strongest counterargument: an established primitive can contribute to a new
method. Correct, but this specification adds no observed SLRet bottleneck or
distinct causal intervention. Six guide gates therefore do not pass: bottleneck
missing; falsifiability possible; local separation only partial; direct generic
prior collision; no cheap empirical protocol fully specified; fair baseline
preservation plausible but untested. Do not train or extract features for it.

Operational read-only check: `artifacts/proposal7/phase2` is currently absent.
This is a specific missing historical cache, not a global asset census or proof
that no usable features exist. No recreation was launched and no user permission
is needed to resolve this rejected candidate. Recovered C27 assets were not
changed. No CPU model experiment, GPU, TEST, deferred-job polling or external
contact. Algebra above was inspected manually, not presented as a test run.

This completes a bounded new specification/collision screen. It changes the
next action by rejecting extraction for this generic anchor route, but supplies
no supported primary, method efficacy or SOTA claim. Do not retry with a kernel,
anchor-count, normalization or weighting sweep to bypass the missing mechanism.
