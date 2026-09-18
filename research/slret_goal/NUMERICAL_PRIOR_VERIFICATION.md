# Numerical-prior claim verification

## Material Passport

2026-09-19; academic-research-suite, inline source-verification phase. AI-assisted
reading of public primary-source HTML, no external model uploads. This is a
targeted check of two already-cited sources, not a systematic novelty search,
independent review, theorem-proof audit or human-read attestation.

## Sources and bounded reading

| Source | Directly inspected scope | Evidence fitness and unverified scope |
|---|---|---|
| [Mixed Precision Training, arXiv v3](https://arxiv.org/html/1710.03740v3) | Main text §§1–5, tables1–3 and reference text; equations in HTML | Controlled computational comparisons (designIII); primary evidence for its stated implementation. Figures not visually inspected; no local reproduction or separate venue/COI clearance. |
| [A Convergence Analysis of Adaptive Optimizers under Floating-point Quantization, v2](https://arxiv.org/html/2510.21314v2) | §§3–5, assumptions3.1/4.1–4.4, algorithms1–3, theorem statements4.5/4.6; abstract/introduction and selected related-work text | Mathematical analysis plus controlled experiments (III for empirical component). Appendices/proofs not audited; figure panels not inspected. GradeC provisional for scoped background, not proof of this campaign's behavior. |

Existence verified at the primary arXiv endpoints. No journal-quality score,
impact factor, predatory-publication clearance, ORCID verification or absence
of conflicts is inferred. For the first source, listed Baidu/NVIDIA affiliations
are author-provided context, not an independently verified financial disclosure.
For the second, COI/peer-review status remains unverified. Bibliographic API
verification was not enabled; there is no S2_VERIFIED status. No source rejected
as fabricated. Grades concern fitness for the narrow claims below, not a global
endorsement of every experiment or proof.

## Claim checks

**Established numerical remedies — SOURCE_VERIFIED.** Mixed Precision Training
§3 distinguishes FP32 master-weight updates, loss scaling, and FP32 accumulation.
Our moment-only correction has neither master weights nor a loss-scaling change;
do not label it a complete reproduction of that training recipe. Its scope also
precludes presenting ordinary precision safeguards as our algorithmic invention.
[Primary text](https://arxiv.org/html/1710.03740v3#S3).

**Theory directly explains our underflow — NOT SUPPORTED.** The quantization
paper §3.1 explicitly excludes underflow/overflow when motivating its relative
error model. Theorem4.5 additionally requires strict inequalities involving
first/second-moment relative errors and uses its specified schedule. §5 notes
exact arithmetic on quantized states as a limitation. Its result is relevant
background, not a guarantee for the observed FP16 zero-second-moment regime.
[Assumptions and theorem](https://arxiv.org/html/2510.21314v2#S3.SS1).

**Local mathematical implication — our inference, not a quoted theorem.** For
a positive reference value x represented as zero, |0−x|/|x|=1. Substituting a
second-moment bound qV≥1 into the cited strict condition makes its right side
β2(1−qV) nonpositive, while the left side is nonnegative: the condition cannot
hold. This does not refute the theorem; it identifies a scope mismatch. Our
arithmetic probes compare complete optimizer operations, not merely the paper's
quantization operator, another reason not to apply that theorem mechanically.

**Retrieval efficacy — LOCAL MEASURED_EFFECT only.** Evidence remains the
hashed real-gradient contrasts, matched continuation pairs, complete DEV seed
replication and locked PH TEST. Neither external paper independently verifies
these SLRet results. Our training contrast jointly changes first/second-moment
arithmetic; unique long-run second-moment mediation remains unproven.

## Routing consequence

Keep both references with their correct scopes. Do not add a convergence theorem,
new-optimizer claim, or claim that absence of underflow implies retrieval gain.
Send this limitation to the final claim audit. No training configuration, TEST
lock, gate or running extraction source changes follow from this source check.
