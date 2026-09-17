# Q01: positive controls do not repair the missing adequacy evidence

## Material Passport

2026-09-16; academic-research-suite, inline source verification and analysis.
ANALYZED, not an experiment or a method candidate. AI-assisted; no human
linguistic review or independent reviewer. The previous turn was progress:
the all37 audit corrected coverage and its analytical follow-up removed two
overbroad expressivity arguments. This note resolves its Q01 control question.

## Decision

Do not launch an injected-label, teacher-copying, random-label or MDL campaign
as a purported repair of Q01. None of these, by itself, supplies the missing
evidence that the probe can recover real held-out retrieval distinctions.
The existing runs already establish narrower execution/learning controls.

This is **not** a finding that an adequate probe is impossible. It closes the
specific idea that another generic positive-control exercise would make the
existing negative readouts evidence of information absence. The original
held adequacy threshold and the method GO gates remain unchanged.

## Current code and evidence rechecked

The full [readout trainer](../../../../methods/information_probe/train_readout.py)
was inspected through its main entry point. It learns from TRAIN only, selects
on DEV, and always evaluates `baseline + readout`. The C05 coefficient-zero
condition removes the baseline only from its training objective. Therefore a
successful standalone TRAIN discriminator is not a direct test of the deployed
combined rule. This discrepancy was deliberate and already disclosed; it is not
a newly discovered implementation bug.

- [C01 summary](AS-C01-R1_summary.json): exact baseline-preserving initialization
  and all nine selectors at step zero. This validates that null readouts preserve
  the original model, not that informative readouts can learn held-out residuals.
- [C04 support](AS-C04-TRAIN-SUPPORT.json): zero different-input nonpositive T2V
  hard-pair margins and one in V2T. These are fixed original confusers, not a new
  clean held population. The PH-fitted representation cannot become out-of-fold
  merely by withholding rows from the later readout optimizer.
- [C05 summary](AS-C05-SUMMARY.json): aggregating the six informative coefficient-
  zero runs gives distinct-input TRAIN pair accuracy **78.727487–82.580455% T2V**
  and **77.103632–83.407145% V2T**. All six selected DEV gains are zero. The six
  coefficient-one runs have selected gains from zero to 0.192678 pp. These
  extrema were recomputed from saved summary fields, not by running models.
- [C02 registered controls](AS-C02_crossed_screen_protocol.md) already include
  same-capacity spatial, pooled, shuffled, existing-token and zero arms;
  [completed summary](AS-C02-CROSSED-SUMMARY.json) fails attribution. Shuffled
  TRAIN information with aligned DEV information is not an independent positive
  calibration of the probe's sensitivity to an actual missing linguistic cue.
- [C20 result](AS-C20_result.md), [C30 result](AS-C30_result.md) and
  [C42 result](AS-C42_result.md): learning and exact replay are established in
  their stated scopes; the clean held residual model still fails its original
  50%-per-direction adequacy. High fit recall does not override that failure.

The bounded C05 training signal is real. It neither proves the pipeline broken
nor proves semantic sufficiency. Repeating it with an easier target would answer
less than the unresolved question.

## Targeted primary-source check

Search date 2026-09-16. Queries were the three exact titles below, restricted in
query text to ACL Anthology. Only official conference records and their PDFs
support this note; secondary search hits were not used. Metadata/DOIs verified
on the official records, not through independent DOI resolution. No downloaded
local PDF, full-paper read, external-result replication or novelty clearance is
claimed. All three are computational NLP studies, not SLRet efficacy evidence.

| Verified source / read scope | Relevant result and boundary |
|---|---|
| [Hewitt & Liang, *Designing and Interpreting Probes with Control Tasks*, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1275/), DOI 10.18653/v1/D19-1275. Official abstract and PDF §2 including §§2.1–2.3. | Their controls assign random outputs to word types to assess memorization/selectivity. This is not a positive test of recovering our missing sign distinctions; a random sentence-pair task would need its own interpretation. |
| [Pimentel et al., *Information-Theoretic Probing for Linguistic Structure*, ACL 2020](https://aclanthology.org/2020.acl-main.420/), DOI 10.18653/v1/2020.acl-main.420. Official abstract, PDF §§2.2–2.3 and Assumption 1 excerpt. | Probe cross-entropy gives a lower bound on mutual information. Poor prediction can mean a loose bound, not absent information. Their uniqueness assumption is not established globally for our finite-precision visual pipeline. |
| [Voita & Titov, *Information-Theoretic Probing with Minimum Description Length*, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.14/), DOI 10.18653/v1/2020.emnlp-main.14. Official abstract, PDF §2.2.2 and adjoining §2.3 excerpt. | Online coding trains on growing labeled blocks and scores subsequent blocks, accounting for learning effort as well as prediction quality. Our fixed-TRAIN, repeated-DEV logs are not such a sequence; summing their losses would not retroactively produce the proposed online codelength. |

The papers address different quantities: memorization-relative selectivity,
information availability, and learning efficiency. They should not be presented
as unanimously prescribing one universal probe or one adequacy threshold.
Their methods do not validate our arbitrary 50% threshold as a universal law;
that threshold remains the existing project's preregistered investment gate.

## Why a failed probe does not upper-bound available information

For a properly defined population target Y, representation R and probabilistic
probe q, the cross-entropy identity is

`CE(q) = H(Y|R) + E_R KL(p(Y|R) || q(Y|R))`.

Hence `H(Y) - CE(q) <= I(Y;R)`. This is the lower-bound direction in
[Pimentel et al., §2.3](https://aclanthology.org/2020.acl-main.420.pdf).
A bad q can give a weak lower bound even when information is available.
Empirical finite-sample cross-entropy additionally has estimation uncertainty;
the population identity is not an exact finite-data confidence bound.

Project inference: R@1 is not this cross-entropy, and heavily reused DEV is not
independent confirmation. Neither the identity nor a weak held R@1 supplies an
information ceiling. No MI number is estimated here, and no ceiling is inferred
from C36's finite inventory of distinct inputs.

## Positive-control decision matrix

These are assessed diagnostic specifications, **not three proposed methods**.

| Control | What success could establish | Why it does not resolve Q01; action |
|---|---|---|
| Exact baseline/null replay | Input alignment, scoring/evaluation parity, zero-initialization contract. | Already established. Does not test learning a correction; no repeat. |
| Fit or memorize existing TRAIN pairs | Optimization can fit that training population. | Already observed in C05/C20/C42. Does not prove held learning or adequate residual support; no easier-target repetition. |
| Inject a known label bit / paired-ID code | Harness can recover an artificial accessible signal in its chosen location. | Target/input distribution changes and cue difficulty is stipulated. Would validate only a fixture, not real missing information; do not inject DEV IDs or report resulting retrieval as efficacy. No launch. |
| Distill the existing frozen scorer | Probe can approximate the teacher's function on held inputs if it succeeds. | Can inherit the very errors under study; successful copying is not recovery beyond the scorer. Teacher distillation also has prior-art/closure issues if promoted to a method. No launch. |
| Corrupt a known successful score and learn to undo the corruption | Recovery of that known corruption under its controlled distribution. | Artificial corruption need not model the real failure. Useful only with a separately established corresponding source defect, which is not present here; no generic corruption sweep. |
| Random-label control/selectivity | Memorization relative to the specified label construction. | A negative/task-relative control, not proof of sensitivity to true held residuals. Do not change official positives or relabel its result as positive calibration. |
| MDL/online coding | Learning effort under a newly specified sequence of training blocks and held predictions. | Requires new runs and provenance, not the old loss trace. PH-fitted release initialization compromises a clean unseen-row interpretation. Does not itself supply a new retrieval mechanism; no campaign. |
| Existing compatible human cue labels | Could validate that a particular recoverable distinction is actually represented and relevant. | Compatibility and conditional relevance must be demonstrated. Q17/Q37 checks do not currently supply such a join; new annotation remains out of scope. No fabricated labels. |

## Consequence for the research loop

Keep Q01 **unresolved**, but stop this generic positive-control detour. Another
probe must begin with a concrete phenomenon and an admissible intervention, not
with a hope that a more elaborate control can convert old negative results into
an information-loss claim. The source-led search remains the user's priority.

An optional user clarification has been requested about whether the blanket
Proposal1–6 closures remain binding for newly evidenced source mechanisms.
**Until an explicit answer changes them, they remain binding.** This is not
approval to reopen any family, launch training or change the original GO gates.
The current work supplies no novel candidate and no global exhaustion proof.

Do not repeat this adequacy note, the all37 inventory or Q02/Q22 existence proof
as another progress cycle. Subsequent work must add new source/data evidence,
resolve a specific existing resource dependency, or receive a material scope
decision; relabeling unresolved questions is not progress.

## Integrity and validation

No model execution, new DEV evaluation, data/TEST read, asset acquisition or
upstream edit. Code reading and saved TRAIN/DEV summary analysis only. The
earlier quarantined public-comment TEST exposure is not erased by this scope.
No statistical efficacy inference, p-value, new model seed or method GO.

Input snapshots:

- C01 summary: `68655fc88dd5cb089ac22b2547aa06931620d682a5ce7af5d38560f2562a00af`.
- C04 support: `6354dbafe9d8513d186b1778ea106a7c7077f5a4d977055f66801cfc933ae5a0`.
- C05 summary: `6dea5122ab24413fa5a4432e941d547b87c3f423f31dba0a22c11556ef99fa0c`.
- C02 summary: `599bdc353cf20c30ee83b481a89f24830221224352a52927b5146e729cb6bbe1`.

Inline adversarial checks: (1) execution success is not sensitivity calibration;
(2) source methods answer different estimands and are not SLRet results;
(3) inability to nominate a valid positive control does not establish scientific
impossibility or license a terminal barrier. These limit the conclusion rather
than claim independent reviewer agreement. Goal remains active.
