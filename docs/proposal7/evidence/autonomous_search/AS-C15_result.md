# AS-C15 — robust cohort headroom, bounded expressivity inference

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (synthetic algebra and saved-result replay verified)
- Version Label: AS-C15-result-v1

## Evidence and result

`AS-C15_protocol.md` was recorded before running. Three complete dual LP solves
and60 fixed perturbations completed exit0 in29.98s. No crash, retry, dropped
condition, changed threshold or test access. Exact outputs/provenance:
`AS-C15-COHORT-AUDIT_run.json`; compact results and11/11 fallacy checks:
`AS-C15-SUMMARY.json`. Command in protocol. Small offsets/assignments are saved
as `AS-C15-s*-DEV_COHORT_ONLY.npz`, not training targets or production parameters.

### What the dual certificate does and does not show

For maximum-weight assignment, feasible dual potentials satisfy
`u_i+v_j >= S_ij`; optimal selected edges have equality. Consequently,
`S_ij-u_i-v_j` has every assigned edge at a non-strict row AND column maximum.
All three observed matrices have zero reported duality gap, zero maximum
constraint violation and zero selected-edge residual in CPUfloat64. All519
selected edges per matrix receive this non-strict support.

This is **not exact independent-rank reproduction**. The selected dual solution
has strict row margins for68.02%/68.40%/69.75% of assignments and strict column
margins for only1/519 in each seed. Dual solutions are nonunique; these strict
fractions are solver-specific, not intrinsic bounds. Independent argmax ties
can disagree with the assignment. No tie-rounded score matrix is promoted as a
new retrieval result.

The potentials consume all evaluation queries. A two-query synthetic example
keeps the focal score row `[2,1]` fixed while changing the other row from
`[100,0]` to `[-100,0]`; its cohort assignment changes from candidate1 to0.
An isolated-query function cannot reproduce both outcomes without access to
the changing cohort. Thus algebraic additive support does not eliminate the
extra-resource assumption or turn this into the rejected fixed-bank method.

### Numerical stability

Uniform pair-score jitter±2e-5 is applied ONLY to assignment selection. Full
gallery promotion remains the AS-C07 operation on original scores. At R1 its
bidirectional correctness is exactly the fraction of assigned paired IDs;
there is one strictly promoted edge per row/column. All original AS-C07
full-cohort R1 values reproduced exactly before jitter.

|Backbone seed|Baseline mean R1|Original cohort R1|Jitter mean R1|Jitter R1 range|Minimum gain pp|
|---|---:|---:|---:|---:|---:|
|42|75.240848|83.622351|82.620424|82.080925–83.429672|6.840077|
|1337|74.759152|83.044316|82.042389|81.502890–82.851638|6.743738|
|2026|74.951830|81.695568|81.078998|80.539499–81.695568|5.587669|

Average gain over the three baselines is6.929994pp across all fixed jitter
realizations, versus7.803468pp for the original deterministic assignment.
These dependent perturbations are NOT independent training replications or a
confirmatory significance test. Original-objective regret was bounded by
`2*n*amplitude` and checked in every realization.

After observing the sensitivity, an explicitly **post-hoc descriptive** audit
classified changed assignments by exact deployed32-slot text token IDs, using
the shared tokenizer. All239/239/234 changed-row occurrences across the20
realizations per backbone exchange candidates with IDENTICAL text inputs;
zero exchanges involve different deployed text inputs. Occurrences are not
unique samples. The dev set contains17 repeated-input rows. This locates the
observed jitter sensitivity at input-collision permutations, not linguistic
distinctions; it does not authorize changing official positives or PMGR revival.

## Research decision

Cohort headroom remains substantial after tiny-noise stress, but neither this
nor the dual supports a claim that a nonadditive independent-query scorer is
necessary. No new method follows. Existing fixed-bank correction, standalone
permutation loss and OTTER-like distillation rejections remain binding. The
search must target a separately measured independent-query gap, not relabel the
same cohort result as a novel reranker.

Validation:27 focused tests passed in1.06s, including three new dual/jitter/
query-isolation tests. No independent full-run replication is claimed. Previous
goal turn: progress (C13/C14); this turn: progress. No GO or global exhaustion.
