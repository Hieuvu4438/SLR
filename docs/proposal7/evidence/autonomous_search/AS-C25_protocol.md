# AS-C25 — exact inner-entropy contribution, not temperature tuning

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C25-v1

AS-C24 rejects hard/mean endpoints. A distinct analytic question: the current
E=sum(p*a), p=softmax(a/tau), equals tau*logsumexp(a/tau)-tau*H(p).
Does the implicit entropy subtraction improve rank-critical margins, or does
removing it help? Its derivative is p_i*(1+(a_i-E)/tau), which can be negative.
This algebra alone is not measured harm, a linguistic defect, or novel pooling.

Keep tau=.07, existing R0seed42 frozen representations, full519dev gallery,
legacy masks/inner competitors and outer averages. Compare expectation to
log-mean-exp U=tau*(logsumexp(a/tau)-log(N)) in A, B, both channels separately.
The log(N) terms are global constants (Ntext32,Nvideo65), so this is rank-equivalent
to adding tau*entropy to E. Check the exact algebra numerically and verify that
the resulting per-channel ranking is identical to E+tau*H up to numerical ties;
report any near-tie discrepancy rather than silently choosing favorable arithmetic.
The implementation's primary score is direct U, not a selected formulation.

Record negative derivative fractions over active outer rows for positive pairs
and fixed strongest T2V/V2T confusers. These are token-similarity coordinate
derivatives, NOT feasible isolated encoder perturbations or parameter gradients.
No sampled subset or matching significance claim; all519pairs per category.

Soft replay error<=2e-5 and exact directional rank parity required. Algebra error
<=5e-5 AFTER logit scaling. Fixed lead gates per variant: mean R1 gain>=.5pp,
neither direction loses>.25pp, R5/R10 losses<=.5pp, persistent mean ranks improve
both directions. All three variants reported. No sweep, training, dev labels as
targets, test access or new positives. Ordinary log-mean-exp is not a new method.
If this second scorer-cycle also fails, switch causal layer; do not rescue via
entropy coefficients, masks, confidence weighting or temperature selection.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.entropy_term_probe`
Cwd `/home/haipd/SLR`; timeout300s; monitor process plus`AS-C25-ENTROPY_run.json`.
Small local matrices/metrics only. Failures remain visible and distinct from
negative scientific findings under autonomous execution authority.
