# AS-C15 — cohort expressivity and numerical-tie audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C15-v1

Question: does AS-C07's assignment gain demonstrate a nonadditive score
requirement, and is the measured gain an artifact of tiny score perturbations?
This is an analytical/label-free diagnostic, not another candidate correction.

For a square score matrix S, maximum-weight assignment has the dual
`min sum(u)+sum(v) subject to u_i+v_j >= S_ij`. At optimum every assigned
edge is tight. Thus `S_ij-u_i-v_j <= 0`, with assigned edges equal to zero:
every selected edge is a NON-STRICT row and column maximum. This does not
guarantee independent argmax tie breaking recovers the assignment. Potentials
depend on the whole evaluation query cohort and are NOT train-bank parameters
or ordinary independent-query inference. No label enters this LP.

Fixed checks on three historical PH-dev score matrices (519 each):

1. Reproduce AS-C07 full-cohort assignment R1 exactly.
2. Solve the dual with SciPy HiGHS, fixing u0=0 for the additive gauge.
   Numerical feasibility, duality gap and selected-edge residual tolerance1e-7.
   Report assigned-edge row/column margins and non-strict support; do not
   misreport floating-point argmax ties as an independent retrieval method.
3. Twenty deterministic seeds0–19 of independent uniform[-2e-5,2e-5] pair-score
   noise, matching only. The amplitude is the existing scorer-parity tolerance,
   fixed before results. Promote the resulting assignment in the ORIGINAL
   full-gallery scores exactly as AS-C07. Report all60 R1 values, assignment
   changes and original-objective regret. No noise realization selected.
4. Synthetic checks: dual certificate on random and tied matrices; unchanged
   focal query with changed other query can require a different cohort match.

No test access, encoder training, positive changes, fitted dev-label parameters,
or new method. Labels only evaluate assignments after solving. No bootstrap
or significance claim from these dependent perturbations. Their20 repetitions
are not20 trained models. Small-jitter stability cannot validate one-to-one
capacity or semantic identifiability.

Output: `AS-C15-COHORT-AUDIT_run.json`; small dual/assignment artifact explicitly
marked dev-cohort diagnostic only. No derived production score or training target.
Command `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.cohort_dual_audit`
from `/home/haipd/SLR`, hard timeout30min; LP limit120s per solve; monitor session
and JSON≤60s. Preserve failures, no silent restarts or result overwrites.
