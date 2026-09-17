# CMCM covariance derivative audit

## Material Passport

- Origin Skill: academic-research-suite, source fact-check / inline execution
- Date: 2026-09-15
- Verification Status: PLANNED before numerical execution
- Role: source fidelity prerequisite, not a candidate or efficacy experiment

Pinned CMCM commit5d458719d1da2f082e188cc44705003d919e7e97.
Read its MPNCOV/MPNCOV.py completely. The Sqrtm forward normalizes covariance
by trace+1e−5; custom backward divides by trace and trace². Hypothesis: this
produces a derivative discrepancy for valid positive-semidefinite covariance
inputs, especially when trace is comparable to epsilon. Do not compare a
nonsymmetric arbitrary matrix to a symmetric-domain custom derivative.

Extract exact committed Covpool, Sqrtm, Triuvec AST classes, no source edits.
Test their composed output on FP64 feature inputs[1,3,2,2], centered in four
spatial observations. Fixed torch seed20260917 creates one full-rank base and
one output contraction tensor. Scale the base so covariance traces equal
1e−7,1e−6,1e−5,1e−4,1e−2,1. Three Newton–Schulz iterations, exactly as TMCP.
The four-observation centering matrix has exactly representable binary entries.

Control: independent differentiable implementation of the SAME finite-iteration
forward, including epsilon and post-scaling; not an exact eigendecomposition
substitution. Compare forward outputs, source/reference feature derivatives,
and full composed central finite differences at h=1e−5*||X|| and half that h.
All perturbations pass through covariance formation. Report every scale and
both absolute/relative errors; error is relative to reference gradient norm.
Also inspect zero-feature output/backward finiteness, without treating that
degenerate input as representative of trained covariance activations.

Evidence gate: forward parity<=1e−12 and reference/FD relative error<=1e−5;
source/reference relative gradient error>1e−3 establishes a bounded numerical
fidelity discrepancy, NOT measured harmful learning or retrieval weakness.
No epsilon tuning, repair implementation, training, dataset, checkpoint or TEST.
Do not reopen generic covariance, nuisance or causal-invariance candidate families.
Absent real activation distribution and a runnable matched baseline, no method GO.

Command: PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m
methods.information_probe.cmcm_covariance_audit. CPU, two threads, timeout120s;
report exact source/code/protocol hashes and every tested case. No upstream imports
beyond AST-extracted classes, and no downloaded assets.
