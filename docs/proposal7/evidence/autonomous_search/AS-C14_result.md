# AS-C14 — mask-factor result

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (identity and scorer-unit parity verified)
- Version Label: AS-C14-result-v1

## Experiment result

Completed exit0,1.41s, peak GPU296833024bytes. Command and fixed factors in
`AS-C14_protocol.md`; provenance, source hashes and all results in
`AS-C14-MASK_run.json`; eight complete metric files/score arrays retained.
No crash, retry, tuning or omitted factorial cell. Dev contains519 video CLS
positions,3400 video padding positions,7828 text padding positions. Frozen R0.
No train/test access in this experiment, new model or changed official positives.

Identity maximum channel error1.5258789e-5≤2e-5; per-query ranks identical.
Joint exclusion agrees with independent masked scorer within3.8146973e-6.
Unit test verifies channel isolation: text padding affects A only; video
CLS/padding affect B only. Both use the unchanged historical outer masks.

|Exclude video CLS|Exclude video pad|Exclude text pad|T2V R1|V2T R1|Mean R1|Delta pp|
|---|---|---|---:|---:|---:|---:|
|No|No|No|74.181118|76.300578|75.240848|0|
|No|No|Yes|72.447013|75.144509|73.795761|−1.445087|
|No|Yes|No|74.181118|75.337187|74.759152|−.481696|
|No|Yes|Yes|73.025048|74.566474|73.795761|−1.445087|
|Yes|No|No|73.988439|75.722543|74.855491|−.385356|
|Yes|No|Yes|72.447013|75.529865|73.988439|−1.252408|
|Yes|Yes|No|74.373796|75.337187|74.855491|−.385356|
|Yes|Yes|Yes|72.639692|74.759152|73.699422|−1.541426|

R5/R10 and persistent mean-rank differences are in the run JSON. Video-pad-only
exclusion improves persistent mean rank by.2826T/.2299V despite worse mean R1;
do not claim every metric worsens. Text-pad-only exclusion worsens persistent
mean rank by.25T/.8276V. The largest single-factor mean-R1 loss is text padding,
not video padding. Effects are non-additive; these are fixed input-path
interventions, not three independently additive mechanisms.

Decision: no inference-mask method survives even the first R1 screen. This is
not a clean training-consistent comparison, a universal argument for keeping
padding, or a novelty result. No new global/local-mixture or mask-loss proposal.

## Statistical validation

11/11 fallacy types checked; descriptive/exploratory, not confirmatory:

1. Simpson: all factorial cells and both directions retained; persistent rank
   versus overall R1 disagreement explicitly disclosed.
2. Ecological: no individual linguistic claim from aggregate retrieval effects.
3. Berkson: fixed selected PH backbone; no population generalization.
4. Collider: no statistical conditioning or adjusted causal estimate.
5. Base rate: full519 gallery and all pad counts disclosed; no success subset.
6. Regression to mean: no treatment gain inferred from selected persistent errors.
7. Survivorship: all8 cells completed and retained; no omitted failures.
8. Look elsewhere: repeated dev exploration; no p-values, winner selection or GO.
9. Forking paths: fixed3-factor protocol before execution; no tuned mask rule.
10. Correlation/causation: intervention identifies this scorer's sensitivity only;
    no claim about the causal benefit of training with a new policy.
11. Reverse causality: no inference about learned encoder behavior from these
    fixed inference-mask substitutions.

Full independent experiment rerun not performed. Focused suite24passed in1.04s;
compileall and git diff --check passed. No Proposal8; research goal remains active.
