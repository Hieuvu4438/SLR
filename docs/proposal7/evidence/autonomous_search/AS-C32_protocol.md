## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C32-v1

## Fixed stronger-control question

Does uniform score averaging across the existing selected baseline checkpoints
seeds42/1337/2026 improve ordinary independent-query retrieval and repair
persistent errors? This is a headroom/stronger-control diagnostic. Ordinary
ensembling is NOT proposed as a new method. One ensemble of3 models is NOT
three independent ensemble seeds. No new checkpoint selection or fitting.

Full519 PH DEV gallery, same official singleton positives and tie kernels.
Read original score matrices and evaluation metadata; verify paired query ID
order, ID hashes, checkpoint SHA256 and baseline metrics for all3. Existing
checkpoint selection is dev-dependent and exploratory; no independent-test
or population-general claim. No test data loaded.

Seven fixed variants: each single model; each single model repeated3times;
and uniform3-seed score average. Float64 elementwise accumulation then float32
output. Repeated-model means must reproduce originals bitwise in value.
Require identical ranks before/after casting the mean to float32; if violated,
preserve failure and diagnose numerical sensitivity, no tolerance/weight tuning.
No per-model scale normalization, learned routing, confidence weighting,
pairwise seed mixtures, cohort assignment, candidate bank or new encoder.

Compare primary ensemble with the strongest single model (expectedseed42),
and report all single/repeat metrics. Explicit cost3model encoder/scorer passes
and3model parameters/storage versus1; arithmetic here uses existing caches,
so its CPU execution time is NOT an inference speed benchmark.

## Outcomes and diagnostic gate

Report official bidirectional R1/R5/R10, full-gallery rank changes, fixed
persistent intersection and its recovered queries. A query can be wrong under
all individual models but correct under averaging when their competitors differ.
If one candidate STRICTLY outranks the positive under every model, positive
weighted score averaging cannot remove that candidate's advantage (check the
actual ranks and float32 rounding). Report this common-confuser limitation.

Conditional paired source-prefix cluster bootstrap,10000draws,seed20260915,
full gallery held fixed. Preserve T2V tie-expanded numerators/denominators,
not a substituted optimistic per-query primary. Inferred filename prefix is
not verified recording identity. CI does not capture training/checkpoint
selection uncertainty or repeated exploratory search.

Diagnostic lead: >=+.5pp mean R1 over strongest single; cluster95% lower>0;
neither R1 direction loses>.25pp; R5/R10 loss<=.5pp; persistent mean ranks
improve in both directions. This is NOT a method GO gate: no novelty, no
2-of3 independent trained method seeds, and no second-dataset confirmation.
Failure forbids weight/scale/seed-subset tuning to rescue this fixed ensemble.

## Execution

    PYTHONPATH=shared:. timeout 300 /home/haipd/miniconda3/bin/python -m methods.information_probe.seed_ensemble_probe

Working directory /home/haipd/SLR. Output AS-C32-ENSEMBLE_run.json and
artifacts/proposal7/phase2/AS-C32-uniform3_scores.npy. No overwrite. Expected
seconds, CPU only; process/output monitoring30–60s if needed. Unit tests cover
same-model identity, elementwise independence and official tie-aware bootstrap.
Report all failures, numerical issues, outcomes and11/11 fallacy scan.
