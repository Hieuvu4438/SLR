# AS-C20: bounded PH-unfitted TRAIN learning calibration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run (autonomous research authority in governing loop)
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (preregistered before execution)
- Version Label: AS-C20-v1

## Question and scope

Does the exact AS-C19 generic-CLIP initialization learn retrieval on its fixed
internal source partition, leaving substantial *strict, different-text-input*
held errors? This is infrastructure calibration, not a method, teacher pipeline,
benchmark change, or revival of AS-C07. No official dev/test inputs or labels,
PH retrieval release/checkpoint, cached R0 representations, correction targets,
assignment, or additional supervision are permitted.

Use AS-C19's immutable 5,721-fit / 1,375-held TRAIN partition. Inferred source
prefixes are disjoint, not independently verified recording IDs. The known 72
held rows sharing exact fit text remain visible; positives stay singleton.
Generic pretraining overlap is unknown. Feature extraction provenance is the
AS-C19 audit, not a fresh per-pickle content verification.

## Fixed training and evaluation

- Seed 42, 1,000 updates, batch 128, shuffled fit rows, drop incomplete batches.
- All active retrieval parameters trainable; float32 throughout, no AMP/TF32.
- AdamW beta=(.9,.98), epsilon=1e-6, weight decay=.001 except biases/1D;
  AS-C19 random tensors lr=1e-4, copied generic tensors lr=1e-5.
- Linear warmup 100 updates then cosine to zero at update 1000; norm clip 1.
- Shared deterministic one-swap caption augmentation, epoch changed each pass.
  Video encoded once, clean and augmented text separately. Four CE losses:
  A-clean, transpose A-clean, B-augmented, transpose B-augmented, average.
- Evaluate steps 0/250/500/1000, no early selection or tuning. Final checkpoint
  is update 1000 irrespective of metrics. Save scores/learning curve and final
  model only (no large new input caches). CPU input cache may be used.
- Held gallery is all 1,375 held rows. Fit diagnostic gallery is 1,375 fit rows
  selected by ascending SHA256(pair_id), then original index order. This matched
  gallery size makes fit/held comparisons less confounded, but fit results are
  explicitly a subset, not full 5,721-gallery retrieval.
- Clean evaluation, .5(A+B); encode batches128, score blocks64. Direction-specific
  official singleton tie kernels; report R1/R5/R10 and ranks. Strict residuals
  require some different-clean-token-key confuser scoring ABOVE the positive;
  ties and identical-text ambiguity do not qualify. This excludes known identical
  text inputs, but does not establish semantic correctness of any error label.

## Fixed adequacy criteria and decisions

At update 1000 require BOTH directional fit-subset R1 >=80%, BOTH held R1 >=50%,
and mean held R1 improves >=5 percentage points from initialization. Also require
>=50 strict different-text-input held errors in EACH direction. These are
practical diagnostic gates, not validated strong-baseline-equivalence thresholds.
Passing permits further residual *diagnosis* only, never correction training on
these same confirmation labels or a GO claim. Failing the learning gate means
insufficient calibration under this budget, NOT absence of information or
residual supervision. Failing the error-count gate does not prove sufficiency.
No seed aggregation, hypothesis test, novelty claim, B0/B1/B2/B3 method pilot,
or inference about R0 is licensed by this run. Any later training continuation
needs a new disclosed protocol; do not reduce these thresholds post hoc.

## Execution and monitoring

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.clean_train_calibration`
Working directory `/home/haipd/SLR`. Hard timeout 3,600 seconds, monitored via
process plus JSON progress every25updates and each evaluation. Learning plateaus
are advisory. Numerical/nonfinite failures stop and are preserved as failures,
not negative research results. Autonomous authority permits disclosed, separately
identified corrective attempts, never silently overwriting completed/failed runs.
Outputs `AS-C20-TRAIN_run.json`, and task-specific scores/checkpoint under
`artifacts/proposal7/phase2/AS-C20/`. Expected <1GB disk, bounded GPU<49GB.
