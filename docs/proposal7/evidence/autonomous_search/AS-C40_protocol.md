## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C40-v1

## Actual augmentation learning effect, Q21/layer G

AS-C39exactly reproduced original seed42first13updates, including all state,
logs and full scores. AS-C08inference/AS-C18gradients did not establish actual
word-swap training harm. Compare augmentation ON versus OFF for seeds42,1337,
2026, original baseline-only200epoch schedule, at a fixed endpoint:20epochs,
260updates, the end of original260step warmup. This is not a horizon sweep or
complete200epoch experiment. No claim about eventual training optimum.

Use the AS-C39archived function/dependency checks. OFF changes ONLY config
data.text_augmentation to null, causing original CiCoCollator augment=False.
Both policies still execute TWOtext encoder calls and the same dual-channel
four-CE objective; no shortcut for identical text, capacity/compute difference
or objective-weight change. Original train7096, batch512drop-last13steps/epoch,
per-seed shufflegenerator, cleaninputs, features, optimizer/scheduler, AMPbf16,
24threads and8loaderworkers all matched. Original release initialization is PH
fitted, not clean out-of-fold; no AS-C20adequacy claim or residual supervision.

Do NOTchange configepochs from200. Seed42ONepoch0must pass AS-C39score/model/
optimizer/scheduler/RNG/sampler replay, failing closed before other conclusions.
For all runs, capture initial model-state tensor digest after construction and
record per-epoch ordered training batch IDs plus clean-input digests. OFF/ON
must have identical initial states, batch order and clean inputs for a seed.
Augmented-input digests and changed-row counts are recorded separately.
Recording must not consume RNG. No data edits, new positives, test or upload.

Historical trainer does not evaluate initialization. Preserve that training
trace; evaluate initialization in a separate model load after training, using
the same original full-gallery scorer. Record its score matrix as comparison,
not an extra checkpoint-selection opportunity. Both policies share the same
initialization matrices within each seed, verified directly.

Evaluate DEV once per epoch as in original trainer, but endpoint is fixed260.
Retain all20compact epoch metrics, save epoch0and19score matrices plus separate
initialization. No best-epoch choice. Checkpoint-save callback writes NO model
checkpoint and ends after epoch19last-save boundary. No200epoch completion claim.
Logs/config/provenance in six new folders. Preserve terminal failures separately.

After all six runs, compare OFF minus ON at260updates per seed: official
T2V/V2TR1/R5/R10, strict ranks and persistent-rank deltas, full-score hashes,
tie-aware10000draw315filename-prefix cluster bootstrap(seed20260915). Intervals
conditional on fixed gallery, not search/selection uncertainty. Report original
strong singles and AS-C32ensemble reference, plus own initialization. No dev fit.

Training-harm diagnostic lead requires OFF minus ONmean>=+.5pp and CI lower>0
in>=2/3seeds, no directionR1loss>.25pp or R5/R10loss>.5pp in those seeds,
persistent ranks improve both. Separately require beyond-initialization gains
and report comparison with stronger existing baselines before any candidate
promotion. Even a lead is not GO: ordinary augmentation removal is NOTnovel;
derive3materiallydifferent candidates, collision-search and adversarial review
would still be required. If unsupported, no schedule/augmentation-strength sweep.

## Execution and monitoring

    PYTHONPATH=shared:methods/elsc:. timeout 900 /home/haipd/miniconda3/bin/python -m methods.information_probe.augmentation_training_probe --seed 42 --condition on

Same command for all3seeds and on/off, sequential GPU runs, ON42first. Expected
~3–5minutes each,~43GBGPU,900s hard per-run timeout. Monitor30–60s, compact
epoch progress. All six planned up front; no seed/policy selection. Final report
with11/11fallacy scan and all failures. No original outputs overwritten.
