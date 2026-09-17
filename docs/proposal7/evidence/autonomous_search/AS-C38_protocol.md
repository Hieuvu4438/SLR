## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C38-v1

## Fixed numerical batching control, layer J / Q29

AS-C31 found same-input text-token changes between full128 and shorter tail
batches. Does the deployed519-query retrieval depend on this numerical choice,
and does the fixed AS-C32ensemble gain survive? This is computational validity,
NOT a novel batching method, input precision tuning or acquisition rescue.

Use original PH519DEV only, original3best checkpoints42/1337/2026, nativeFP16,
canonical64video/32text inputs, no augmentation, eval mode, same ordered IDs.
For each checkpoint reencode with original batches128,128,128,128,7 and
require exact full-gallery score equality to its AS-C32-hash-checked original.
Then reencode with the same four full batches and the final7real rows followed
by121copies of its first row. Pad every encoder input tensor consistently;
discard filler outputs BEFORE concatenation/scoring. Gallery remains519x519.
Both video and text encoders get this fixed control; no per-tower selection.
Full512real outputs must be exact between policies, masks identical throughout.
Report tail output deltas separately for both towers, not only text.

Repeat the padded encoding and scoring once per checkpoint and require exact
tokens/masks/CLS/scores. This tests reproducibility, not independent method seeds.
All scores use unchanged128-block scorer, original per-checkpoint scale. Save
original and padded matrices per seed, plus their two fixed uniform means;
8matrices total,9actual full score calls. No encoder cache/checkpoint writes.
Manifest, checkpoint, parent score and source hashes retained. Existing AS-C32
ensemble must replay exactly. No score, checkpoint or policy selection.

Report officialR1/R5/R10 in both directions, optimisticT2V diagnostic ranks,
official tie-expanded T2V denominator, V2Ttorch tie ranks, per-query rank changes,
R1gained/lost indices, maximum score delta, and persistent-rank deltas. Directly
validate all8matrices with official evaluator. For fixed padded ensemble minus
padded seed42 and padded ensemble minus original ensemble, reuse10000-draw
filename-prefix cluster bootstrap with seed20260915, tie expansion preserved.
Intervals are descriptive and conditional on the fixed gallery, not independent
training or repeated-search uncertainty. No significance selection.

Any changed rank establishes bounded numerical rank sensitivity. No changes
establish stability only to this fixed tail treatment, not all batches/hardware.
Assess whether AS-C32lead gates still hold with padded scores:mean>=+.5pp,
CI lower>0, eachR1>=−.25pp, eachR5/R10>=−.5pp, persistent ranks improve both.
Passing is one ordinary3-model ensemble control, NOT novel GO or3method seeds.

## Execution

    PYTHONPATH=shared:. timeout 300 /home/haipd/miniconda3/bin/python -m methods.information_probe.batch_shape_probe

Expected tens of seconds,300s hard timeout, progress per seed, process checks
30–60s. Output AS-C38-BATCH-SHAPE_run.json and8score matrices; no overwrite.
Unit-test padding/order/dtype/nonmutation/full-batch identity and synthetic
rank/tie sensitivity before launch. Fail closed on parity or provenance error;
preserve failed output and disclose repairs. No training, test split, external
upload, changed positives or main-method claim. Final11/11fallacy scan.
