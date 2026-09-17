## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (six matched runs; historical42control exactly replayed)
- Version Label: AS-C40-v1

## Result: registered augmentation-harm lead fails

All six registered condition runs completed260updates, the fixed end of the
original warmup, with original200epoch schedule retained. No seed passes every
training-harm diagnostic gate; requirement was at least2/3. Turning word-swap
OFF does not produce a consistent attributable improvement under this test.
This rejects the registered lead, not every possible long-horizon augmentation
effect. It is not a method candidate, GO or global research-exhaustion result.

|Seed|ON meanR1|OFF meanR1|OFF−ON pp|Conditional95%CI|All gates pass?|
|---|---:|---:|---:|---|---|
|42|67.437380|67.052023|−.385356|[−3.254438,2.446235]|No|
|1337|65.703276|69.364162|+3.660886|[.297604,6.866538]|No|
|2026|68.208092|66.666667|−1.541426|[−4.696146,1.530612]|No|

Seed1337improves bothR1directions and persistent ranks, but T2VR5falls from
88.439306 to87.861272 (−.578035pp), exceeding the allowed.5pp loss. It fails
even before considering the missing second successful seed. Do not weaken that
gate or promote the favorable seed while hiding the other two.

|Seed|ON T2V/V2T R1|OFF T2V/V2T R1|Persistent mean-rank change T2V/V2T (OFF−ON)|
|---|---|---|---|
|42|67.630058 /67.244701|68.015414 /66.088632|+.641304 /−.747126|
|1337|66.088632 /65.317919|68.786127 /69.942197|−1.000000 /−.367816|
|2026|67.244701 /69.171484|67.052023 /66.281310|−1.010870 /−2.563218|

Persistent sets remain the original
three-seed intersection; negative rank change means improvement. Persistent
rank movement and all-queryR1can differ; neither replaces the registered gates.
AllR1/R5/R10details and per-query ranks remain in the validationJSON.

## No improvement beyond initialization or stronger controls

All six separately evaluated initialization matrices are bit-identical:
mean74.759152,T2V74.181118,V2T75.337187. Inactive randomly initialized adapter/
head state may differ between seeds, but within each ON/OFFpair all initial
state tensors have the same digest and inference ignores those disabled heads.

|OFFseed|OFF−initialization pp (conditional95%CI)|OFF−strong original42 pp|OFF−AS-C32ensemble pp|
|---|---|---:|---:|
|42|−7.707129 [−10.444983,−5.038545]|−8.188825|−10.211946|
|1337|−5.394990 [−8.657628,−2.091255]|−5.876686|−7.899807|
|2026|−8.092486 [−11.007574,−5.148373]|−8.574181|−10.597303|

All OFFendpoints fall below initialization; all ONendpoints do too. The exact
historical42trajectory was already known to select epoch0. This experiment
isolates augmentation at a fixed horizon, not a new claim that every later
checkpoint is worse or that training can never improve the released model.
Removing augmentation is not an explanation sufficient to eliminate the observed
fixed-horizon degradation. Optimizer, schedule, precision and other hypotheses
remain separate; no rescue sweep or sufficiency conclusion follows here.

## Matched-treatment integrity

Each pair has identical initial model-state digest,260ordered batch lists,
clean-token and visual-input digests, scheduler state and initialization scores.
Resolved configs differ ONLY in augmentation. Every training record has the
same two text forwards,one video forward,zero teacher forwards,511in-batch
negatives and effective batch512. OFFstill computes both text branches; it is
not a compute-saving shortcut or changed four-CE objective.

Each run processed133120training-row occurrences. Changed ONtext occurrences
were65224,65210,65559 for42/1337/2026. OFFhad zero. No new data, positives,
test information, auxiliary ELSCobjective, teacher or external upload. Release
initialization is PHfitted, not clean out-of-fold; no AS-C20adequacy claim.

The full seed42ON20epoch numerical log trace matches history:280records,
including260training and20DEVrecords, exactly excepttime/memory. At epoch0,
model/optimizer/scheduler/scaler/RNG/sampler andscores match original checkpoint.
The old archived train function is used; no added initialization evaluation
before training. Initialization is evaluated separately after each completed
run, so it cannot perturb training RNG. No checkpointwrites or full200epoch claim.

All18saved matrices (epoch0,epoch19,initialization for each run) passed official
R1/R5/R10checks; saved epoch metrics agree with training/evaluation logs. No
best-epoch or favorable-policy selection. Batch/hash comparisons are exact,
not approximate aggregate-metric checks.

## Execution, failure and repair

Six successful runs consumed1577.354959seconds total, peak43,260,497,920GPUbytes.
Final validation exit0,2.897022seconds. Its immutable result SHA256 is
f43eb75ffd00cb600fc1cb0013e387fcb6beebce9be2240beda8681cdb028373.

Original2026ONfailed at169updates because progress-print stdout closed,
BrokenPipeError, after172.416172seconds. FailedJSON/folder/epoch0score/partial
logs retained. It is not a seventh replicate or a performance-selected exclusion.
The queue stopped and2026OFF had not begun. No resumable checkpoint was saved.

Output-only attempt suffix and durable stdout/stderr logging were added.
Removing those three worker-source additions reconstructs its exact prior hash;
no numerical code changed.2026ON-attempt2reproduced initialstate,169batchtraces,
13evalrecords and181logs through the failed prefix exactly, then finished.
2026OFF finished afterward. Both2026conditions use the same revised worker;
each other pair uses matching original worker source. See infrastructure repair
record for commands, hashes and failure details. No original output overwritten.
Final controller exited0; no active experiment remains.65focused tests passed
after the repair; compileall and gitdiffcheck passed.

## Statistical scope and fallacy scan:11/11 checked

Intervals use10000paired draws over315inferred filename-prefix clusters,
seed20260915, officialT2Vtie expansion preserved. They are conditional on this
fixed gallery and trained runs; not verified recording-cluster independence,
training-population uncertainty or search-wide confidence. Twelve contrasts
(four per seed) are reported without familywise correction as descriptive
intervals, not confirmatory discovery. The unadjusted favorable1337interval
cannot rescue the failed preregistered conjunction; no adjusted claim asserted.

1. Simpson: each seed and direction reported; mixed signs are not hidden by a
   favorable average over seeds.
2. Ecological: aggregate changes do not establish sign-level semantic damage.
3. Berkson: all declared seeds andfull519DEVgallery retained, no error subset
   chosen as the primary evaluation population.
4. Collider: matching is on predetermined inputs/state, not learned outcomes.
5. Base rate: three paired seeds,six endpoints,one preserved failed attempt,
   all row/augmentation counts explicit.
6. Regression to mean: fixed260endpoint, no favorable epoch or checkpoint picked.
7. Survivorship: failed run retained and output-only retry disclosed/verified;
   all six intended endpoints ultimately completed.
8. Look-elsewhere:12descriptive unadjusted contrasts and repeated-search scope
   disclosed; no one favorable seed promoted as discovery.
9. Forking paths: protocol before runs; later validator integrity checks and
   infrastructure repair do not change treatment, endpoint or gates.
10. Correlation/causation: controlled on/off supports local treatment comparison,
    not a universal semantic explanation or every-horizon augmentation verdict.
11. Reverse causality: augmentation policy and seeds fixed before outcomes;
    endpoint differences did not drive treatment assignment or retry choice.

## Research consequence

The earlier frozen perturbation and gradient evidence now has a genuine learning
intervention: the registered harmful-augmentation explanation fails this test.
No augmentation-strength, schedule, precision or seed sweep is justified as a
rescue. Do not promote ordinary removal/freezing/optimizer fixes into novelty.
Update Q21 and choose a materially different unresolved layer from the open map.
No AS-C41 launched. Broader information/learning limits and supported novel
method remain unresolved; goal active, noGO, noProposal8, no global exhaustion.
ARS influenced exact matched controls, unchanged gates, failure preservation,
and the distinction between a local treatment comparison and a novel method.
