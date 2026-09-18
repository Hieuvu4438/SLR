# Measured results — baseline stage, 2026-09-17

## Joint RGB/head training smoke, 2026-09-18

`rgb-train-smoke-001` completed two fullB32 updates/64examples in49.903408s.
Both initialization features and replay features bitexact; both losses and head
gradient norms exactly match the firsttwo moment-control001 steps. Warmup step1
changes0 head/tail tensors; step2 changes651 head and36 tailtensors. Allfrozen
I3D weights/BNbuffers unchanged. Tailcheckpoint contains36FP32moment states,
each atstep2. Refreshed secondbatch-firstvideo features change0.0001142025 after
update, establishing actual encoder update/refresh, not recall benefit.

Stepwalls22.991151/17.408119s; encoding14.350548/9.789893s and replaybackward
5.644528/5.106071s on1922/1903windows. Only6/32 videos carrynonzero feature
gradient ineachbatch, but allwindows are processed; no selective exposure.
HeadCUDApeak17.879642GB and worker0.999408GB measured separately (not a measured
simultaneous peak); sampled workerRSS4.107GB midrun, not a measured RAMpeak.
Output86.558MiB includes nonresumable tailoptimizer/RNG, not headcheckpoint.
NoDEV/TEST or efficacy selection. Baseline fine-tuning pilot remains unrun.

## RGB gradient feasibility, 2026-09-18

Native SEDS firstB32 loss matches the completed control exactly;6/32 videos
have nonzero RGB-input gradients. Original firsttwo chosen videos havezero
gradients, so an explicitly amended activation check uses firstnonzero indices
3,7; it is not representative efficacy evidence. Native torch2.3/OpenCV4.11
replay misses feature parity(maxabs0.0003829002>0.0001), despite byte-identical
preprocessed frames. Original torch2.11/OpenCV4.13 runtime is bitexact; helper
and shared extractor agree in both. Changing cuDNN deterministic policy does
not explain the discrepancy. Failedruns001/002 preserved; no tolerance relaxed.

`rgb-cached-vjp-001`: real native representation gradients drive all36 existing
Mixed5b/5c parameter tensors (4,759,008parameters), finite/nonzero on both
activatedvideos, featuresbitexact, BatchNormbuffersunchanged,2.977s/0.917GBpeak.
Eight-versus-four window numerical test passes its tiny-absolute gate only:
maxabs2.327e-9,relative2.5726%. This is NOT strong relative equivalence.
Stronger `rgb-vjp-equivalence-001` normalizes actual VJP to unitL2 and compares
direct retained graph versus replay with identical8-window batching:features
bitexact, parametergradient relativeL2=1.0884e-8,maxabs9.3132e-10,1.423s.

Inference: fixed-microbatch two-runtime gradient caching is mechanically
feasible; actual two-update training/IPC and efficacy remain untested. TRAIN
census417322windows/827354frames; two-video worst-rate plus50% RGB margin/head
control projects10278s, excluding IPC/newoptimizer/DEV extraction. Not a fullrun
admission or average runtime estimate. No optimizer updates, DEV or TEST used;
generic end-to-end fine-tuning/GradCache are established techniques, not novelty.

## Scorer cost diagnostic, 2026-09-18

`seds-scoring-profile-001` completed27.004s,all parity gates pass. Native PH519
evaluation15.358s,score stage1.556368s,289fusion calls. Caching identical fusion
requires17calls/0.058722s. Five warmed synchronized score-stage timings:
allthree streams block32 median0.218735s; fusedonly block32 0.072491s;
fusedonly block128 0.084238s. Add fusion precompute when reporting cold scoring.
Block32 matrices are bitexact for allthree streams;block128 maxabs7.629e-6,
zero primaryrank changes. Raw extraction and online encoding are not removed.

Decision: optimized fused scorer <100ms, so the preregistered gate for investing
in new certified pruning FAILS for this PH regime. Do not move the threshold,
inflate the gallery or claim novelty from routine caching. Native-versus-cached
score-stage gains are engineering observations, not whole-pipeline speedups or
a validated alternative contribution. Normalized frozen cache retained for
reproducibility; no weight change, no new method, no TEST.

## Fresh CSL-Daily baseline replay, 2026-09-18

`cico-csl-dev-replay-001` completed6.340s,peak1.115GB,exit0. Exact selected
checkpoint tensors/config/manifest and2154feature hashes verified. Shared
groupedgallery1077videos/797texts: T2V R1/5/10=68.883312/81.806775/86.951066;
V2T65.923863/81.429898/86.908078;meanR1=67.403588. Max score delta0 andzero
historical rankchanges. This is fresh inference, superseding the weaker cached-
score-only evidence, but not a fresh selection split: epoch179 was historically
selected on this DEV. BSL5K+H2S-transfer-aware regime remains distinct from
target-domain release. No method, TEST access or independent confirmation.

## Material Passport

Origin: academic-research-suite, experiment-agent run, inline. Verification: runtime replay completed for the stated scopes. No method improvement or SOTA claim.

2026-09-18: `seds-continuation-smoke-001` completed2updates in30.162s;
step0 score and metric parity all3streams,changed tensors[0,664],peak18.769GB.
Full control001 is running: midpoint111 fusedR1=68.786127 both directions,
mean delta−8.766859pp against initialization;selector stillstep0. This is an
interim result,not final endpoint. `upret-load-check-002` independently
loads exact363tensor student checkpoint onCPU,3.931s,no dataset read. Its failed
attempt001 (vendoredGit layout) is retained. Neither establishes retrieval gain.

| Run / scope | T2V R1 / R5 / R10 | V2T R1 / R5 / R10 | Mean R1 | Interpretation |
|---|---|---|---|---|
| CiCo selected local single, official PH DEV519 | 74.181118 / 91.136802 / 95.375723 | 76.300578 / 91.907514 / 94.990366 | 75.240848 | `cico-ph-dev-replay-003`: max score delta0, zero changed ranks versus historical artifact. H2S-transfer-aware feature regime. |
| SEDS release fused, TRAIN first256 | 97.656250 / 100 / 100 | 97.656250 / 100 / 100 | 97.656250 | Resubstitution and subset gallery only; not a benchmark validation. |
| Same SEDS pose branch, TRAIN256 | 97.656250 / 100 / 100 | 96.875000 / 100 / 100 | 97.265625 | Native checkpoint branch diagnostic, not separately trained pose-only baseline. |
| Same SEDS RGB branch, TRAIN256 | 97.656250 / 100 / 100 | 97.265625 / 100 / 100 | 97.460938 | Native checkpoint branch diagnostic, not separately trained RGB-only baseline. |
| SEDS release checkpoint, adapted PH DEV519, fusion | 76.493256 / 92.870906 / 96.146435 | 78.612717 / 93.063584 / 95.375723 | 77.552987 | `seds-adapted-dev-eval-002`; input-transfer diagnostic, not release preprocessing or a method improvement. |
| Same adapted PH DEV519, pose branch | 61.078998 / 86.512524 / 91.714836 | 59.922929 / 84.778420 / 90.558767 | 60.500963 | Jointly trained branch, not independent pose baseline. |
| Same adapted PH DEV519, RGB branch | 73.603083 / 93.063584 / 96.146435 | 73.988439 / 93.063584 / 95.761079 | 73.795761 | Jointly trained branch, not independent RGB baseline. |
| Native SEDS continuation step111, fusion | 68.786127 / 89.017341 / 93.641618 | 68.786127 / 89.210019 / 93.448940 | 68.786127 | Rejected by initialization-eligible selector. |
| Native SEDS continuation step222, fusion | 71.098266 / 92.292871 / 94.990366 | 72.639692 / 92.292871 / 94.797688 | 71.868979 | No gain; step0 retained.222updates/7096examples,663.731s. |
| Same native continuation step222, pose | 56.647399 / 84.778420 / 90.173410 | 57.610790 / 83.044316 / 89.210019 | 57.129094 | Joint branch,not a separately trained baseline. |
| Same native continuation step222, RGB | 66.666667 / 89.017341 / 93.448940 | 66.088632 / 89.210019 / 93.641618 | 66.377650 | Joint branch,not a separately trained baseline. |
| UPRet corrected historical partial step767, PH DEV519 | 20.809249 / 45.279383 / 57.032755 | 18.882466 / 41.425819 / 54.720617 | 19.845857 | `upret-ph-dev-replay-001`:exact historical ranks,5.710s,peak0.895GB. Incomplete training,not B_release. |

Numerical exposure diagnostic `seds-moment-audit-001`: FP16 native Adam has
205,610,372 entries with nonzero first andzero secondmoment,97.5132% of FP16
elements;FP32 counterpart0%. All states finite. This does not prove recall
causality. A preregistered FP32-moment-only control is in activation testing,
not a new retrieval method. Native failed control is retained in full.

SEDS TRAIN256 run14.16s total, evaluation5.53s approximately, backward16 examples0.856s, peak7,994,248,704 bytes. All checkpoint tensors match exactly; native and shared metric kernels agree within1e−5 on R1/5/10/MeanR. All696 gradient-bearing parameter tensors have nonzero gradients. Total loss0.000285228; matching auxiliary loss0 on this batch. This is an activation/engineering smoke, not an optimizer or training-gain result. No optimizer steps run.

The first4-example smoke is successful but saturated (batch2 loss rounds to zero), so not used as positive evidence of trainability beyond backward execution. New B16 smoke provides stronger gradient evidence.

CiCo attempts001/002 failed before evaluation in the SEDS Python environment (restricted torch.uint32 unpickler, then NumPy2 pickle import). Attempt003 uses existing base environment torch2.11.0+cu128/NumPy2.5.3 and passes exactly. The local selected checkpoint was hash-verified before full deserialization. Failed attempts remain in ledger; no method search was performed by changing environments.

Raw reports/scores/IDs/hashes in `artifacts/slret_goal/`; metadata-label erratum for the first two SEDS run reports is documented in BASELINE_AUDIT. Read `*_metrics.json` for correctly named directions. Native logging itself is correct.

No multi-seed new training, matched method controls, ablation, independent confirmation or locked final test yet. Current evidence establishes a runnable SEDS training path and an exact CiCo dev reference; it does not establish a research contribution.

## Preprocessing reconstruction diagnostics

Two preregistered RTMPose-L hypotheses on24 original PNG frames from the first8
release-TRAIN videos (all same broadcast date, not a representative quality test):

| Configuration | Mean absolute XY error, px | Maximum score error | Disagreements at confidence0.4 /3192 | Release-equivalence gate |
|---|---:|---:|---:|---|
| 384x288, full-image bbox, no flip | 9.432752 | 0.638532 | 234 | Failed |
| 256x192, full-image bbox, no flip | 10.272252 | 0.660632 | 223 | Failed |

Actual model tensor loads are exact. Inference0.360/0.377s, peak175.3/173.4MB;
these are tiny extraction diagnostics, not retrieval/pose-accuracy benchmarks.
Failures before inference: attempt384-001 used missing distribution name
`mmcv` rather than `mmcv-lite`;384-002 hit MMEngine/PyTorch weights-only metadata
incompatibility. Both retained; explicit trusted-public-checkpoint load resolves
startup without changing weights. No further extractor guessing under this gate.

`rgb-cache-parity-001`:504 windows across the same8 train videos match native
SEDS retained-frame index sequences against existing CiCo cache window indices.
BSL5K agnostic cosine (window-weighted)0.883981; per-video maximum absolute
feature deltas0.635672–1.298705. No feature-equivalence claim; matching indices
does not certify PNG/MP4 pixel identity or original RGB extraction provenance.
Both cached streams fail direct substitution as native SEDS features.

Decision: explicit adapted pipeline in `ADAPTED_SEDS_PROTOCOL.md`, shared across
train/dev, with native frame filtering and per-window original-frame metadata.
Any released-checkpoint evaluation on it is an input-transfer diagnostic until
a matched trained adapted baseline exists. These mismatches do not establish a
model flaw, recall degradation, scientific NO-GO, or alternative contribution.

## Adapted full DEV execution and decision

`seds-adapted-dev-001` completed519/519,55,775 decoded frames,29,816 selected
windows in390.556s, peak1,133,859,840bytes, generated218,668,421bytes. All videos
are25fps, none loses frames under this extractor/native confidence filtering;
this remains explicitly different from the paper's stated24fps. All519 native
loader checks pass; per-window source-frame indices and per-asset hashes saved.
TRAIN smoke uses identical pipeline (2 videos,212frames128windows). Full TRAIN
is not yet extracted or trained.

Evaluation attempt001 failed before model loading: global NumPy2 module aliases
interfered with SciPy sparse C-extension initialization. The independent import
trace localized this cause; no model/data inference occurred. Attempt002 uses
a scoped PH-loader unpickler class-name mapping instead (no global NumPy change),
completes17.862s, native eval11.541s, peak2,948,302,336bytes. Strict checkpoint
contract has no missing/unexpected/mismatched tensors; native/shared metric
parity passes for all streams. T2V has0 positive-tie queries; each V2T stream17,
handled with the shared direction-specific official kernel.

Preregistered `seds-adapted-dev-branches-001`: fused meanR1 exceeds RGB by3.757225pp.
Of122 T2V fused errors,93 are common RGB/pose errors (76.23%); of111 V2T errors,
84 are common (75.68%). Fused failures with at least one successful branch:
29/27. Oracle branch union418/426 correct is diagnostic, not usable gain.
Decision: prioritize input/representation analysis; no evidence here to select
a fusion-degradation lead. Input-transfer confounding and jointly trained
branches prevent causal claims about the native architecture. No new mechanism
admitted and no NO-GO reopening. Do not call the difference from local CiCo a
method gain: feature/pretraining regimes differ. Do not compare DEV to paper TEST.

## TRAIN preparation and UPRet asset follow-up

`train-frame-census-001`:7096 TRAIN videos827,354 header frames,23 longer than300,
max475,none missing. Projected extraction5793s from DEV frame throughput.
`resume-smoke-check-001`: clean pause after1/2 videos followed by resume computes
only the second. Pose/RGB arrays and pickle hashes match the earlier smoke
exactly for both videos. Four CPU tests validate locking, hash mismatch rejection,
atomic report writing and recoverable orphan preservation. These are pipeline
checks, not retrieval improvements.

Full TRAIN attempt1 lost its process/session after169videos135.527s without a
recovered exit reason. Original process/timeout absence verified before resume.
Attempt2 uses the unchanged recipe/source contract, verifies all169 completed
items, and resumes with6150s remaining timeout and persistent console log.
Live job details in STATE; do not treat stale `running` JSON alone as a live job.

`upret-asset-audit-001` verifies partial local checkpoint step767 and incomplete
run1060/2600,363 tensors. Native startup needs explicit envelope unwrapping and
strict model-state parity. See BASELINE_AUDIT for scope; no UPRet metric or
method effect is claimed from this CPU-only asset check.

Verification: `/home/haipd/miniconda3/bin/python -m pytest -q tests/test_evaluation_contract.py methods/elsc/tests/test_bridge_parity.py -k 'not zero_init_adapter'` completed9 passed,1 deselected. Inspected compatibility modules alias the actual shared implementation. Covered asymmetric score orientation, tie expansion, gallery/positive contracts, multi-positive grouping, block parity and mask/bridge geometry. Synthetic fixtures support mechanics only; real score evidence is the separate replay above. `git diff --check` passed; pre-existing UPRet edits preserved.

## Continuation control preparation (not a method result)

`seds-continuation-data-check-002`:10.751s CPU,35 native TRAIN examples including
3 clips longer than300frames, plus all519 DEV examples. Frame selection matches
adapted extraction metadata exactly; every collated tensor finite; restoring
Python/NumPy/torch RNG reproduces all TRAIN batch tensors bit-for-bit. Native
random_swap changes20/35 captions on this seed/state. This validates data wiring,
not GPU optimization or retrieval gain. The new GPU wrapper still requires
step0 matrix parity and a real optimizer-update smoke before full continuation.

Failed check001 retained: verifier compared the source-label hash with the
reserialized/subset export hash. Fixed by checking source hash plus exact selected
label contents and recording the export hash separately. No data changed.
Five regression tests pass in SEDS environment; all9 campaign tests pass with
base-env pytest. An initial unittest-discover command could not import the older
pytest-based tests because native SEDS environment lacks pytest; the5 new tests
were then run with its built-in unittest, and all9 with existing base pytest.

## Resource stop, 2026-09-18

TRAIN extraction attempt2 exited1 at6623/7096 upon hitting15GiB free-space guard.
`train-extraction-stop-audit-001` verifies all committed feature hashes with zero
errors;473 remaining videos contain54,566headerframes. This is a storage stop,
not failed scientific evidence. No full TRAIN continuation has run. User asked
for additional storage; no assets deleted and no safety threshold lowered.

## Secondary dataset readiness and cached-score verification

`secondary-dev-assets-001`: official CSL DEV1077 video IDs exactly match local
manifest; raw1077, both CiCo streams1077 and temporal metadata1077 present.
Native SEDS pose/RGB DEV availability0 at checked paths, with0 train-ID overlap.
CSL has797caption groups: gallery must preserve1077videos×797texts and native
group relevance, not PH one-to-one. SEDS TRAIN6598groups contain18401videos.
H2S local DEV table1741rows has1739rawclips and1739 matching JSON sentence IDs;
JSON groups1527, so row count is not group count. SEDS TRAIN30786groups contain
31019videos. H2S native DEV pose/RGB absent at checked paths; no test loaded.

`csl-score-contract-001`: CPU recomputation from existing historical score
matrix verifies exact rank/metric parity, 1.076s. T2V R1/5/10=
68.883312/81.806775/86.951066; V2T65.923863/81.429898/86.908078;
meanR1=67.403588. Source score hash and ID order checked; selection metadata
agrees. This is cached-score verification, NOT fresh model/checkpoint replay,
feature parity, method gain, or independent confirmation. Historical DEV used
for epoch179 selection. CiCo CSL route is ready for subsequent model replay
once resource gates permit; preserve H2S-transfer feature regime disclosure.
## 2026-09-18 storage recovery and TRAIN completion

`seds-adapted-train-001` attempt3 completed7096/7096,exit0;6623 verified/reused,
473 newly extracted. Wall425.054s,total5947.341s within original6300s budget.
Source/recipe unchanged, no TEST loaded. This completes adapted TRAIN inputs,
not retrieval improvement. Provenance: its `run.json`,per-video metadata,
`jobs/seds-adapted-train-resume-003/launch.json` and console log.

## Matched continuation controls (PH adapted DEV, not TEST)

Native control001 completed222updates/7096examples in663.731s;peakCUDA21.080GB.
Step0 remains eligible and selected. All rows use the same519-video/text gallery,
native asymmetric tie policy,seed42,B32 and unchanged extraction/loss recipe.

| Optimizer | Step | T2V R1 | V2T R1 | Mean R1 | Delta vs initialization |
|---|---:|---:|---:|---:|---:|
| Initialization | 0 | 76.493256 | 78.612717 | 77.552987 | 0 |
| Native moments | 111 | 68.786127 | 68.786127 | 68.786127 | -8.766859 |
| Native moments | 222 | 71.098266 | 72.639692 | 71.868979 | -5.684008 |
| FP32 moments | 111 | 76.493256 | 78.805395 | 77.649326 | +0.096339 |
| FP32 moments | 222 | 76.685934 | 77.649326 | 77.167630 | -0.385356 |

Native step222 R5/R10: T2V92.292871/94.990366,V2T92.292871/94.797688.
Full branch metrics/ranks/scores are retained in each run's `eval_step*` files;
no rounding is used by selectors. `continuation_results.csv` records all18
step/stream rows with full-precision R1/5/10. FP32 control completed659.686s,
peak21.916GB,222updates,zero nonzero-m/zero-v cases in293FP16-parameter states.
Selectedstep111 improves only+0.096339pp vsinitialization, below+0.5pp pilot gate.
Its T2V R5 drops0.963391pp; do not hide that trade-off. Finalstep222 fails
the0.5pp directional guardrail (V2T R1 -0.963391pp), despite improved R5/R10.

Matched endpoint deltas vsnative are+8.863198pp at111,+5.298651pp at222. This
supports a numerical-correction effect in this seed/regime, not a new method or
generalizable benefit. No retuning/scale-up from this result. `seds-pair-audit-001`
passes4.015s CPU: all222batch IDs, shared config/assets/source, firsttwo losses
and gradients, allthree initialscore matrices (delta0), RNG at111/222 and model
shapes/dtypes match exactly. Checkpoint tensors differ as expected. This does
not prove every intermediate tensor was identical; correction changes updates.
No multiseed, independent confirmation or SOTA claim. Both jobs terminal;
process absence and idle GPU verified. Tests15passed; shared evaluator/bridge
9passed,1deselected; native precision3passed. No TEST loaded, no assets deleted.

## Same-I3D tail fine-tuning pilot — negative selection result

`rgb-finetune-pilot-001`: completed222updates/7096examples,4297.401s,
seed42/B32. Same head/control exposure; additional existing-I3D tail updates.
Head peak21.933GB and separate worker peak1.087GB (not simultaneous total).

| Step | Fused T2V R1 | Fused V2T R1 | Mean R1 | Delta vs matched frozen endpoint |
|---|---:|---:|---:|---:|
| 0 | 76.493256 | 78.612717 | 77.552987 | 0 |
| 111 | 76.878613 | 77.649326 | 77.263969 | -0.385356 |
| 222 | 77.263969 | 77.842004 | 77.552987 | +0.385356 |

Selector retains initialization: final mean ties it, and both trained endpoints
fail V2T guardrail against initialization (-0.963391/-0.770713pp). Selected
delta vs strongest FP32control111 is -0.096339pp; no +0.5pp lead. Final T2V
R5/R10=92.485549/96.531792,V2T R5/R10=93.448940/95.953757.
T2V R5 drops0.385356pp vsinitialization. All9 branch/step rows with fullprecision
R1/5/10 in rgb_pilot_results.csv; complete MedR/MeanR/ranks in run artifacts.

`rgb-pilot-audit-001` passed8.553s CPU:222matching exposure rows, cachedfeatures
and gradient versions/finite tensors, replay equality, initialscores, RNG atboth
checkpoints, pairedhashes and stricttail reconstruction;1038 refreshedDEVfeature
files verified. Recomputed18 score/metric artifacts across pilot and control,
including exact stored ranks and native kernel parity. Not a new checkpoint
GPUforward, independent confirmation, or benchmark TEST result.

Decision: close this fixed baseline configuration; no learning-rate/horizon
rescue or scale-up. This bounded negative result does not exclude other RGB
training mechanisms. Method contribution and alternative contribution remain
unvalidated. All artifacts preserved; no TEST opened.

## Cross-model stored optimizer exposure (CPU, not recall causality)

optimizer-exposure-001 completed7.540s; identities verified beforeloading.
Fraction is nonzero-first/zero-second elements divided by ALL stored FP16
moment elements, not fraction of trainable parameters or a measure of accuracy.

| Local checkpoint | Stored step | Optimizer | FP16 exposure |
|---|---:|---|---:|
| CiCo PH selected | 13 | local AdamW | 92.504713% |
| CiCo CSL selected | 2160 | local AdamW | 85.303473% |
| SEDS native continuation | 222 | BertAdam | 97.309112% |
| SEDS corrected continuation | 222 | FP32 moments | NA; no FP16 moments |
| UPRet partial baseline | 767 | BertAdam, FP32 model wrapper | NA; no FP16 moments |

AllFP32 moment groups havezero such entries; allinspectedmomentsfinite.
UPRet is incomplete and explicitlycastfloat32, not a nativeprecision control.
CiCo states have195FP16 moment tensors/124263424elements each; nonzero-first
denominator fractions99.998717% PH and99.832660% CSL. This establishes stored
phenotype acrossdatasets, not itscause or efficacy of changingprecision. The
>1% exposure screen admits real-gradient arithmetic validation and study
planning only. Do not infer anything about author training/published rankings.

## CiCo fixed-real-gradient arithmetic (no model updates)

Preregistered in CICO_MOMENT_ARITHMETIC_PROTOCOL.md. One actual TRAINgradient
per dataset/batch condition, reused verbatim across optimizer arithmetic arms;
freshzero moments/step1 with eachselectedcheckpoint's storedgroupLR. Not resume
and not a learningcurve. Nativeparameterdtypes,loss/augmentation/amp_bf16 kept.
All originalmodelweights unchanged; noDEV/TEST. PH001importfailure retained.

| Dataset / batch | Loss | Native v=0 / reference v>0, % FP16elements | Moment-only weight-delta relativeL2 | Activation |
|---|---:|---:|---:|---|
| PH32 | 0.000004289564 | 8.553020 | 0.944640 | pass |
| CSL32 | 0.000000741329 | 0.099354 | 0.952768 | FAIL: exposure<1% |
| PH512 | 0.036409900 | 98.029660 | 0.994792 | pass |
| CSL512 | 0.000914031 | 97.366622 | 0.998616 | pass |

B512 registeredafterB32 tocheck originalbatch fidelity; IDs are exactsame
sampler prefixes, not a batch/seed/layer search. B32failure is not overturned.
Underloss/scoringchanges fromdifferentnegativepools, exposure differs strongly;
do not generalize smallbatchdiagnostics toactualB512 training. FP32parameter
controls exactlymatchacrossmomentarms. Nativeforeach versus singlekernel gives
zeroFP16weightdifferences; FP32relativeweightdifferences2.36e-7/7.52e-7 atB512.
Normalizedupdate relativeerror vsFP32reference is10.093954 PH/15.620762 CSL
(ratios, not percentages). Holdingnativefirstmomentfixed andswappingonlysecond
moment denominator isolates partofthe arithmetic effect; allper-tensormeasures
are retained. Large relativechange is NOTrecallgain or a sufficient training
effect, especiallywhenabsoluteupdates aretiny. FP32moments alone also leave
parameter-roundingerror (B32PH relativeweighterror .993780 vsallFP32reference).

All5attempts total27.428048s, peakB51242.541GB. Fourfixedgradientarchives
~334.2MiBeach retained; no inference/checkpointselection. This admits paired
training feasibility only, not noveloptimizer, SOTA, or independentconfirmation.

## Matched CiCo PH continuation — numerical effect, no selected accuracy gain

Preregistered CICO_NUMERICAL_CONTINUATION_PROTOCOL.md; selectedlocalinitial
checkpoint,seed42,B512,260updates/133120examples perarm. Original200epoch
schedule retained; onlymomentprecision differs. Noaugmentation/loss change.
Bothfreshoptimizers; not resuming historicalmoments. Runtime native263.986s,
corrected261.756s; correctedpeak43.774GB. Training speeddifference not an
efficiencyclaim (singleuncontrolledtiming). Allrunscompleted, noTEST.

| Arm / step | T2V R1 | V2T R1 | Mean R1 | Delta vs initialization |
|---|---:|---:|---:|---:|
| Shared initialization | 74.181118 | 76.300578 | 75.240848 | 0 |
| Native moments /260 | 65.510597 | 65.510597 | 65.510597 | -9.730250 |
| FP32 moments /260 | 73.603083 | 75.337187 | 74.470135 | -0.770713 |

Fixedendpoint treatment effect+8.959538pp mean (+8.092486T2V/+9.826590V2T).
Bothselectorsretaininitialization: selecteddelta0, +0.5ppaccuracygateFAIL.
FP32finalR5/R10 T2V91.329480/95.568401,V2T91.136802/94.990366;
native85.549133/90.944123,85.549133/91.329480. All21evaluationsperarm retained
inrun.json andeval_step*score/rankartifacts, not cherry-pickedfinalonly.

cico-numeric-audit-001 COMPLETE6.590sCPU:all260pairedinputs/IDs/LR/RNG,
firstgradienthash/loss/normexact,15230featurehashes verified,42scorematrices
reranked exactly,checkpointRNG/sampler/scheduler/modelshapesdtypes match.
NativeFP16momentexposure98.866390%; correctedFP32zero,allfinite. Auditdoesnot
independentlyexecutethetrainedcheckpoint or establish statisticalconfirmation.

Registeredmechanism-replicationgate passes>=1ppendpoint gainbothdirections.
This admitsfixedrecipe multiseed/crossdatasetreplication only. StandardFP32
moment arithmetic is not novel; mitigation oftrainingdegradation is notSOTA.
No validatedalternativecontribution yet; independentconfirmation and broader
mechanism/claim checks required beforethatclaim.

## Fixed-recipe replication activation (results pending)

CSL two-updatepairedB512smoke completednative8.721301s/FP328.083615s,
43.776GB peak. Nativegroupedsampler12steps/epoch,2400stepschedule; identical
firstgradient/loss andbothstepbatch/RNG pairing verified. NoDEV/TEST in smoke.
Registeredreplicationqueue nowrunningPH1337/2026 andCSL42/1337/2026. No new
multiseed efficacy conclusion untilallregisteredpairs andCPUaudits complete.
OldPHseed42pair remainsdiscovery; newseedsdo notmakeexposedDEVindependent.

## Completed numerical replication (three training seeds per dataset)

All five newpairs andCPUaudits completed; queue2757.250s includingaudits.
Summary cico-replication-summary-001; 504epoch/direction metricrows in its
all_epoch_metrics.csv. Variability belowis sampleSD overtrainingseeds, notSE/CI.

| Dataset | Seed | Native final meanR1 | FP32 final meanR1 | Delta pp | Selected delta pp |
|---|---:|---:|---:|---:|---:|
| PH | 42 (discovery) | 65.510597 | 74.470135 | +8.959538 | 0 |
| PH | 1337 | 66.377649 | 73.988439 | +7.610790 | 0 |
| PH | 2026 | 67.726397 | 74.181118 | +6.454721 | 0 |
| CSL | 42 | 64.008311 | 67.308232 | +3.299921 | -0.109161 |
| CSL | 1337 | 62.891600 | 67.496438 | +4.604838 | +0.109161 |
| CSL | 2026 | 62.613049 | 67.466323 | +4.853274 | +0.062735 |

PH endpointdelta7.675016±1.253643pp; meanT2V/V2T directional deltas
6.743738/8.606294pp. CSL4.252678±.834409pp, directions3.429527/5.075828pp.
All6seedmeancontrastspositive. Bothregisteredmechanismreplicationgatespass;
NOseedpassesaccuracygate. PHallselectorsinitialization75.240848. CSLnative42
selects36,FP32seed1337selects228/seed2026selects216, remaininginit; thesevery
smallselectedgains do notsupportimprovementoverstrongbaseline. InitialCSL
67.4035875014 (fullprecision groupedmetric; priorroundedfiguresnotused).

Source/exposure/gradient/RNG/feature/checkpoint audits passedforeachpair.
SamehistoricallyexposedDEV andfixedselectedencoderinitialization remainlimits.
No statisticalsignificance, unseenqueryconfirmation or newoptimizerclaim.
11/11 statisticalfallacyitems explicitlyaddressed in summaryreport. AllR5/R10
trade-offs areavailable, no pooledcrossdatasetleaderboard. Freshcheckpoint
forward replay was separate; its completion and later TEST opening follow.

## Fresh-process checkpoint replay

`cico-checkpoint-replay-001` completed in 70.947796s: all twelve fixed endpoints
(two datasets, three seeds, two moment-precision arms) reconstructed strictly.
Full DEV score matrices are bit-exact; ranks, metric values and ID hashes match.
All source, feature and checkpoint locks pass. This is execution/serialization
verification, not a new training replicate or independent query confirmation.

## Locked PH TEST confirmation — mitigation, not accuracy improvement

`CICO_LOCKED_PH_TEST_PROTOCOL.md` locked seven models before TEST scoring.
`cico-ph-locked-test-001` completed all seven in 26.587924s (zero updates).
`cico-ph-test-analysis-001` rehashed the lock and inputs and independently
recomputed every stored score-matrix metric and per-query rank/tie record.
Full R1/5/10/MedianR/MeanR are in its `model_metrics.csv` (14 direction rows).

| Seed | Native T2V / V2T R1 | FP32 T2V / V2T R1 | Mean paired delta pp | FP32 minus initial mean pp |
|---|---|---|---:|---:|
| 42 | 60.280374 / 61.214953 | 69.470405 / 71.962617 | +9.968847 | 0 |
| 1337 | 63.084112 / 66.666667 | 70.093458 / 71.028037 | +5.685358 | -0.155763 |
| 2026 | 64.174455 / 65.420561 | 69.937695 / 71.028037 | +5.685358 | -0.233645 |

Shared initialization: T2V 69.470405, V2T 71.962617, mean 70.716511.
Three-seed mean paired endpoint effect **+7.113188 ± 2.473074 pp** (sample SD),
direction means +7.320872 / +6.905504. Registered paired bootstrap: 331 inferred
video-prefix clusters, 10000 draws, seed 20260918, conditional 95% interval
**[5.120159, 9.202454] pp**. Mechanism-confirmation gate PASS, all three positive.

This interval conditions on trained models and the fixed gallery; it is NOT an
interval over the population of training seeds, signers or datasets. Filename
prefixes are imperfect broadcast proxies, not verified independent sources.
The held-out status is campaign-specific: historical checkpoint pretraining and
TEST-selection provenance are incomplete, not proven clean or contaminated.
No CSL TEST result exists. Seeds vary continuation order/augmentation, not the
pretrained encoder initialization or complete from-scratch training pipeline.

All six PH DEV selectors kept the shared initialization: selected-model accuracy
delta **0**, so no improved deployable selected model/SOTA claim. Endpoint seed42
matches initial R1 but lowers T2V R5/R10 (86.760125/92.211838 vs
86.915888/93.146417) and V2T R5 (87.071651 vs 88.161994); the other corrected
endpoints also show higher-recall trade-offs relative to initialization. Report
the full table, not only recovery over the degraded native endpoint. No TEST
retuning, best-seed selection, or new horizon allowed for this claim.

## Locked CSL TEST: held-out videos, not novel-caption confirmation

Input preparation completed5250.261803s, both1176-video streams, batch128,
all provenance/finiteness checks; fixed DEV translator798 caption groups.
Zero TRAIN/DEV video-ID overlap;795/798 TEST caption IDs overlap DEV. This
limits interpretation even though no CSL TEST scores were used for selection.

`cico-csl-test-lock-001` locked allten distinct initialization/endpoint/selected
models before inference, SHA
88151788467a2130a2b0c2f2b79650dc9e8986d226ec442bcda7704ff25b74c6.
`cico-csl-locked-test-001` completed45.886551s, peak1.12032GB CUDA, zero updates.
GPU had another user's706MiB allocation with0% observed utilization; headroom
admission was documented before launch. Timing is not an efficiency comparison.

CPU `cico-csl-test-analysis-001` completed6.438022s, reverified all locked sources,
assets/checkpoints, recomputed all score metrics/per-query ranks and ID/mapping
hashes. Its `model_metrics.csv` contains all20 direction rows of R1/R5/R10/
MedianR/MeanR; SHA50cd99241c6395bc35ba53e50a568c463f6be350c97448a0ed9cce47274fab34.
Unchanged grouped best-positive metric on1176 videos/798 caption queries.

| Seed | Native T2V / V2T R1 | FP32 T2V / V2T R1 | Endpoint mean delta pp | DEV-selected mean delta pp |
|---|---|---|---:|---:|
| 42 | 65.914787 / 64.285714 | 67.794486 / 66.156463 | +1.875224 | +0.440834 |
| 1337 | 63.283208 / 62.414966 | 67.293233 / 66.241497 | +3.918278 | -0.380415 |
| 2026 | 64.035088 / 60.289116 | 67.669173 / 66.411565 | +4.878267 | -0.273004 |

Initial T2V67.669173, V2T66.496599, mean67.082886. FP32 endpoint mean differences
versus initial are−.107411/−.315521/−.042517pp. Endpoint paired effect
**+3.557256 ±1.533727pp** (three-seed sampleSD), direction means+3.174603/
3.939909pp. Registered paired caption-cluster bootstrap10000 draws, seed20260919,
798clusters gives conditional95% CI **[2.299635,4.810784]pp**. All three positive;
mechanism confirmation gatePASS. It is conditional on fixed models/gallery, not
a seed-population or signer-independent confidence statement.

Selected contrast **−.070862 ±.446384pp** (descriptive). Native42 selected36 is
compared to FP32initial; native1337/2026 initial to FP32selected228/216. FP32
selected models are initial/−.380415/−.273004pp versus initial: none improves
meanR1. No TEST model selection or new training follows. This does not rescue
the failed DEV accuracy gate or support SOTA.

Trade-offs versus initial remain: corrected endpoints T2V R5 fall
.501253/.375940/.626566pp. V2T R1 falls.340136/.255102/.085034pp. Some V2T R5/R10
rise, so do not summarize all metrics as uniformly better or worse. Selected
FP32seed1337 V2T R1 falls.510204pp, slightly beyond the earlier.5pp directional
guardrail; full selectedT/V metrics remain in the CSV, not hidden by the mean.

Across PH and CSL, the replicated finding is mitigation of local continuation
degradation relative to native-moment endpoints. Neither dataset yields a
validated improved selected retriever. Do not pool the two protocols into a SOTA
number. CSL was registered after PH TEST and shares DEV captions; both retain
historical pretraining/selection provenance limitations.
