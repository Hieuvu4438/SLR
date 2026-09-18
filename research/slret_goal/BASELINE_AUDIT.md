# Baseline audit — current partial evidence

MEASURED_EFFECT2026-09-18: matchedCiCoPH260update pair completed, FP32moments
recover+8.959538pp finalmeanR1 overnativemoments; bothselectinitialization,
so noaccuracygain. NativeFP16statephenotype98.866390% versusFP32zero.
ActualCPUaudit verifies260batch/RNG pairs,firstgradient,15230featurehashes,
42score/rankartifacts andfinalcheckpointstate. DetailsRESULTS.md. This is
same-model numerical baselinecorrection, not newmethod/independentconfirmation.

RUNTIME_VERIFIED2026-09-18: same-real-gradient CiCo moment arithmetic. PH/CSL
B512nativeFP16secondmoment collapses tozero whileFP32referencepositive in
98.0297%/97.3666% ofFP16elements; sourceweights unchanged. Nativeforeach and
singlekernel produceidenticalFP16weightoutputs; FP32control momentprecision
armsbitexact. This identifies arithmeticloss for theseactualgradient tensors,
not itsrecall impact or universaltrainingfailure. CSLB32exposuregatefailed and
isretained. Fullconditions/results in CICO_MOMENT_ARITHMETIC_PROTOCOL.md,
RESULTS.md and fiveattempts'run.json. Gradientsretained, noDEV/TEST loaded.

RUNTIME_VERIFIED2026-09-18: fullRGBpilot001 and CPUaudit001 complete.
222updates/7096examples, allpairedbatch/RNG/feature/checkpoint checks passed;
all18 pilot/control scorematrices recomputed to identical saved ranks/metrics.
No trainedcheckpoint passesdirectionalguardrail; selectorinitialization. No
learningrate/horizon rescue. Fullresults RESULTS.md/rgb_pilot_results.csv.

Cross-model checkpoint census optimizer-exposure-001 (7.540s CPU) verifies
actual CiCo PH/CSL selected AdamW states:195FP16momenttensors each,
92.5047%/85.3035% ofelements have m!=0 andv==0, allfinite. Actual localfactory
`shared/slr_common/upstream/factory.py:_load_cico_core_from_state` retains
mixed dtypes; `methods/elsc/elsc/train.py:_optimizer` uses PyTorchAdamW,
not nativeBertAdam. Configamp_bf16 does not meanoptimizerstateisBF16/FP32.
UPRet historicalwrapper `methods/sssc/method1/model_factory.py` explicitly
castsmodel.float(); all350momentstatesFP32,zero phenotype. Do not substitute
sourcezeros_like for evidence ofactualdtype/exposure. Storedmoments cannot
proveunderflowcause or recallharm; nextgate requires same-real-gradient
arithmeticreference before cross-model causaltraining. No newGPUjob.

### Historical smoke observations (fullpilot now complete)

RUNTIME_VERIFIED2026-09-18: two-runtime same-I3D gradientbridge smoke001
completes2fullB32 updates49.903s; headloss/norm exactcontrol, tail36tensors
update atsecondstep, frozenweights/BNunchanged, feature refreshverified.
`RGB_TRAIN_SMOKE_PROTOCOL.md` declares nativeheadFP32moments andtailBertAdam.
Not accuracy evidence. Fullpilot001 step0 allthreeDEV scorematrices bitexact;
thirdTRAINbatch consumesupdatedfeatures with exactreplay, establishing active
encoder adaptation. Fullpilotoutcomes pending, TESTlocked. No vendor edits.

2026-09-18 `seds-fusion-layout-cpu-001`: suspected K/V head–time reshape defect
FALSIFIED before GPU. Earlier native lines already transpose K/V; independent
CPU attention sampling and gradients match native. An extra permutation would
break correct routing. Installation API removed; counterexample fixture retained
only for regression tests. Three native-environment tests pass; no source patch,
runtime intervention or training admitted. Do not report this as a SEDS defect.

2026-09-18 runtime updates: SEDS continuation smoke001 passes exact release
tensor loading and step0 parity across all3 streams;warmup updates[0,664]
changed tensors,2steps,30.162s,peak18.769GB. Native control001 completed222updates,
selector retained initialization. FP32-moment control completed222updates;
selected111 mean77.649326, below+0.5pp gain gate vsinitialization. Final77.167630
vsnative71.868979: measured numerical effect in one seed,not a new method.
Paired audit verifies identical source/assets/config (exceptoutputdir),batchIDs,
initialscore matrices and checkpoint RNG/dtypes. SeeRESULTS forR5/10 trade-offs.
UPRet `upret-load-check-002` CPU loads363/363 historical student tensors exactly.
Legacy nestedGit guard failed in001 because this is a vendored checkout;new
wrapper checks historical source hashes,allowing only the existing disabled
debugger line in module_cross. Changed modeling_clip4clip is a different
entrypoint,not imported by modules.modeling. No upstream files modified.
Checkpoint remains historical partial step767. Fresh full DEV replay001 now
verifies all1038feature hashes and exact historical rank parity;T2V20.809249,
V2T18.882466,5.710s. These scores do not characterize fully trained UPRet.

New bounded numerical check preregistered2026-09-18: continuation checkpoint111
contains295 FP16 model tensors;optimizer has586 FP16 and806 FP32 moment tensors.
Native BertAdam initializes moments with zeros_like(parameter). Inspect existing
CPU checkpoint only: count nonzero first moments with exactly-zero second
moments,by dtype,without modifying training. Material exposure (>1% of FP16
moment entries) admits consideration of a matched numerical-baseline correction,
not a research method. If absent/minor,do not pursue this explanation. Even high
exposure does not establish recall causality. This is not augmentation-off
precision rescue(Q21),input-collapse thresholding(Q28),or inference tie/sorter
precision tuning(Q29);those remain closed. No precision method/sweep admitted.

Evidence levels: AUTHOR_CLAIM, SOURCE_VERIFIED, RUNTIME_VERIFIED, MEASURED_EFFECT, HYPOTHESIS, UNKNOWN. HEAD and all actual Python-file hashes: `artifacts/slret_goal/inventory/report.json`; prior dirty UPRet diff retained there. No claim of exhaustive source audit.

## SEDS PH active path

`dataloaders/dataloader_ph_retrieval_pose.py:ph_DataLoader_pose` → RGB release features and cropped hand/body keypoints → `modules/modeling.py:get_sign_output` (SignBERT GCN, windows, sign convolution) → `get_visual_output` (pose/RGB contextual CLIP) → `module_fusionencoder.py:Gloss_Fusion_Transformer` → `flip_similarity_softmax` → `forward` losses → `main_task_retrieval.py:prep_optimizer/train_epoch` → `eval_epoch` → `metrics.py`.

- SOURCE_VERIFIED: `scripts/train_ph.sh` uses 8 GPUs, batch128,200 epochs, lr1e−5/sign_lr1e−4; our single-GPU checkpoint diagnostic is not a reproduced training recipe.
- SOURCE_VERIFIED + RUNTIME_VERIFIED: `freeze_exfusion` exists in current parser, false; `rgb_pose_match` enabled coefficient.4; `rgb_pose_kl` false. No top-k KL critique applies to this recipe.
- RUNTIME_VERIFIED: all checkpoint tensors load exactly, no missing/unexpected/mismatched tensors, PH smoke and 256-row run. Original release checkpoint never modified.
- RUNTIME_VERIFIED: first16 TRAIN batch has total loss0.000285228, 696/696 parameters with present gradients nonzero; match loss is0 on that state. A zero term at a saturated released model does not establish branch inactivity.
- SOURCE_VERIFIED: default entrypoint selects with test loader. New runner constructs TRAIN explicitly; no new test evaluation.
- SOURCE_VERIFIED: PH `dev.pkl` contains train+dev. Canonical519 dev IDs have zero overlap with train but no SEDS release RGB/pose files in downloaded package. No full official dev reproduction yet.
- RUNTIME_VERIFIED: native scorer matrices are `[video,text]`; shared evaluator agrees within1e−5 on R1/5/10 and MeanR for all three streams on TRAIN256. Native log direction labels are correct: local variable/function names are misleading but the `format` calls swap correctly. Our initial `run.json` field labels `native_log_*ACTUAL_*` were inaccurate metadata; use the saved `*_metrics.json` and corrected runner field names. Actual numeric metrics and assertions are unaffected.
- UNKNOWN: exact cross-stream source-frame parity. Count assertion alone is insufficient. Local raw pose available via `docs/proposal1/datasets.md` differs from release: normalized coordinates, scores outside[0,1]; no unverified conversion accepted.
- MEASURED_EFFECT (feature level only): `pose-parity-l384-003` and `pose-parity-l256-001` fail predeclared release-coordinate/confidence parity on24 TRAIN frames. Strict model tensor loading passes. `rgb-cache-parity-001` fails RGB feature parity at504 index-aligned windows. No retrieval effect measured; do not silently merge local CiCo RGB/local pose with release assets.
- SOURCE_VERIFIED: shared `slr_common/features/i3d.py` explicitly declares square-Lanczos plus center-resampling as an adapted reproducible spatial recipe; it is not proven SEDS release preprocessing. Native SEDS frame filtering and selected window indices can be reused to ensure alignment in a newly generated, explicitly adapted pipeline.
- RUNTIME_VERIFIED: explicit adapted519 DEV pipeline completed; recorded matching original-frame sequences for RGB/pose and passed all native loader checks. Native/shared full-gallery metrics pass with exact released checkpoint tensors. Fused meanR1=77.552987, not release-preprocessing parity and not a novel improvement. Full adapted TRAIN7096 completed with the same locked contract.
- RUNTIME_VERIFIED: global NumPy module aliasing breaks SciPy sparse import in SEDS environment. Corrected wrapper uses PH-loader-scoped unpickler reconstruction-name mapping only; no global NumPy/torch/source edits. Failed eval001 preserved; eval002 completed.
- SOURCE_VERIFIED: eval attention-mask indexing uses `segment_ids[input_mask,...]`, but `get_sequence_output` derives text mask from CLIP tokens. No measured retrieval effect; no method proposal from it.
- SOURCE_VERIFIED: native `prep_optimizer` uses six BertAdam groups (CLIP/sign/other, each decay/no-decay), weight decay.001, warmup_cosine.1,b1.9,b2.98,eps1e-6; `train_epoch` also clips global norm1 and clamps CLIP logit_scale. Continuation wrapper preserves these operations with explicitly adapted B32/one-pass schedule, not release8GPU training. Inactive KL losses are floats, so wrapper finiteness validation must accept scalars. GPU optimizer activation is now verified by smoke and full native control.
- SOURCE_VERIFIED (2026-09-18): SEDS paper Eq4 auxiliary coefficient.8 differs from current native `modules/modeling.py:forward` line253, which adds pose/RGB text losses with coefficient1. Parser `alpha=.8` is not used by this forward. Native continuation keeps coefficient1; no B_corrected run or effect measured. This ordinary recipe discrepancy is not a method candidate.
- RUNTIME_VERIFIED (CPU): native TRAIN loader on35 adapted examples including3 long clips selects exactly the recorded extraction frames; all519 DEV samples collate; RNG replay of augmented TRAIN batch is bit-exact. `GetTotalFrameList` draws a random `start` for long clips but slices from0 in both native loaders; unused draw is preserved, not patched. No sampling mismatch or recall effect claimed from this unused variable.

## CiCo and UPRet

CiCo implementation is `third_party/SLRT/CiCo/CLCL`, not a separate SLRT baseline. Shared `slr_common` provides strict checkpoint bridge, canonical manifests, masks, tokenizer and asymmetric tie-aware evaluator. Preserve its feature provenance: historical local PH dev uses BSL5K agnostic plus How2Sign-transfer-aware re-extracted features, not automatically release PH features.

UPRet local modifications predate this campaign and are preserved. `test_ph.sh` accepts an optional checkpoint; absence of one would initialize rather than reproduce a trained retrieval release. Entry-point/checkpoint/feature completeness still needs runtime verification. Old transport reduction gradient failure remains closed as a candidate.

`upret-asset-audit-001` now verifies the historical local checkpoint hash
`c72ecbc4ce35bec1f5fe53692a7d3f88b27e591c4692a93e82df26fdde00ff82`:
`runs/method1/ph/base/seed42/best_dev.pt`,363 tensors under `student_state_dict`,
step767/epoch59, `training_run_complete=false`. Latest run step1060 against
declared2600, no training-complete marker. This is a partial corrected local
checkpoint, not a complete trained release. Native `main_task_retrieval.py`
imports `modules.modeling.CLIP4Clip`; its `init_model` passes the whole loaded
object directly as state_dict and does not unwrap this historical envelope.
An explicit bridge/strict tensor check is required before inference. No new GPU
UPRet forward performed in this initial asset audit; later replay001 is described
at the top of this file. Recorded historical source hashes differ
from current `modules/modeling_clip4clip.py` and `modules/module_cross.py`; these
are pre-existing user changes, preserved, not automatically a new defect.

## CSL/H2S DEV contracts, 2026-09-18

`secondary-dev-assets-001` verifies CSL raw/CiCo availability for all1077 DEV
videos, no SEDS DEV features at inspected paths. Existing shared
`evaluation/runtime.py:encode_gallery` deduplicates caption_id to797texts and
builds group-positive mappings; native SEDS CSL loader also groups captions
with multiple videos. Existing PH replay wrapper hardcodes519 and pair_id
singleton relevance: do not merely change its manifest path to evaluate CSL.
`csl-score-contract-001` verifies historical shared-evaluator metrics/ranks
exactly on1077×797 cache. Subsequent `cico-csl-dev-replay-001` fresh checkpoint
inference now verifies score delta0,zero rankchanges and2154feature hashes;
not native target-domain-release parity or independent confirmation.
H2S local DEV table has1741clipIDs/1739raw files, and JSON groups are1527 rather
than1741. Missing raw rows and complete availability lists are in artifacts.
