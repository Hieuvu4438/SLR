# Run budget — initial tranche, 2026-09-17

Local RTX 5880 Ada only. Initial ceiling 12 GPU-hours, not an unlimited allocation: baseline/parity 2 h; discovery and matched controls 4 h; confirmation/ablation reserve 6 h. Actual throughput from smoke determines whether full training fits. At most three admissible mechanisms, two pilot configurations per mechanism including needed controls; no tuning rescue after two failures.

Before every GPU launch: check device idle and available disk; preserve at least 15 GiB. Generated files under `artifacts/slret_goal/`, total incremental storage cap initially15GiB, revised24GiB on2026-09-18 for the explicit RGB pilot below. No raw asset deletion. Keep checkpoint/optimizer/RNG for any resumable training; retain only preregistered checkpoints, scores and configs. No full source-tree copy containing model assets.

Inventory timeout 10 min CPU; first SEDS smoke (4 examples, backward on 2) hard timeout 10 min, full train diagnostic (first 256 canonical release-train rows) at most 20 min. These are baseline diagnostics, not method pilots. Predeclared FP32 score parity tolerance max absolute 1e-4, exact metric/rank agreement except explicitly identified ties; capture both native and shared kernels.

Pilot gate: +0.5 pp mean bidirectional R@1 over strongest matched control, no direction decline >0.5 pp. Confirmation requires three training seeds when feasible, majority gains, positive independent paired CI, and R@5/10/group trade-offs reported. Aspirational strong gain +2 pp against comparable SEDS; SOTA requires current comparable literature and final test lock.

Monitoring: authoritative PID/process handle + timeout; progress every ~30 s. Per skill, timeout is enforced; user's explicit implementation/experiment authorization overrides the skill's generic ask-before-command and no-script-generation defaults.

Preprocessing tranche: two public RTMPose checkpoints268,942,231 bytes total;
two24-frame inference diagnostics consume less than1 GPU-second measured,
startup/download wall time is recorded separately (not all GPU time). Existing
RGB cache comparison is CPU-only. Adapted extraction smoke first2 TRAIN videos
has hard timeout5min; canonical DEV519 maximum30min after successful smoke,
counted against the2h baseline budget. No additional method budget allocated.

Completed adaptedDEV extraction390.556s and eval17.862s; TRAIN extraction smoke
5.920s. Sum414.338s wall occupancy (~0.1151h), not a claim of constant GPU
utilization. Add earlier short parity/smoke runs when projecting remaining
baseline reserve. Full adaptedTRAIN at comparable throughput ~89min, so assess
frame census and resumability before launch. Storage added by dev219MB, model
downloads269MB, evaluation score/rank files additional small artifacts; well
below15GiB incremental cap. No training pilot or confirmation budget spent.

Full TRAIN admission (2026-09-17): OpenCV header census827,354frames/7096videos,
23 above300frames,max475,no missing. Frame-scaled estimate5793s (~96.6min),
superseding the earlier count-only89min estimate. Hard timeout6300s (105min),
plus prior baseline work remains below7200s. Resume smoke completed with exact
numeric/file hash parity on2 TRAIN videos;4 CPU safety tests pass. Around3GiB
expected output. If timeout occurs, inspect remaining baseline budget before
resuming; do not silently allocate another105min. Paused or orphaned item files
are retained, not deleted. No source/recipe changes while extraction is live.

Continuation control preregistration: two-step smoke<=300s, full222 updates<=1200s,
charged to4h discovery/matched-control reserve. Neither is launched yet. B32,
seed42, native optimizer/loss, no sweep; at most two full optimizer-state
checkpoints. Step0/111/222 DEV evaluation and initialization-eligible selection.
Real CPU data checks cost0 GPU time and do not spend method-pilot configurations.

2026-09-18: TRAIN attempt2 failed on disk reserve after5386.760s, total including
attempt1=5522.287s. Remaining original6300s admission=777.713s.473 videos remain,
54,566frames; predicted383.5s plus resume verification. Do not restart until
storage restores15GiB reserve plus projected outputs. No additional GPU jobs
launched. Complete campaign artifacts3.4GiB; filesystem pressure is not caused
by exceeding this campaign's15GiB artifact cap. No deletion performed.

2026-09-18 storage restored by user,168G available after completion. TRAIN
attempt3 completed remaining473 in425.054s,all7096 now complete,total5947.341s
within6300s. Campaign3.6GiB. Continuation smoke001 launched with300s bound;
Smoke completed30.162s,18.769GB peakCUDA,2updates. Full native continuation
control001 launched1200s bound,222updates,not a method configuration.

UPRet PH DEV replay admission2026-09-18: one full519-gallery inference from
verified historical partial corrected checkpoint step767;<=300s baseline
budget after SEDS control releasesGPU. No optimizer/update or test. Require
exact historical rank agreement and independent stable-tie rank calculation;
also expose sharedCiCo tie-policy diagnostics without changing primarymetric.
Verify allDEV featurefile hashes and recordedmanifest/source identities.
Failure means investigate bridge/provenance,not a method pilot or tuning.

Native SEDS control001 completed663.731s,222updates,peak21.080GB;smoke30.162s.
Discovery/control wall694s (~0.193h). No method configuration used. Campaign
10.456GiB after both trained checkpoints;remainingcap4.544GiB. UPRet replay001
launched300s bound. Numerical moment audit1.910s CPU only.

UPRet replay completed5.710s. FP32-moment smoke completed28.550s;fullcontrol001
launched1200s bound after passing activation/parity. Disk161G,campaign10.522GiB,
conservative additional4.030GiB=>14.552GiB projected under15GiBcap. Fullcontrol
retention differs only in checkpoint payload (step111 model-only,step222full),
not training/evaluation. No method configuration consumed.

2026-09-18 final correction control659.686s,paired audit4.015s CPU. Discovery/
matched-control total30.162+663.731+28.550+659.686=1382.129s (~0.384h), no method
configuration used. Campaign14.480GiB of15GiBcap;filesystem157GiBfree. Onlyabout
0.520GiB incremental headroom remains undercampaigncap: disk cleanup resolved
filesystem pressure, but does not silently revise this cap. No new checkpoint-
producing run before an explicit revised storage/retention plan;small CPU or
inference artifacts may still fit. Preserve all present checkpoints. No livejob.

CSL fresh replay admission: `tools/csl_replay.py`, <=300s, no optimizer/checkpoint,
reserve128MiB output under15GiB cap. Verify selected checkpoint/config/manifest,
2154feature hashes and shared grouped1077x797 gallery, then require score maxabs
<=1e-4 and zero historical rank changes. No new selection or TEST. CPU/preflight
failures retained; do not mistake grouped metrics for PHsingleton policy. Charge
baseline reserve; available after TRAIN5947.341s and short earlier baselinejobs.

2026-09-18: CSL replay completed6.340244s baseline; scoring profile27.004249s
discovery. RGB-gradient-feasibility001 startup failure0.799167s is retained and
conservatively charged to discovery although no model/GPU work occurred.
Discovery/control cumulative1409.932s; retry002 capped300s, zero updates and
<=32MiB artifacts, same15GiB cap. Package installation is CPU/network setup,
not GPU occupancy. Current filesystem154GiBfree does not expand campaigncap.

2026-09-18 RGB diagnostics charged: feasibility00210.169149s, native-runtime
parity1.783803s, base-runtimeparity1.724426s, cachedVJP2.977175s, strongerVJP
equivalence1.423046s. Discovery/control cumulative1428.009969s (~0.397h), all
jobs terminal. Campaign14.592503GiB, filesystem153GiBfree; no deletion.
Original-runtime fixed8-window replay passes. Two-video conservative fullTRAIN
projection10278.177s excludes IPC/newoptimizer/DEV extraction. Measure fullB32
two-update transactional bridge before considering full admission; revise
checkpoint retention explicitly before exceeding current15GiB cap. No longrun
or extra method configuration admitted by these numerical checks.

RGB two-update smoke admission2026-09-18: <=600s discovery/control, <=128MiB
outputs (two versions of B32feature/gradients and one tail+optimizer checkpoint).
Campaign14.5925GiB+0.125GiB<15GiB,filesystem153GiBfree. No caprevision/deletion.
No headcheckpoint, no resume, noDEV/TEST. Matched firsttwo fullcontrolbatches;
fixed tailLR1e-5 nativeBertAdam schedule222/warmup.1, not a parameter sweep.

Smoke001 completed49.903408s,86.558MiB; cumulativediscovery1477.913377s
(0.4105h), campaign14.677045GiB. No livejob. Worst observedstep22.991151s
x222=5104.036s; with30% margin6635.247s before DEVrefresh/checkpoint overhead.
This replaces the earlier two-video estimate with fullbatch measurements but
still requires explicit fullpilot registration and checkpoint storage revision.

FullRGB B_tuned pilot preregistered `RGB_FINETUNE_PILOT_PROTOCOL.md`:9000s hard
bound,222updates, same64-smoke recipe/no tuning. Discovery used1477.913s; even
fullbound yields10477.913<14400s;6hconfirmation untouched. Storage cap revised
24GiB with23.3GiB conservativeprojection and153GiBfilesystemfree,15GiBreserve.
Preserve all oldruns; retention/accounting details inprotocol. No automatic
resume orbound extension. This registers one baselineaccuracyconfiguration,
not novelcandidate or successfulpilot. Launch only after implementation checks.

Terminal RGBpilot4297.400694s, below9000s hardbound. Discovery/control cumulative
5775.314071s/14400s;8624.685929s remain,6hconfirmation reserve untouched.
CPU integrity audit8.552580s recorded separately, no GPU. Campaign22.435108GiB
of24GiB;filesystem145GiBfree. All prior artifacts preserved. Pilot failed lead,
so no confirmation allocation or further tuning of this configuration. Additional
checkpoint-producing work requires storage admission; remaining1.565GiB cap
is not sufficient for another fullSEDS head+optimizer pair without revision.

optimizer-exposure-001 completed7.540084s CPUonly,5existingcheckpoints read,
no GPU/updates/features/checkpoints added. Registered300s/1MiB maximum in
OPTIMIZER_EXPOSURE_PROTOCOL.md. DiscoveryGPU accounting unchanged; no causal
training budget spent or admitted by thisscreen.

CiCo real-gradient arithmetic:PH001startupfailure0.702426,PH0025.992154,
CSL0016.161086,PHB5120017.160713,CSLB5120017.411669seconds;total27.428048.
Discovery/control5802.742118s/14400s, remaining8597.257882s. Confirmation6h
untouched. Four~334.2MiB gradientarchives, campaign23.741953GiB/24GiB.
All jobs terminal. No remainingstorage admission for fullcheckpointpair;
explicit revisedcap/retention needed before any pairedtraining smoke/run.

2026-09-18 CiCo pairedB512smoke native8.179961s/fp328.209355s complete,
two updates each, firstloss/gradienthash/norm exact andbothbatch/RNGpaired.
Peak43.776GB, worststep1.716316s;260stepsx1.716316x1.5=669.363s plus21DEV
evals(~5.4s baselineeach,2xmargin227s),featurehash/checkpointoverhead<300s:
projected<1200s/arm, admitted hardbound1800s/arm. No trainingparameter change.
Discovery5819.131434s so far; pairbound3600s staysbelow14400s.

Storagecap NOWexplicitlyrevised24→30GiB beforepilot launch per
CICO_NUMERICAL_CONTINUATION_PROTOCOL.md. Existing23.742GiB + pairedfinalfull
states~2.7GiB + selectedmodels~.7GiB + scores/ranks/logs.25GiB +overhead.5GiB
projects<28GiB. Filesystem144GiBfree,15GiBreserve. Preserveallpriorartifacts;
onlynewrun rollingselected.pt may be replaced bybettereligiblecheckpoint.

CiCoPHpair completed native263.985601s +FP32261.756318s=525.741919s;
discovery6344.873353s/14400s, remaining8055.126647s. CPUaudit6.589856s separate.
Campaign26.229328GiB/30GiB, noartifactdeletions, alljobs terminal. Accuracygate
fails; empiricalmechanismreplicationgate passes. No multiseed/crossdatasetlong
run yetadmitted; register sharedrecipe/retention/compute first. Confirmation
reserve6h untouched, andisnot permission toclaim independentconfirmation on
historicallyexposedDEV. New storageplan needed for morethanone retainedpair.

Replicationadmission2026-09-18: CICO_NUMERICAL_REPLICATION_PROTOCOL.md locks
PH1337/2026 andCSL42/1337/2026 pairednative/FP32, originalB512/fixed20epochs,
plusCSLtwo-update smoke. Sequentialdriver hardbound7000s, eachfullrun<=900s,
smoke<=300s, CPUaudit<=180s; beforeeachstage reservefullbound+15s within7000s.
Actualwallincludesaudits conservatively;8055.127s discoveryremaining admits
this7000sphase withouttouching6hconfirmation. No failure autoretry.

StoragecapNOW44GiB (was30), existing26.229GiB + fivepairs<=16.65GiB +1GiB
margin<44GiB. Filesystem142GiBfree/15GiBreserve. Allpriorartifacts preserved;
newrollingselectedmodelonly may be atomicallyreplaced underregisteredselector.
CSLfullruns requirepairedsmoke<=60s andpeak<47GiB. Do not mutatequeuedsource
filesafterlaunch; driver hash-checks them beforeeverychild. 39CPUtests passed,
v1runner preserved bitexact tooriginalPHpairscript hash.

Replicationqueuecompleted2757.250260s includingCPUaudits; conservativelycharge
allwall todiscovery:9102.123613s/14400s,5297.876387s remaining. Confirmation6h
untouched. Campaign39.940078GiB/44GiB, filesystem125GiBfree. No olddeletes.
Preregisteredall12checkpointDEVreplay launched300sbound,<=100MiB; failure
mustbeinvestigated, no toleranceweakening. NoTESTGPUwork yet.

Completed checkpoint replay: 70.9477956295s, all 12 models bit-exact. Discovery
total now 9173.0714082252s/14400s, remaining 5226.9285917748s. CPU lock
cico-ph-test-lock-001 took 4.327497959s, reported separately.

Locked PH TEST cico-ph-locked-test-001 completed 26.5879242420s/300s bound,
peak CUDA 1120320000 bytes. Charge confirmation reserve: 26.5879242420s of
21600s; no training updates. CPU analysis cico-ph-test-analysis-001 is separate.
No remaining budget is authorization to retune against the now-opened PH TEST.
All old artifacts retained; 44GiB campaign cap and 15GiB free reserve unchanged.

CSL TEST input probe001 completed11.9552571774s, both DEV streams bit-exact and
64 translations exact. Confirmation cumulative38.5431814194s. Metadata census
only on TEST:1176 raw videos,798 caption groups,185718 dense windows. 795groups
overlap DEV, zero TRAIN groups; no TEST retrieval yet.

Full input-preparation csl-test-assets-001 admitted <=9000s from confirmation,
<=2GiB extra under44GiB cap,15GiBfree reserve. Measured campaign39.983GiB,
filesystem125GiBfree. Protocol CSL_TEST_ASSET_PROTOCOL.md. Sequential unchanged
I3D streams, fixed DEV translator, no checkpoint or retrieval score. Historical
window-scaled projection~6210s, all-bound budget9038.543181<21600s. Failure or
batch fallback blocks scoring; no automatic retry/bound extension/deletion.

2026-09-19 CSL preparation COMPLETE5250.2618033886s/9000s: agnostic2769.824901s,
aware2460.121779s plus translation/preflight/audit; all1176perstream, batch128,
no failures. Confirmation actual cumulative5288.8049848080s/21600s; remaining
16311.1950151920s. Peak CUDA11624168960bytes. Campaign44461947151bytes
(41.408GiB including directory overhead), filesystem114GiBfree,cap44/reserve15.
CPU CSL lock completed5.734766s separately. Evaluation<=300s and128MiB fits;
not launched while unrelated user GPU processes741186/759881 are active.

2026-09-19 operational admission amendment, BEFORE CSL TEST scoring: replace
the agent-imposed empty-process-list condition for this one <=300s inference
with measured headroom: >=8GiB GPU free and low observed utilization, plus all
existing disk/time/source locks. Process741186 retains706MiB but two samples
10s apart show0% GPU/VRAM utilization and47805MiB free. This is not reserved
exclusive compute, nor a user prohibition on shared use. No process is killed,
changed or inspected beyond public resource metadata. Required inference peak
was~1.12GB on PH and CSL batch/blocks unchanged, well below free headroom.

The original scientific lock/protocol/checkpoints/selector remain byte-unchanged.
No training, hyperparameter change or extra model admitted. Existing300s hard
timeout and failure retention apply. Concurrent occupancy is disclosed; measured
walltime is accounting only, not a latency/efficiency comparison. Do not add a
throughput claim. If available memory fails the headroom check, do not launch.

CSL locked evaluation completed45.8865513802s/300s, alltenmodels; peak CUDA
1120320000bytes. Confirmation cumulative5334.6915361882s/21600s (~1.482h),
remaining16265.3084638118s. CPU analysis6.4380223751s separate. Actual retained
campaign44503943332bytes (~41.448GiB), no deletions. Both TEST datasets opened
under their own locks; remaining budget is NOT authorization for TEST retuning.
