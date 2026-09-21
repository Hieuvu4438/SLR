# SLRet V2 — archived state through C08 repair (see STATE.md for current)

## Material Passport

academic-research-suite / experiment-agent, inline run; 2026-09-19.
Current user specification: `docs/guide/ASTRA6_SLRET_RESEARCH_GOAL copy.md`.
V1 is history. Goal ACTIVE: improve SEDS/CiCo; do not close on a negative pilot.

## Reference and incumbent

- Code snapshot `e3f53b3`, pushed after V1 snapshot `347f44c`.
- SEDS release `third_party/SEDS/ckpt/ph_best_model.bin`.
- TRAIN `artifacts/slret_goal/seds-adapted-train-001`, 7096 videos.
- DEV `artifacts/slret_goal/seds-adapted-dev-001`, 519-video full gallery.
- Fixed release reference mean bidirectional R@1 77.5529865125.
- Existing incumbent FP32 control step111 meanR1 77.6493256262;
  `artifacts/slret_goal/seds-moment-control-001/checkpoint_step111.pt`.
- New provisionalDEVincumbent C04step44477.9383429672, T77.071291/V78.805395;
  `artifacts/slret_goal_v2/seds-lora-001/best.pt`. One-seed exploratory only.
- Reuse native model/loader/evaluator plus V1 FP32 moment helper. No corpus
  rehash/re-extraction; both TEST sets stay outside hypothesis and selection.

## Current candidate

LATEST REPAIR (user requested): c08-masked-pair-001 failed during first backward,
0 updates, child20.801845s. DDP find_unused_parameters could not see decoder
loss stored outside forward output; separate autograd.grad diagnostic was also
removed. No C08 efficacy result exists from attempt001; keep its logs/source.
Repair returns auxiliary as eighth forward tensor, preserves seven native losses,
uses one backward and an auxiliary-feature-edge gradient hook. Tests now include
actual single-process CPU/Gloo DDP treatment/control, two updates and eval parity;
5 tests pass0.468s. No method/LR/data/horizon change. Registered rerun attempt002,
c08-masked-pair-002 with same child limits240/1080/1080,parent2460s.
Charge failed child once: used16890.120074s, remaining4601.879926s,
currentpool2527.263531s. GPU smoke must complete before pilot; background/log/yield.
Attempt002 launched timeoutPID2394058. GPU smoke COMPLETED 2 updates27.841263s:
fullDEV inference parity, single backward, auxiliary gradient and encoder/decoder
updates, frozen buffers all passed. Chain advanced to seds-masked-pose-002;
child process confirmed live at handoff. Logs in c08-masked-pair-002/. Remaining
above is BEFORE attempt002; charge its children together once at next collection.
Do not poll training; wait for actual user completion message per standing request.

LATEST USER RETURN: C09 frozen pair completed; XY and XYZ both select step64
mean77.745665 (T77.071291/V78.420039), below previous C09 77.842004 and C04
77.938343. Final XY77.263969/XYZ77.360308 both fail V2T guardrail. Freeze-only
repair did not help; defer this additive geometry family, NOT all future 3D.
Keep corrected H4W cache and compact checkpoints. No further extraction.
Charge child times 24.541166+144.602654+171.974329=341.118149s once, not parent
wall358.712285 twice. V2 used16869.318229s, remaining4622.681771s, currentpool
2548.065376s (old382.616394+unallocated1692 unchanged).

CURRENT: C08 train-only masked-pose reconstruction + matched clean continuation.
See METHOD_MASKED_POSE_C08.md. Full TRAIN7096, release base, seed42/B32/1epoch,
222 updates; native RGB/2D inference retained. GCN weights1e-6 + fusion1e-5;
decoder1e-4 only treatment, mask20%, reconstructionweight.05, targetXY/256-.5.
GCN BN/dropout eval in both arms; no new data/teacher/scorer or TEST use.
Four focused CPU tests passed0.418s. Register smoke2steps -> treatment -> control,
c08-masked-pair-001; limits240/1080/1080s,parent2460s, within current pool.
Reserve <=8GiB new storage, disk76GiB free. Background/log/yield, no polling.
Launched jobv2-c08-masked-pair-001 timeoutPID2308132, confirmed live once;
child logs/reports artifacts/slret_goal_v2/c08-masked-pair-001/. Real-data smoke
and training results pending at handoff; no claim that GPU checks passed yet.
Next actual user return: collect reports, compare both selected and fixed111/222
against release, matched control and C04; charge child runtimes once. Mask-only
control needed later for reconstruction-specific attribution if positive.

LATEST USER RETURN: fusion512controlcompleted128steps165.271246s,selectedinit
77.552987,step64/128mean77.167630(T76.493256,V77.842004). Geometryremainedzero,
LRgroupsfusion1e-5; batchorder/datahash match both priorC09arms. XY/XYZselected
77.842004 are+.289017vsselectedcontrol,+.674374 atfixed64. ExplicitZlead still0.
Single-seedpathwaylead notSOTA/capacity-controlled informationattribution.

RegisterC09refinement1: --geometry-freeze-fusion,trainonlygeometry1e-4,freeze
originalfusionparameters AND dropout/BNmode. Samebase,TRAIN512,seed42,B32,
8epochs128updates,DEV0/64/128,earlystopmean-drop>2pp,guardraileachR1>=ref−.5.
Motivation: bothfusioncontrol andlatejointadaptationV2Tdecline; testdrift
hypothesis without changingarchitecture/LR/data. Counterfactual notprovenyet.
8focusedCPUtests pass0.459s inclgradientthroughfrozenfusion andonlygeometry
updates; real2stepsmoke assertsentirefusionstateunchangedbeforefullpair.
Newchain c09-geometry-frozen-pair-001 / jobv2-c09-geometry-frozen-pair-001;
children seds-geometry-frozen-smoke-001, seds-geometry-frozen-xy-001,
seds-geometry-frozen-xyz-001. Limits240/900/900s,parent2100s.
LaunchedtimeoutPID2281366,confirmedliveonce;logs/reports in
artifacts/slret_goal_v2/c09-geometry-frozen-pair-001/. Yieldnow,no pollingloop.
Chargecontrolonce:V2used16528.200080s,remaining4963.799920s,currentpool
2889.183525s,reserve2100s;old382.616394s+unallocated1692s unchanged.
Disk77GiBfree;retaincurrentC09cache/compactcheckpointsandincumbent;no deletion.
Nextactualuserreturn: comparefrozenpair bothdirections/fixed64/128 andselected,
vspriorjointpair,referenceandC04. Noautopromotion/followon;launch/log/yield.

LATEST USER RETURN: C09pair001 completed. Smoke21.503825s,XY154.318401s,
XYZ154.329949s; charge330.152175s childtimes once. Bothselected64mean77.842004,
T77.071291/V78.612717,+.289017vsrelease,−.096339vsincumbent77.938343.
XYZ−XYselected0pp. Bothfinal128mean77.360308,V2T77.263969(failsguardrail).
Model plumbing passed: zero-initparity,geometryupdated,frozenbuffers/weights,
compactroundtrip,groupLRs. Different scorematrices showZpath not identical;
sameR1 alone not evidence that3D never helps. See RESULTS fullR1/5/10.
Next C09matchedcontrol: freeze geometrybranch atzero,train onlyfusion1e-5,
sameTRAIN512/order/128steps/DEV0,64,128/base/native losses/seed42. Generic
fullTRAINfusion controls cannot attribute this subsetgain. NoLRsweep yet.
New --geometry-control retains zero branch solely to preserve identical data
and compute plumbing; geometry parameters frozen/excluded fromoptimizer.
7focused CPU tests pass0.424s, including no nonfusionparameterchange.
Keep both C09compactbest+last,cache andallhistoricalmetrics; disk78GiBfree,
V2artifacts16GiB. No further pruning needed for this<256MiBcontrol.
V2used16362.928834s,remaining5129.071166s,currentpool3054.454771s;
reserve900s for seds-geometry-control-fusion-001, no budgetincrease.
Launched timeoutPID2270567,jobv2-c09-fusion-control-001,confirmedliveonce;
log artifacts/slret_goal/jobs/v2-c09-fusion-control-001/console.log. Yieldnow.
Nextactualuserreturn: collectcontrol,compareselected+fixedstepsagainstXY/XYZ,
then choose targeted geometry refinement or nextmodeldirection. Noautomated
follow-ontrials; background/log/yield unchanged.

LATEST USER RETURN: corrected extraction002 COMPLETE1031/1031,60023/60023frames,
wall3753.572271s (~62.6min),output53672160bytes (~51.19MiB),peak4078313984bytes.
No detector training;2hand labels/21keypoints/trainer-absent invariant recorded.
Current cache scan checks all1031 XYZ shapes/finite values and exact planned
frame IDs/clip starts. TRAIN512,DEV519, no TEST. Filesystem79GiBfree.
Charge extraction once: V2used16032.776659s;remaining5459.223341s.
Tranche3+4 remaining1584.606946s plus old382.616394s/unallocated3492s.
Allocate tranche5=1800s fromunallocated3492s, leaving1692s; available current
pool3384.606946s, reserve2100s for C09 smoke+XY+XYZ. No total allowance increase.

C09 integration implemented: optional geometry fields in shared train input
adapter/native DEV evaluator, all original tensors/scoring unchanged. Dataset
wrapper preserves native ID order and fullDEV, validates cache clips/masks.
Only fusion+geometry train; explicitLRfusion1e-5/geometry1e-4, all frozenBN
remain eval; checkpoint adaptation-only plus base reference (best+last).
Six focused CPU tests passed0.413s for geometry alignment/depth isolation,
parameter groups, original input forwarding and compact checkpoint rebuild.
Real-data smoke/fullDEV step0 parity still to run; no new retrieval gain claim.
Registered chain c09-geometry-pair-001: smoke2steps then XY128/XYZ128, both
seed42,B32,8TRAIN512epochs,fullDEV0/64/128,earlystop ifmean drops>2pp.
Save same batch order and data provenance; compare both directions and full
R1/5/10 vsrelease77.552987/currentincumbent77.938343. Child limits240/900/900s,
parent2100s. Fail closed on smoke, source drift, frozen state or LR mismatch.
Launched timeoutPID2254343,jobv2-c09-geometry-pair-001,checked live once.
Primary progress report artifacts/slret_goal_v2/c09-geometry-pair-001/run.json;
perchildlogs in samefolder. Yield now, no polling until user's completion message.
Next actual user return: collect chain/children, charge measured usage once;
compare depth vsXY before attribution/promote. No additional automatic jobs.

LATEST REPAIR + USER STORAGE AMENDMENT2026-09-19 (supersedes below):
User explicitly authorizes pruning unused old checkpoints, retaining methods
still in use. Deleted15 fixed-allowlist V2 checkpoint files,20593818524bytes
(19.179488GiB), permanently; all code/config/log/metrics retained. Manifest:
artifacts/slret_goal_v2/storage-prune-20260919-001/manifest.json. Keep release,
C04 incumbent+replications+controls, correctedC07best, all shared features,
datasets and external H4W assets. V2artifacts15GiB,filesystem79GiBfree.

Critical runtime correction: nn.Module.eval() recurses to YOLO.train(False),
whose overloaded method starts detector training on coco8-pose. Logs of prior
sample002 confirm this occurred; earlier overlay/cost assessments are INVALID
for untouched H4W inference. Sample001 used same unsafe call and is likewise
not accepted as clean inference. Raw reports preserved, no scientific NO-GO.
Extraction001 caught this on initial launch check, intentionally stopped
process group2142912 (not hard timeout): failedwall55.416033s,0 saved videos,
14 transient frames not committed. Misleading TimeoutError is signal-handler
text, not actual timeout. External side-effect runs main/runs/pose/train-2 and
train-3 retained pending scoped cleanup; original checkpoint paths not replaced.

Fix: set every module.training=False and freeze parameters directly, bypass
overloaded workflow methods. Assert detector two classes(left/right),21×3
keypoints, trainer absent. Regression test rejects invoking child.train.
safe-eval-sample003 COMPLETED13.774706s,24 frames,inference1.960022s,
detector.320883s,peak4078313472bytes; invariant passed and log has no trainer.
All3 clean overlays inspected: hands now broadly track signer, residual errors
remain; no annotated3D accuracy claim. C09 geometry hypothesis still untested.
Charge failed00155.416033s +safe sample00313.774706s once: V2used12279.204388s,
remaining9212.795612s. Tranche3+4 pool5338.179217s; reserve5200s for corrected
extraction002, expected~4800–5100s with corrected detector (~.073s/model frame).
No total allowance increase; remaining old382.616394s+unallocated3492s intact.
New extraction ID h4w-ph-center-pilot-002 / job v2-h4w-center-pilot-002.
Launched timeoutPID2152660,hard5200s,confirmed live once;10 focused CPU tests
pass including overloaded-train regression. Yield; no monitoring loops.
Same frozen TRAIN512+DEV519/60023frame plan, no use/resume of corrupted001.
Next actual user return: inspect002report, chargeactual, integrate training
geometry routing and matched XY/XYZ trials; no auto training while waiting.

LATEST USER RETURN: person-crop sample002 COMPLETED68.140132s,24 frames,
no detector fallback, peak4.209842GiB. All3 overlays inspected: placement
offsets remain, particularly hand/body anchoring. Stop sample-crop diagnosis;
record limitation and proceed to C09 user-priority estimated3D retrieval pilot.
No baseline replay and no further environment install.
Implemented exact native clip-center geometry cache/loader and optional
across-clip temporal convolution (clip_temporal=True). Native pose2D/RGB inputs
remain unchanged.9 CPU tests passed0.229s; plan-only validated60023 exact source
frame IDs across hash-selected TRAIN512 + complete DEV519. No interpolation,
one observed center per16frame native clip (offset8), up to64 clips/video.
Extractor h4w-ph-center-pilot-001 / job v2-h4w-center-pilot-001, hard4800s;
estimated~3800–4300s,<6GiBVRAM,<128MiBcache;60GiBfree,V2artifacts~34GiB.
Registered design and follow-on matched XY/XYZ recipe: METHOD_POSE3D_C09.md.
Trainer/evaluator routing of geometry is still pending; no retrieval run or
claim yet. Do not use the cache while status is incomplete; exact batch clip
starts/masks must match before every sample can be consumed.
Charge sample00268.140132s once: V2used12210.013649s; remaining9281.986351s.
Tranche3remaining1807.369956s. Allocate tranche4=3600s from7092sunallocated,
leaving3492sunallocated and old382.616394s residual; reserve4800s from
tranche3+4 pool5407.369956s. Total21492s allowance unchanged.
Next actual user return: read extraction terminalreport, charge actualwall,
complete trainer/evaluator geometry routing, smoke on real cache then matched
XY/XYZ retrieval pilots. Save adaptation-only deltas to avoid2GiB/model waste.
No chained training while waiting; launch/log/yield rule remains authoritative.

LATEST USER RETURN 2026-09-19: user reports h4wpp fully configured. Confirmed
interpreter /home/haipd/miniconda3/envs/h4wpp/bin/python, torch2.1.1+cu121,
CUDA available; do not reinstall. Existing h4w-ph-train-sample-001 now has a
completed report from PID2112939:24 finite TRAIN frames, wall70.350850s,
inference1.542257s, peak4.20GiB. This later run is separate from runtime002's
failed missing-kornia child; preserve the failed parent report unchanged.
AI inspection of all3 first-frame overlays finds hand/body placement offsets;
not a retrieval failure or proof depth is unusable. Coordinate formula matches
native model; full-frame crop differs from native demo's person-detector crop.
Register h4w-ph-train-person-sample-002: same24 frames, native YOLO person crop,
otherwise same checkpoint/forward.5 focused CPU tests passed0.180s (unittest;
pytest absent in both environments, no package installation needed).
Charge runtime00262.155916s plus later sample70.350850s once, conservatively
including externally completed sample: V2used12141.873517s, remaining9350.126483s,
tranche3remaining1875.510088s. Reserve240s for crop sample, no budget increase.
Background job v2-h4w-person-sample-002; no chained full extraction/training.
Launched timeoutPID2125058, hard240s, checked alive once; yield per user rule.
Next actual user return: inspect report/overlays, choose geometry extraction
policy and volume within budget, then connect added-XY/XYZ branch. Original
RGB/pose2D remain. Do not re-run environment bootstrap or baseline audit.

LATEST USER RETURN: correctedC07 COMPLETE, selected11177.745665 below incumbent;
keep checkpoint, defer. Current user-priority additional H4W Pose3D pathway;
RGB and original pose2D MUST remain. User confirmed deleting h4wpp environment
and explicitly authorized rebuilding it/downloading packages.
New pose3d_branch.py +3CPUtests passed: original streams unchanged at zero init,
CLS/pad preservation, XY-vs-XYZ no depth leakage through scale, gradients/mapping.
This is branch code only, not end-to-end retrieval training or measured3D gain.
Extraction wrapper prepared for24 TRAIN frames; environment/bootstrap and sample
must succeed before corpus extraction or retrieval training is admitted.

LATEST 2026-09-19: C06 intentionally stopped by explicit user request, last
logged419/666, DEV22277.263969; do not resume. Raw failure is SIGTERM, not
spontaneous runtime failure; see RESULTS and chain user_stop.json.
Current C07: pre-pooling anatomical joint exchange; strategy/card in
METHOD_SELECTION_C07.md. Borrowing/combining prior-art modules is explicitly
allowed by user: evidence of benefit takes priority over novelty screening.
No published idea is rejected merely because it already exists. C07 is an
adaptation hypothesis, not an established novel method or guaranteed gain.
Earlier entries below are chronological history, not live jobs.

UPDATE ON USER RETURN: C07 COMPLETE222, selectedinitialization, final77.167630.
Actual fusionLR1e-4 deviated from registered1e-5 due native optimizer defaults.
One corrected run registered: explicit joint1e-4/fusion1e-5 groups, otherwise
same recipe. Raw original run retained; causal effect of LR mismatch unproven.
Latest user priority after C07 no-lead: investigate/apply Hand4Whole++ Pose3D
from /home/haipd/DexAvatar/Hand4Whole-plus-plus_RELEASE; checkpoints user-provided.
Treat as estimated3D with matched2D controls, not guaranteed extra information.

C01 fused-priority supervision: branch auxiliary weights 1 → .25, fused loss
and RGB–pose matching unchanged. No added labels, teacher, gate or inference
component. Model warm-starts release. First 222-step contrast matches the
existing FP32 control recipe; short horizon is screening, not a family NO-GO.
Code: `methods/seds_adaptation/objectives.py`; runner in `tools/train_seds.py`.
Smoke seds-aux025-smoke-001 COMPLETE:13.014711s, finite losses/gradients,
653 tensors actually updated across clip/clip_rgb/fusion/signbert; weight1
objective matches native on real batch. Peak17.51GiB; no inference change.
Pilot seds-aux025-001 COMPLETE620.507372s,222updates. DEV111mean76.782274,
DEV222mean77.456647,selectorinitialization77.552987; no gain. Incumbent unchanged.
Next candidate C02: fusion-only222 then fusion+upper visual444,LR1e-5/2e-6,
aux1 unchanged; all lower weights/BN frozen. Stage-transition6-step smoke
seds-staged-smoke-001 COMPLETE13.138077s: fusion80tensors updated in stage1,
upper+fusion103 in stage2; no frozen-weight drift; peak2.70GiB. C02 pilot
seds-staged-001 COMPLETE666steps614.156384s, timeoutPID1633233 absent. Launch in
artifacts/slret_goal/jobs/v2-seds-staged-001 and run.json under V2 artifacts.
DEV111/222/444/666means77.360308/77.552987/77.456647/77.360308;
selectorinitialization, no gain. Deprioritize fixed C02 recipe.
Current C03 interaction pilot, no C01/C02 combination: rank32 hand/body product,
aux1,all native trainable,releasewarmstart222updates. Smoke2steps plus necessary
zero-init fullDEV parity (new representation path) passed in smoke; interaction
output weights actually updated, gradients finite. Pilot seds-articulator-001
COMPLETE690.346252s; DEV11177.360308/22277.071291, selectedinitialization,
V2Tfinal−1.348748pp. Defer full-model recipe; no gain.
Current C04: LoRA rank8 uppervisualattention+existingfusion, otherweights/BN
frozen; aux1, LoRA LR1e-4/fusion1e-5,3epochs666steps. Smoke seds-lora-smoke-001
COMPLETE24.617811s: zero-init fullDEVparity, actualB-factorupdate and frozen
weight checks passed; peak2.565GiB. Pilot seds-lora-001 launched hard1800s,
launch/log artifacts/slret_goal/jobs/v2-seds-lora-001. Next read DEV111/222/444/666
and decide refinement/control. FIRST EXPLORATORY LEAD atC04step444:
mean77.938343,T77.071291/V78.805395, +.385356vsrelease,+.289017vsoldincumbent,
guardrailPASS; best.pt saved under seds-lora-001. Pilot COMPLETE580.021394s,
final666mean77.552987; selected444. Provisional incumbent only,
not replicated/attributable/SOTA. Fusion-only matched666-stepcontrol RUNNING:
seds-lora-control-fusion-001 COMPLETE584.461611s,timeoutPID1704042 absent. Launch/log under
artifacts/slret_goal/jobs/v2-seds-lora-control-fusion-001; report underV2artifacts.
Controlselectedinitial77.552987, LoRAselected77.938343, paired+.385356pp.
Pairedqueue lora-paired-replication-001 COMPLETE, timeoutPID1721916 absent;
queue report artifacts/slret_goal_v2/lora-paired-replication-001/run.json,
childlogs samefolder. Sequentialfusion1337→LoRA1337→fusion2026→LoRA2026.
Verify current.child_pid/report before resume; do not duplicate existing IDs.
Summary lora-pairs-summary-001 COMPLETE: selectedpaired+.385356/0/+.289017,
mean+.224791±.200546pp,2positive/1tie. AllLoRAselected exceedrelease, still
exposedDEV/sharedinitialization, not independent/SOTA. V2TselectedR5 drops
.192678pp in seeds1337/2026 vsrelease; fulltradeoffs in summarymetrics.csv.
Current C04b refinement1: B128/3epochs168updates, samearchitecture/loss/LRs;
moreactualnegatives, same3trainpasses but fewerupdates. Smoke
seds-lora-b128-smoke-001 COMPLETE28.082312s,2.8s/update,peak8.42GiB,
LoRAactive/frozenchecks passed. Pilot seds-lora-b128-001 COMPLETE564.681579s,
selected168mean77.745665, +.192678vsrelease,-.192678vscurrentincumbent.
SelectedT2V77.071291/92.870906/96.146435,V2T78.420039/92.870906/95.375723.
FusionB128control COMPLETE865.149460s,selectedmean77.649326; LoRApaired+.096339.
Rawselector chose168over28byroundoff; equalmeans, directionaltradeoff inRESULTS.
DeferB128recipe, keepB32incumbent. Prospective selector tolerance1e-8pp.
C05epsilon.05/fullmodel COMPLETE222steps647.172093s,smoke11.530890s.
DEV11158.574181/final60.597303 (-16.955684ppvsrelease),selectedinitialization.
Drop thisrecipe, no rerun/extension/LoRAcombination. Fullmetrics inRESULTS.
C04c COMPLETE666steps615.138660s+smoke23.504938s; selected44477.842004,
+.289017vsrelease,-.096339vsincumbent; no earlystop. Deferdepth3recipe.
C04d COMPLETE666steps654.159093s+smoke18.398908s; selected44477.842004,
noincumbentgain; deferC04after3refinements thisround. Incumbent unchanged.
C03b COMPLETE666steps1414.744518s+smoke23.211164s, selectorinitialization;
no gainvsrelease/fusioncontrol. Deferthisrecipe; currentC06 temporaldifference.
C06: zero-init rank32adjacentclipdifference adapter aftervideoencoder/pre-fusion,
separateRGB/pose; preserveCLS/padding. Frozenencoders; trainadapter+fusion.
Release/seed42/B32/3epochs666steps, LR1e-5,aux1, noLoRA/C03/smoothingcombination.
CPUidentity/mask/order/gradient/roundtrip passed; GPU2steps+zero-initDEV gatepilot.
Prospective stop if DEVmean drops >5ppvsrelease, saveslast/stopping_reason.
Driver tools/run_c05_pilot.py --profile temporal-delta (notC05profile).
Chain c06-temporal-smoke-pilot-001: smoke180s→pilot1800s, overallhard2100s.
Sources hashguarded; no edits whilelive, noTEST, noautomaticfollow-onjob.

## Resource tranches

Allocate 7200s (2h) from V1 remaining local allowance ~21492s; actual spend
used7009.523341s throughB128control; unused190.476659s oftranche1 retained.
Allocate tranche2 3600s fromremaininglocal~14482.476659s; totalbudget unchanged.
ThroughC03b totalV2used10417.383606s; remaininglocal11074.616394s.
Tranches1+2allocated10800s, residual382.616394s. Chargechildtimes once.
Bookkeepingfix: oldunallocated10882.476659 doublecounted190.476659; true10692s.
Allocate tranche3 3600s, newunallocated7092s; no totaltop-up.
ReserveC06chain2100s againsttranche3; leavepriorresidual382.616394s unchanged.
After C06 intentional stop, release reservation and charge388.311097s once.
V2used10805.694703s, remaininglocal10686.305297s; tranche3remaining3211.688903s.
Reserve C07 smoke+pilot hard2100s within tranche3; no budget increase.
After original C07 completion charge594.191356s childtimes once: V2used11399.886059s,
remaining10092.113941s; tranche3remaining2617.497547s. Reserve correctedC07 hard2100s.
After correctedC07 charge585.144735s once: V2used11985.030794s,
remaining9506.969206s; tranche3remaining2032.352812s. Reserve2000s for isolated
runtime rebuild +24frame sample; account setup vs inference separately, no top-up.
User-authorized environment storage is separate from36GiB V2artifacts cap;
clone source environment with --copy, do not alter shared existing environments.
Disk75GiB free before setup, require40GiB to start and keep15GiB reserve.
V2storage36GiB; current~31GiB,filesystem78GiBfree, nextchain<=2GiB,
preserve15GiBreserve; retain allhistoricalartifacts. No cloud or otheruserjobs.
One GPU job at a time; observed48519MiB free
beforelaunch,0% utilization.
First smoke hard180s; pilot hard1200s, expected ~660s, ~22GiB VRAM.
Before C03 admission, extend V2 storage tranche8→12GiB to retain both earlier
pilots and one full-model candidate without deleting history. Filesystem108GiB
free at latest check, still15GiB reserve. This changes storage allocation only,
not the7200s compute tranche or selection protocol. V1 remains ~41.45GiB.
Initial filesystem111GiB
free, preserve15GiB reserve. V1 agent-only44GiB subcap reallocated for V2.
Each pilot retains only best model + final resumable state, scores/logs; no
per-batch tensors. Temporary atomic replacement of V2-owned best files allowed;
no deletion of historical/user data. Broader tranche requires usage update.

## Process rule

USER AMENDMENT 2026-09-19 (authoritative): when training/feature extraction
requires waiting, launch background with persistent logs, report job/log paths,
then yield to user. Do NOT repeatedly poll/sleep or spend turns monitoring.
User will report completion/request a check; only then inspect terminal status
and logs and resume research. Preserve bounded timeout and error logs. Do not
kill the active job or treat waiting-for-user as scientific failure/completion.
User reported previousjob done; terminalcompleted/exit0 verified2026-09-19.
CorrectedC07 terminalcompleted; no continuation queued.
Currenthandoff runtime rebuild→24 TRAIN frame sample, hard2000s; launch in
artifacts/slret_goal/jobs/v2-h4w-runtime-sample-001/launch.json when created.
Launched2026-09-19 timeoutPID2057572; isolated --copy clone started. Sourceenv
size8.5GiB; no overwrite of existing environment. Yield to user after launch;
do not poll for installation/extraction completion or auto-launch further work.
CORRECTION: initial launch check found terminalfailure, not live. Offline clone
failed because Conda package cache lacks required archives (OfflineError).
Runtime001 wall24.335958s charged; no GPU inference/extraction. Agent-created
partial prefix moved recoverably to h4wpp-incomplete-20260919-001; no deletion.
Repair removes --offline under user's explicit download authority, new report
h4w-runtime-002, new job v2-h4w-runtime-sample-002, hard2000s. No silent retry.
Online repair launched timeoutPID2061755 and verified live once; yield now.
V2used12009.366751s, remaining9482.633249s; tranche3remaining2008.016855s.
Reports artifacts/slret_goal_v2/h4w-runtime-001/run.json and
artifacts/slret_goal_v2/h4w-ph-train-sample-001/run.json if sample starts.
Next ON USER RETURN: check setup/sample terminalreports; if failed repair only
the concrete missing dependency, preserving incomplete environment/logs. If
sample succeeded inspect projection images, timing and frame/joint mapping,
then register extraction volume and matched addedXY/XYZ retrieval pilots.
No full corpus autoextraction, no runtime success or3D gain claim from source alone.
Do not restart or queue nextjob automatically. Preserve user background/yield rule.

Implement/train/refine; no repeated overall audits. Novelty unresolved does
not block pilots. Skill confirmation/retry defaults are superseded by explicit
V2 authority to implement and repair bounded experiments autonomously.
