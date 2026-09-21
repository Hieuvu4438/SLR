# C09 — additional clip-aligned estimated geometry

## Latest decision: defer current additive branch after frozen-fusion pair

XY/XYZ selected64 both77.745665, below previousC09 77.842004 and C04 77.938343;
final128 mean77.263969/77.360308, both V2T guardrail failures. Fusion state stayed
exactly fixed. This intervention does not support a fusion-drift-only explanation
or an explicit depth benefit. No extra extraction or blind LR refinement.
Keep corrected51MiB cache and compact weights for a future input-specific idea.
Next C08 tests masked-pose reconstruction, independent of additional geometry.
Frozen pair child charge341.118149s, remaining4622.681771s. Full detail RESULTS.md.

## Refinement1 — train geometry through frozen original fusion

MaterialPassport: academic-research-suite/experiment-agent,inline run,
2026-09-19; verificationCPUtests only fornewpolicy, retrievalresults pending.
Fusion-onlyTRAIN512control complete165.271246s,selected77.552987,step64/128
77.167630. SameIDs/order verifiedvsXY/XYZ. Priorjointgeometryselected77.842004
is+.289017vscontrolselected,+.674374 atfixed64; explicitZ−XYstill0ppR1.
This supports exploringpathway, not aZcausalclaim orSOTA.

Hypothesis: updatingoriginalfusion with512examples introduces avoidableV2Tdrift;
trainingonlynewgeometrybranch throughfrozenfusion may retainoriginalalignment.
Evidence: fusioncontrolV2Tdeclines; XY/XYZlateV2Talsofalls. Not proof ofcause.
Singlechange: freezeallfusionparameters/stateand keepfusion.eval(); preserve
gradientflowthroughfusion tonewbranch. No teacher,lossreplacement,LRsweep,
newgeometryfeatures orremoving2D/RGB. GeometryLR1e-4unchanged,checkpointbase
release(notresume),seed42,B32,TRAIN512×8epochs128updates,fullDEV0/64/128.
Samezero-initcheck,referenceguardrail−.5eachdirection,earlystopmean-drop>2pp.
BothXYandXYZ samepolicy/data/order. This countsas1ofmax3motivatedrefinements.

Implementation: --geometry-freeze-fusion inextendedrunner; explicitoptimizer
onlygeometrygroups; smokechecksoutputupdate andentirefusionstateunchanged.
8CPUtests passed0.459s, includinggradienttransmissionandno fusionparameter
update. Real-data smoke2updates beforepilot; chainfailsclosedonanyinvariant.
New run command: `research/slret_goal_v2/tools/run_geometry_pair.py --freeze-fusion`
via sedsPython. Chain c09-geometry-frozen-pair-001, childsmoke/XY/XYZ prefixes
seds-geometry-frozen-, suffix001. Childlogsandrun.json inchainfolder.
Hardparent2100s/children240,900,900s; expected~350–450s,peak<8GiB,
storage<512MiB,filesystem77GiBfree. Keepcompactbest/last;no extraextraction.
Remaining4963.799920s,currentpool2889.183525s,reserve2100s.No budgetincrease.
Selectusingunchangedmetric;promoteonlyifleadvsreference/control andconsider
currentincumbent77.938343. Ifnogain orlateoverfitpersists, nextrefinementneeds
newinput-specificevidence; don'tassume freezingfusionfixesposeestimation.
Background/log/yield: noautomatedfurtherexperimentsafterthisregisteredpair.

## First pair result and matched fusion-only control

2026-09-19, ARS inline run/collect; local reports, not statistical validation.
XY andXYZ bothselected64mean77.842004,+.289017vsrelease,−.096339vsC04;
depth contrast0ppR1. Bothfinal128mean77.360308,V2Tdeclines. Fullmetrics inRESULTS.
Geometry is active and scores differ, so don't diagnose "Znotused" from equalR1.
Attribution remains unresolved because both runs also fine-tune existingfusion.

Register seds-geometry-control-fusion-001: identicalTRAIN512/nativeorder,
seed42,B32,8epochs128updates,releaseinitialization,aux1,nativeFP32moments,
fusionLR1e-5,DEV0/64/128,sameguardrail/earlystopdrop2pp. Onlydifference:
geometrybranch is frozen atzero and excludedfromoptimizer. Fullzero-initDEV
check ensures unchanged reference; step2/final must confirmzerooutput and
onlyfusionparameterchanges. Cachedgeometry may still be computed but has no
effect; control isn't claimed cheaperinference or a newmethod. Do not reuse
fullTRAINfusion results as this subsetcontrol. No speculativeLRchange yet.

Command: seds Python torch.distributed.run --standalone --nproc_per_node=1,
train_seds_extended.py --run-id seds-geometry-control-fusion-001 --geometry xy
--geometry-control --policy fusion --aux-weight1 --batch-size32 --seed42
--lr1e-5 --sign-lr1e-4 --epochs8 --early-stop-drop-pp2 (optionvalues spaced
inactualcommand saved inlaunch.json). Hard900s,expected~150–200s,<8GiBVRAM,
<256MiBcompactcheckpoint/artifacts. No nextjob queued; waitforuser.
7CPUtests passed0.424s forcontrolgroups,freeze,checkpoint andinputalignment.

If control matches geometry, existingfusionadaptation explains observedgain
under thisrecipe; then consider an input-specific representation/refinement,
not promote3D or assume familyimpossible. If geometry exceedscontrol, retain
lead but still distinguishXYZ fromXY before depthclaim. No TESTselection.
Chargepreviouschildren330.152175s once: remaining5129.071166s, reserve900s.

## Valid extraction complete; matched pair launch2026-09-19

Extraction002 completed3753.572271s,1031videos/60023frames,53672160bytescache,
peak4078313984bytes. Full cache ID/shape/finite checks passed, no TEST.
Trainer integration now implemented (supersedes pending notes below): optional
geometry forwarding in shared train adapter and nativeDEV evaluator; original
RGB/pose2D/scorer retained. Exact native batch clip/mask checks reject misalignment.
Only geometry+fusion train; explicit LR1e-4/1e-5, compact adaptation checkpoints
require the unchanged release base. CPU reconstruction test restores all keys
of a toy base+delta exactly; actual model roundtrip asserted in real smoke.

Chain: `research/slret_goal_v2/tools/run_geometry_pair.py` (seds environment).
Outputs: artifacts/slret_goal_v2/c09-geometry-pair-001, with childlogs;
seds-geometry-smoke-001, seds-geometry-xy-001, seds-geometry-xyz-001 reports.
Commands perchild: torch.distributed.run --standalone --nproc_per_node=1
train_seds_extended.py --geometry {xy,xyz} --policy fusion --aux-weight1
--batch-size32 --seed42 --lr1e-5 --sign-lr1e-4 --epochs8 --early-stop-drop-pp2.
Actual CLI uses spaces between option/value; fullcommands stored in chainreport.
Smoke adds --smoke,XYZ2updates; requires fullDEV step0 scoreparity, geometry
output update, frozen buffers/weights unchanged, compact state roundtrip and
actual optimizerLR assertion. Eachpilot128steps,DEV0/64/128. FullDEV zero-init
parity repeated perarm as fail-closed integration check, not baseline retraining.
Selector includes initialization; eachdirection guardrailreference−.5pp;
earlystop ifmeanR1 drops>2pp at scheduledDEV. Same condition for botharms.
Pair validates identical TRAIN IDs/order and DEV IDs/cache provenance.

Expected~600–1000s entire chain (uncertain newloader overhead); hard2100s,
children240/900/900s. Peak estimate<8GiB, storage reservation1GiB forpair,
adaptation-only weights plus optimizer/RNG. Parent fails on any child failure;
no automatic retry/refinement/promotion. Background/log/yield user rule applies.
Used16032.776659s/21492s,remaining5459.223341s after ETL. Allocate1800stranche5
from3492sunallocated, leaving1692s; pool3384.606946s reserve2100s. No top-up.
Incumbent explicitly updated from C04selection77.938343, not olderV1control.
6focused CPU tests passed0.413s; efficacy remains unverified until pair completes.

## Runtime correction before valid extraction

Previous sample002 and extraction001 used top-level model.eval(), which
recursed into Ultralytics YOLO.train(False) and started its training workflow.
This invalidates previous clean-inference/crop-quality/timing interpretations;
do not conclude H4W3D is poor from those outputs. Raw reports/logs retained.
Extraction001 intentionally stopped55.416033s,0 saved videos (14 transient
frames discarded). Its TimeoutError text reflects SIGTERM handling, not timeout.
Fix sets module.training=False directly across modules and freezes parameters,
bypassing overloaded workflows. Assert original detector labelsleft/right,
keypointshape21×3 and no trainer; regression test covers this exact bug.
Clean sample003 completed13.774706s,24 frames,model1.960022s,
detector.320883s,peak3.798GiB. No training workflow in log; invariant passed.
All3 overlays inspected; residual errors remain but hands broadly track signer.
Original H4W checkpoint paths remain unchanged; unintended generated runs
main/runs/pose/train-2 and train-3 are separate side effects, not inputs.

Corrected extraction uses ID h4w-ph-center-pilot-002, job
v2-h4w-center-pilot-002, same60023 frames, hard5200s, expected4800–5100s.
Old extraction001 is not resumed. Actual remaining9212.795612s after charging
failed001 and clean sample003 once, combinedtranche3+4remaining5338.179217s.
Storage amendment: user authorized unused checkpoint removal;15 files pruned,
19.179488GiB freed, V2~15GiB and filesystem~79GiBfree. Retention manifest under
artifacts/slret_goal_v2/storage-prune-20260919-001/manifest.json; retain incumbent,
controls, correctedC07best, original data/features and all experiment records.
Historical card/timing below is superseded by this correction where conflicting.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline run.
- Origin Date:2026-09-19. Version:c09_pilot_v1.
- Verification Status: CPU mapping/branch checks passed; extraction and retrieval pending.
- Authority: user requests H4W++3D, expressly retaining original pose2D and RGB;
  V2 goal permits bounded implementation/pilots, with background/log/yield.

## Decision from completed samples

Person sample002 completed24 TRAIN frames in68.140132s, inference1.524722s,
detector0.163816s, peak4.209842GiB. Zero person-detector fallbacks; mean joint
projection-inside-image fraction0.894558 (NOT accuracy). All3 first-frame
overlays inspected by AI. Native person crop does not eliminate hand/body
placement offsets. Camera projection arithmetic and inverse-affine mapping
match local source; cannot infer3D truth or retrieval value from these pictures.
No further crop audit. Use the native person-crop recipe and test usefulness.

Hypothesis: wrist-relative finger geometry and body-relative hand placement,
including estimated depth, can complement2D/RGB despite monocular ambiguity.
Full image-space translation is removed by local/body normalization; this
does NOT correct mistaken hand articulation, rotation or body anchoring.

## Registered pilot card

- C09, estimated-geometry representation intervention; novelty unresolved.
- Base: SEDS released PH checkpoint, reference meanR1 77.5529865125;
  incumbent C04 meanR1 77.9383429672. Do not initialize from failed candidates.
- Data: TRAIN512 selected by ascending SHA256("pose3d-pilot-v1:"+video_id),
  independently of captions/retrieval scores; all DEV519 in native gallery order.
- Temporal sampling: exact raw decoded frame at offset8 of every native16frame
  window, following existing adapted metadata. Up to64 clip tokens/video;
  deduplicate shared centers only for extraction. Total60023 distinct per-video
  frames (TRAIN30207, DEV29816). No interpolation/nearest-frame substitution.
- Cache includes XYZ49, UV49, exact raw frame IDs, clip starts/centers, crop
  boxes, detector fallback and projection-inside diagnostics. No mesh files,
  new RGB extraction, changed2D input, caption supervision or TEST construction.
- Geometry branch: shared hand MLP63→64, body21→64, concatenate192;
  one center per clip, Conv1d192→192 kernel3 across clip tokens (not within a
  synthetic window), GELU, zero-output192→512 projection before native CGAF.
  Add to native pose hidden tokens; original pose2D and RGB remain active.
  Original pose auxiliary loss also sees the augmented pose representation.
- XY/XYZ: identical architecture and data. XY zeroes Z before processing;
  both normalize by XY shoulder distance. Wrist-relative hands, shoulder-center
  relative body, original camera axes retained. XY already reflects H4W's3D
  pretraining; comparison isolates explicit Z use, not all3D-prior benefits.
- Proposed first training pair: each seed42,B32,8epochs=128updates on same512
  examples/order; native losses(aux1), freeze original encoders and their BN;
  train branch LR1e-4 plus fusion LR1e-5 with explicitly logged optimizer groups.
  Native FP32 optimizer moments; eval at0/64/128 fullDEV, reference guardrail
  each R1 direction ≥reference−.5pp. Freeze these before training launch;
  any necessary implementation repair must be logged before new run.
- First real-data smoke: zero-init score parity on fullDEV once for this new
  representation path; finite loss/gradients, branch-output parameter update,
  no frozen-parameter drift. CPU-only tests are not a substitute for this smoke.
- After the pair, compare both directions R1/5/10, meanR1 vs release/incumbent,
  XYZvsXY and compute. Positive exploratory lead warrants matched fusion-only
  control on TRAIN512, more TRAIN coverage and replication; no SOTA claim from
  this small subset or repeatedly selected DEV. A negative is scoped to recipe.

## Extraction execution and budget

Command (h4wpp, no new installation):

```bash
/home/haipd/miniconda3/envs/h4wpp/bin/python -u research/slret_goal_v2/tools/extract_h4w_geometry.py --run-id h4w-ph-center-pilot-001
```

Background wrapper: research/slret_goal/tools/launch_bounded.py,
job v2-h4w-center-pilot-001, hard4800s (80min), no follow-on training queue.
Report/cache: artifacts/slret_goal_v2/h4w-ph-center-pilot-001/.
Log: artifacts/slret_goal/jobs/v2-h4w-center-pilot-001/console.log.
Expected~3800–4300s including model load/detection/decode/serialization;
<6GiBVRAM,<128MiBcache. Input videos decoded sequentially and full frame count
checked. Small-sample timing extrapolation is uncertain; hard timeout applies.
Completed videos are atomic and retained on error/timeout. Explicit --resume
requires unchanged input/source/package contract and retains previous report;
never train from a partial cache or quietly omit failed examples.

Previous sample002 charged once: used12210.013649s of21492s, remaining9281.986351s.
Allocate3600s tranche4 from existing unallocated7092s, leaving3492s. Combined
tranche3+4 availability5407.369956s, reserve4800s for this ETL. No total top-up.
~34GiB V2 artifacts of36GiB, filesystem60GiB free, retain15GiB free reserve.
Future pilots must save adaptation-only deltas plus optimizer/RNG for exact
reconstruction from unchanged release checkpoint, not duplicate entire model.

## Implementation status

`geometry_cache.py` implements subset selection, native raw-center checks,
alignment/mask enforcement, native dataset wrapping and geometry collation.
`pose3d_branch.py` supports the registered clip-temporal mode.9 focused CPU
tests passed0.229s, including XY depth isolation, gradients, CLS/padding and
unchanged original tensors. Plan-only validates all60023 planned frame IDs.
Actual trainer/evaluator field forwarding, matched optimizer groups and
delta-checkpoint reconstruction will be integrated before real-data smoke;
do not claim an end-to-end trained C09 model yet.

## Collision and limitations

User specifically authorizes additional estimated3D input despite historical
rejection of unspecified extra-stream stacks. This is not reopening an old
confidence gate, reranker, teacher, or C06 post-encoder delta recipe: new
geometric measurements drive a representation branch. Reusing a temporal
convolution does not make it the same causal hypothesis as C06.
Known pretrained components are acceptable adaptations, not automatic novelty.
Low-resolution, occlusion, crop jitter,3D hallucination, subset overfitting and
single-frame-per-clip information loss remain real risks. DEV has been used
for tuning; historical PH/CSL TEST exposure remains disclosed and is not used
for new method/configuration selection.
