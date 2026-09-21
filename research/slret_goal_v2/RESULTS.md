# V2 model-improvement results

## C20 timed out: negative common-step signal, no final efficacy verdict

Collected2026-09-21 under ARS experiment-agent. Job TIMED_OUT at the2300s hard
limit after146/666 updates; supervisor wall2303.211750268936s, exit1, and all
recorded PIDs are absent. SIGTERM came from the bounded supervisor, not OOM or a
model exception. Owned-process GPU peak17987272704bytes and max telemetry query
.104757s stayed inside registered sampled safeguards. Trainer run.json remains
stale `running`; supervisor summary/status are authoritative terminal records.

The only post-training DEV evaluation was step111: fused meanR1
77.84200385356455, T2V77.07129094412332/V2T78.61271676300578. Historical GCN-R1
seed42 at the identical step and exact batch order is77.9383429672447, so C20 is
-.09633911368015pp at the common checkpoint. It is-.77071290944123pp below that
control's selected78.61271676300578 and-.86705202312139pp below global incumbent.
Directionally, T2V is+.192678pp and V2T-.385356pp versus common-step control.
Pose meanR1 improves about+.867052pp at111, but that auxiliary-stream change did
not produce a fused-primary improvement and is not a success claim.

The mechanism did activate:830 of4672 sample exposures were RGB-dropped through
step146 (17.7654%); probability followed.2 to.1563909774. Private-RNG, inference
identity, native loss reconstruction, GCN/fusion update and frozen-buffer gates
passed. The planned probability0 terminal/complete-input phase was not reached,
so this is incomplete negative evidence, not a completed rejection. Exact resume
is unavailable because full-run periodic optimizer/RNG checkpoints were not saved.
Given the already-negative common-step primary metric and remaining compute, do
not replay or approximate-resume C20. Park this recipe without a final SOTA or
novelty conclusion.

Step111 checkpoint804973524bytes, SHA256
8a87050d2987f08cc1ac0090fe74d0339021d33a1ece98920319808e4b9ca2c1 retained
with logs/scores/source; no last.pt exists. No TEST accessed.

## C19-R1 completed: directed pose-to-RGB exchange is also below control

Collected2026-09-21 under ARS experiment-agent. Retry002 COMPLETED160/160,
exit0, supervisor wall1737.84938955307s; all recorded PIDs are absent. Selected
and final meanR1=78.22736030828517 versus matched native subset control
78.32369942196533 (-.09633911368015902pp). T2V R1/R5/R10=77.26396917148362/
93.0635838150289/96.14643545279384; V2T=79.1907514450867/93.0635838150289/
95.37572254335261. Curve0/80/160=77.55298651252409/78.131021194605/
78.22736030828517. It is also -.096339114pp below completed bidirectional C19.

All preregistered identity, direction-specific trainability, output activation,
all-gradient, native GCN/fusion, frozen-buffer and delta-roundtrip gates passed.
The active131328-parameter pose-to-RGB adapter reached RGB relative norm.00405009
at160 (step150 .00463493); median/mean preclip gradient norm=.529114/.911993 with
57/160 updates above1. This is adequate learning evidence, not an inactive-module
failure. The direction restriction did not resolve the bidirectional tradeoff.
Defer the global-exchange family with no further LR/rank/slot/within control.

Best checkpoint158353779bytes, SHA256
2962647ac1a8c26750f6dba5622827c2b49802c07420b3b70174b37ca09f020d retained.
Owned GPU peak10057940992bytes; CUDA peak6835746816bytes; max telemetry query
4.569541151868179s. The unused `last.pt`453287783bytes was permanently removed
after recording SHA256 c1591c9693dde4b7a5d934055d7ab655c52a74a219d44ea11be0615d0e9aafd4;
all logs/scores/source/provenance remain. No TEST accessed, no early stop, no
novelty/SOTA claim. Full-TRAIN incumbent remains78.70905587668594.

## C19-R1 attempt001: technical trainability failure, no efficacy result

Collected2026-09-21 under ARS experiment-agent. Job FAILED after2 completed
updates, exit1, supervisor wall261.3941104412079s; all recorded PIDs absent.
Step0 full DEV matched release mean77.552987. Output projection and native GCN/
fusion updates passed at step2, but backward3 all-gradient gate failed before a
third row or any post-training DEV score. No best/last checkpoint was produced;
selection remains initialization. Peak owned GPU5167382528bytes. This is not a
negative retrieval result.

Root cause is source- and report-established: asymmetric attachment initially
froze `to_pose`, then generic fusion policy recursively re-enabled all fusion
parameters. Optimizer consequently contained262656 adapter params instead of the
registered131328, although forward intentionally never uses `to_pose`; the gate
rejected its absent gradients. Repair reapplies directional trainability after
fusion policy configuration and before optimizer construction. It also makes
global-exchange smoke use3 updates, so the upstream-gradient check is reachable.
18 targeted tests pass, including exact regression of this ordering and nonzero
gradients for every active asymmetric parameter. Retry002 changes no scientific
factor and uses a unique ID; attempt001 remains preserved as failed evidence.

## C19 completed: bidirectional exchange ties the matched control

ARS experiment-agent/run collection2026-09-20. Recovery COMPLETED160/160, exit0,
wall971.9356958866119s after the original2403.396547317505s timeout. It restored
the step112 compact model, FP32 BertAdam moments and RNG, recomputed nine unsaved
updates113..121, then continued to160 without resetting LR/warmup. All resume,
identity, adapter-gradient/output, GCN/fusion, frozen-buffer and delta gates passed.
This is valid completed evidence, but not a claim of GPU bitwise equivalence to a
single uninterrupted run.

Selected160 meanR1=78.32369942196532, effectively identical to matched native
control78.32369942196533. T2V R1=77.263969 versus77.071291 control (+.192678pp),
V2T R1=79.383430 versus79.576108 (-.192678pp); gains cancel. T2V R5/R10=
93.063584/96.146435; V2T=93.063584/95.568401. Curve0/80/160=
77.552987/78.131021/78.323699; control=77.552987/78.034682/78.323699.
Adapter relative norms at160 were pose.00332578/RGB.00399755: the path was active,
but small activation is not evidence of useful semantics. No promotion, no SOTA/
novelty claim, and no within-stream capacity control because there is no net gain.
Incumbent full-TRAIN seed1337 remains78.7090558767.

Selected compact checkpoint158353907bytes, SHA256
5a349fc571d530d6ecf9136cff2256a5ab7a50661a59cbbdff515d7c650703c8 retained.
Owned GPU peak9927917568bytes; max query.125861s. The two obsolete recovery
`last.pt` states totaling908681322bytes were permanently removed after completion;
both selected bests, reports, logs, scores and source snapshots remain, with manifest
`storage-prune-c19-recovery-20260920-001`. Next is one directed pose-to-RGB contrast;
if it has no useful gain, defer this family rather than sweep LR/rank/slots.

## C18-002 completed: B64 does not improve matched B32 control

ARS experiment-agent/run collection2026-09-20; local terminal/artifact evidence,
not independent replication/statistical validation. COMPLETED80/exit0, all PIDs
absent. Selected40mean78.22736030828517 vscontrol78.32369942196533 (-.096339pp).
Selected T2V R1/5/10=77.071291/93.063584/96.146435;
V2T=79.383430/93.448940/95.375723. Final80mean78.13102119460501;
T2V=77.071291/93.063584/96.146435; V2T=79.190751/93.063584/95.375723.
Curve0/40/80=77.552987/78.227360/78.131021. GCN+fusion update/frozen-buffer and
delta-load gates passed,80 actual train rows; gradient median.965718,39/80 clipped.
Matched TRAIN512/5120exposures, but80 vs160 optimizer updates: no isolated causal
conclusion about negatives. Defer physicalB64 recipe; no larger-batch sweep now.
Best157300581bytes SHA2568f7e29d1c31b4018ed79df67c71fed9cef2d329e5aa2c4a21a3f2db750c4a0a5
checked; last451180619bytes retained. Incumbent78.709056 unchanged.
OwnedGPUpeak17.628660GB, allocator12.224352GB; sampled compliance, not hard partition.
Max query7.457s explains why a5s query timeout was brittle; repair002 completed.
Charge supervisor1563.841474056244s once. Next C19 representation-level experiment,
not C17 loss rescue. No TEST accessed, no baseline replay/extraction.

## C18-001: infrastructure failure, no trained retrieval result

ARS experiment-agent/run collection2026-09-20, verification of terminal failure
only, not method efficacy. Job v4-c18-batch64-001 FAILED/exit1, all3PIDs absent;
supervisor239.5920069217682s charged once. nvidia-smi query exceeded5s; supervisor
exception cleanup sent SIGTERM to its own workers. No OOM in recorded evidence.
Initial fullDEV finished (fusion mean77.552987); no train_steps.jsonl, best.pt or
last.pt was produced. Stale child report remains running/0; do not treat as live.
No recoverable optimizer state, so retry uses a fresh unique ID, not fake resume.
Observed7.363GB peak does NOT establish B64 training compliance; no completed
update gate. No candidate/no-gain conclusion, incumbent78.709056 unchanged.
REPAIR: single same-recipe v4-c18-batch64-002, querydeadline20s, missing telemetry
still fatal. Query duration/last successful sample timestamp now recorded.
14 scoped CPU tests pass, including deadline propagation/failclosed timeout and
exact retry recipe equivalence; py_compile passes. No C17 recovery or TEST access.

## C17-R1: timed out in final evaluation; USER-CLOSED without recovery

ARS experiment-agent/run collection2026-09-20. Supervisor TIMED_OUT at1800s,
wall1803.0584270954132s, worker exit1, all3PIDs absent. Child run.json remains
stale 'running'/159; do not treat it as a live process or complete success.
Training log has all160 rows; atomically saved last.pt next_batch_index=160 of160.
All prior loss/update/buffer/roundtrip gates passed. Final eval directory empty;
no final160 metrics available. Intermediate80mean77.649326 below control78.323699.
T2V R1/5/10=76.493256/93.063584/96.339114;
V2T=78.805395/93.256262/95.953757. Do not substitute this for final performance.
Preclipnorm median.692504, clipped60/160 vsnative57/160; scale moderation worked,
but retrieval efficacy remains unconfirmed at finalstep. No new incumbent.
OwnedGPUpeak10051649536bytes, below20GB at sampled checks. No OOM evidence.
Best157300581bytes SHA256 e152c297f2fa50ee3aabe9d9ad4dc8ba14116dd86fd8e4831a9f4a12af804e32;
last451181259bytes SHA256 995d749e5e2c769b4f989ca772cbbeb2ed4047591884d812c637651def4c1ce9.
Both hashes checked, files retained. Charge supervisor1803.058427s once, not child.
User then explicitly said to skip C17 and change direction. C17/DCL is USER-CLOSED;
prepared evaluation-recovery scripts were NOT launched, no recovery reservation.
No training replay or missing final-result imputation. Next C18 uses native loss
and changes contrastive batch composition/size, not another DCL coefficient.

## C17: pure fused DCL improves pose readout but not fused R1

Material Passport: ARS experiment-agent/run, collected2026-09-20 on user return.
COMPLETED/exit0/160updates, no earlystop/OOM. All real loss-reconstruction,
GCN/fusion update, frozen-buffer and delta-roundtrip gates pass; workers absent.
Selected80mean77.64932562620423, -.674373796pp matched noaux78.323699;
final160mean77.45664739884393, -.867052023pp noaux final, -.096339114pp release.
Selected T2V R1/5/10=76.493256/92.678227/96.339114;
V2T=78.805395/93.448940/96.146435.
Final T2V=76.685934/92.678227/96.339114;
V2T=78.227360/93.641618/95.953757. Curve0/80/160=77.552987/77.649326/77.456647.
Incumbent78.709056 unchanged. Some secondary recalls increase, not primary success.
Pose final T63.005780/V63.776493 vs controlT61.464355/V60.693642; mean+2.312139pp.
This shows a stream tradeoff, NOT proof of semantic cues or a useful learned gate.

Pure DCL global preclip norm median9.313722/mean9.599460, clipped160/160 at norm1;
control median.528228/mean.912024, clipped57/160. First/last10 pureDCL normmeans
12.104082/8.940617; fused DCL -17.415357/-18.621105 (negative loss is valid).
Native fused CE first/last10=.000596612/.009220450; qmean=.000520674/.006721790,
median q usually atFP32 zero. Stable expm1 still limited by rounded log_softmax.
Small NPC factors and large changed gradients observed, but no causal proof that
clipping alone caused lower recall. Strong full decoupling may overfit easy pairs.
Next: one strength refinement, .95*native_fused_CE+.05*DCL, branches unchanged.
.05 is a coarse warm-start heuristic near ratio of historical median total norms
.528/9.314, NOT gradient balancing measured at common weights. Do not call novel.

Best157300517bytes/last451181195bytes retained; bestSHA256 verified
54ff0ee3f5bd1e8263ef2cc2e8c7e6e5ee1c9ceb517b6bd00cff938acdd83bb8.
Charge supervisor1447.346301317215s once, not child1439.103377s. OwnedGPUpeak
10051649536bytes, allocated6835113472bytes; sampled below20GB, not hardware partition.
One seed, TRAIN512/repeatedDEV, no TEST; no attributable improvement/SOTA claim.

## C16-R2: ties noaux; defer direct SignRep transfer in this tranche

Material Passport: ARS experiment-agent/run collection2026-09-20, actual user return.
COMPLETED/exit0/160updates; all gradient/update/buffer/roundtrip gates passed,
all three workers absent. Selected/fixed160 mean78.32369942196532 (floating-point
tie to control78.32369942196533, NOT a meaningful negative delta).
T2V R1/5/10=77.263969/93.063584/96.146435;
V2T=79.383430/93.063584/95.375723. Curve0/80/160=77.552987/78.131021/78.323699.
At160 vs control: one extra T2V hit, one fewer V2T hit of519; R5/10 unchanged.
R2-R1=+.096339pp, R2-incumbent=-.385356pp. No new incumbent or transfer gain.
Relationloss first/last10means .022820727/.012895659; auxiliary edgegradient
.000480154/.000509403. Increased strength reached the graph and reduced loss,
but did not improve selected/final meanR1 over noaux. This does not rule out
all uses of SignRep or all larger-data/horizon regimes. No more C16 coefficient,
LR or horizon trials in this tranche. RepeatedDEV/one exploratoryseed, not SOTA.
Best162032150bytes and last465376158bytes retained. Independently checked bestSHA256
1af057d5a0d5ba5ef2c643404ed4c8c7cc3965247e9deee0448940f4fdd81027.
Supervisor1187.2980473041534s chargedonce; observed ownedGPU9942597632bytes,
allocator6857040384bytes. No early stop/OOM; sampled20GB bound, not a partition.
Next C17 targets native fused InfoNCE saturation, no teacher, no inference change.

## C16-R1: normalized-distance transfer remains below noaux control

Material Passport: ARS experiment-agent/run, collection2026-09-20, GoalV4.
COMPLETED/exit0/160updates; all update/gradient/buffer/roundtrip gates passed,
workers absent; selected checkpoint SHA256 independently checked (no training rerun).
Selected/fixed160 meanR1=78.22736030828517; T2V R1/5/10=77.071291/93.063584/96.146435;
V2T=79.383430/93.063584/95.375723. Curve0/80/160=77.552987/78.034682/78.227360.
Against matched control78.323699: -.096339pp (same T2V, one fewer V2T hit of519).
Against pointwise78.034682: +.192678pp; against incumbent78.709056: -.481696pp.
R5/10 equal matched control at160. Neither loss has demonstrated incremental utility.
One seed, repeatedDEV selection, TRAIN512; not significance/causal diagnosis/SOTA.
Best162032086bytes +last465376094bytes retained; bestSHA256
a76136f488886d231b57fdf99b2589cbecd22755eef76816517755ed80317411.
Supervisor1222.4315004348755s charged once, exclude child wall. Observed owned-process
GPUpeak9942597632bytes; CUDA allocatedpeak6857040384bytes, below user20GB maximum
at sampled observations (not a hardware memory partition). No early stop/OOM.

Train first/last10 means: relation loss .023042823 -> .014856976; native total
.037407324 -> .047221867; auxiliary edge gradient .000048094 -> .000049755.
Last10 weighted relation .001485698 is ~3.15% of native loss; pointwise had much
larger numerical loss at the same .1 coefficient. Loss ratios do not establish
relative native/aux gradient influence (different tensor norms cannot be compared).
Decision: one final strength contrast .1->1 (C16-R2), all other factors fixed,
to challenge the possible too-light constraint explanation. Not a blind sweep;
if still no improvement over matched control, defer direct C16 transfer recipes.

## C16 matched control: pointwise SignRep auxiliary loss does not help this pilot

ARS experiment-agent/run collection2026-09-20, actual user return, GoalV4.
Control COMPLETED/exit0/160steps, gates passed; checkpoint/hash verified, workers absent.
Same TRAIN512/fullDEV519/seed42/B32/GCN+fusionLR/offload+mathSDPA and0/80/160 selector.
Control selected160mean78.32369942196533; T2V R1/5/10=77.071291/93.063584/96.146435;
V2T=79.576108/93.063584/95.375723. Curve0/80/160=77.552987/78.034682/78.323699.
Pointwise transfer curve77.552987/77.842004/78.034682; minuscontrol at80=-.192678pp,
at160/selected=-.289017pp. T2V samefinal; V2T lowerby.578035pp. No higher R5/10
for transferfinal. Reject attributable gain for this recipe, not all cross-modal transfer.
Incumbent78.709056 unchanged; one exploratory seed/repeatedDEV, not statistical significance.
Controlbest fdfc9c77cc2b7bbc89f9b63fcc3b25dca2d658d1dbef7ce10d4c37f73ae8667a retained.
Charge supervisor1382.313277244568s once (child1374.737633s not added).
GPU allocatorpeak6840464896bytes valid; supervisor totalGPU peak0 INVALID because
torchrun uses setsid for workers. No retrospective totalVRAM-cap compliance claim.
Process-tree tracking repaired before nextpilot, no training replay for logging issue.

TRAIN-only64video cache diagnostic (SHA256-ID ordered first64) found meanoffdiag
cosine.2115326494, quantiles10/50/90=.0072096222/.1515041443/.5176084605;
mean within-video centeredenergyfraction.7626562079. Not collapsed-teacher evidence.
Does not measure student collapse or prove negative-transfer cause. Next one structural
contrast is within-video normalized-distance transfer (C16-R1), not a mean-collapse fix.

## C16 SignRep transfer: small DEV gain, not yet attributable

Collected2026-09-20 underGoalV4; TRAIN512/DEV519, seed42,160updates, noTEST.
Completed160/allgradient-update-buffergates, noearlystop/OOM. Selected160:
T2V R1/5/10=77.071291/93.063584/96.146435;
V2T R1/5/10=78.998073/93.063584/95.375723; meanR1=78.0346820809.
Curve0/80/160=77.552987/77.842004/78.034682; +.481696ppinitialization,
-.674374ppglobalincumbent78.709056. Do not replace incumbent or claimSOTA.
Auxcosineloss first10mean.978503 -> last10mean.546527; learning, NOT causal gain proof.
Training1199.781731s; parent1258.412504s includes smoke46.776685s/overhead, chargedonce.
Peak6.387GiBGPU/27.348GiBhost; savedbest162031958bytes,last465375966bytes retained.
BestSHA256 verified6f2fda6dfba74b5ffdb67d15f5c3c670545707a0765a298dac207fd7380dbcc1.
Decision: retain exploratorycandidate, prioritize matched no-transfer control before
furtherrefinement. FullTRAIN GCN-R1 is not a matched TRAIN512 control. Offload/math
SDPA and historical numerical replay caveat must be held/disclosed in comparison.
Budgetremaining365.970956s insufficient for measured~1200s control; request1800s
additionalallocation. No newjob launched, no automatic allocation or finalclaim.

## C15-R1: ties GCN-R1; defer current pooling family

Material Passport: ARS experiment-agent/run, collected2026-09-20, DEV519 only.
Completed666/allgatespass/noearlystop; selected444mean78.6127167630,
T2V R1/5/10=77.649326/92.485549/96.146435;
V2T=79.576108/93.641618/95.568401. Final666 same recalls/mean.
Curve111/222/444/666=78.034682/78.131021/78.612717/78.612717.
+.192678pporiginalC15, exacttieR1seed42, -.096339ppglobalbest,
+1.059730pprelease. No new incumbent, no novelty/SOTA/causal claim.
Retainbest7516623177459defd2d2041bee47563aebd333af282535ef35914616b1d816e6.
Charge27.8947155476+2115.8825769424=2143.7772924900s once;
exclude parent2154.0446043015s. Defer this pooling family, no extraLR/rank/seeds.
Next: public SignRep checkpoint/strict-load/native-window feature feasibility,
not another baseline audit or unmeasured claim of retrieval improvement.

## C15: below incumbent despite intact activation; one structural refinement

Material Passport: ARS experiment-agent/run, collected2026-09-20; DEV519 only.
Smoke00230.920887s+pilot1771.452873s=1802.373760s charged once; exclude
parent1813.829383s. Earlier failedsmoke25.119405s already charged, not recounted.
Pilot completed666/noearlystop, all update/gradient/frozen-buffer/recipe gates pass.
Selected666mean78.4200385356, +.8670520231pprelease, -.1926782274ppR1seed42,
-.2890173410ppbest1337. T2V R1/5/10=77.456647/92.485549/96.146435;
V2T=79.383430/93.641618/95.568401. No promotion; best retained.

| Step | C15 mean R1 | GCN-R1 seed42 |
|---|---:|---:|
| 111 | 77.938343 | 77.938343 |
| 222 | 78.131021 | 78.420039 |
| 444 | 78.227360 | 78.516378 |
| 666 | 78.420039 | 78.612717 |

Peak14.506380GiB CUDA. At660 output-weight norms hand.763968/body.447525;
these establish neither beneficial activations nor a failure cause. Historical R1
comparison retains unresolved R2 training-replay caveat; no causal/SOTA claim.
Selectedsha93518a5d2781ef84b84a14402b069c136287bf34787dc1bece925c7950f719e6.
Decision: one structural contrast, uncentered second moment→within-part covariance;
same rank/capacity/LR/horizon. Motivation is E[zzT]=Cov(z)+mean(z)mean(z)T,
not evidence that the common component caused this decline. Native max retained.
If no incremental gain, deprioritize this family rather than blind LR/rank tuning.

## C14-R1 stronger bone LR: lower fused retrieval — defer current bone family

Material Passport: ARS experiment-agent/run, collected2026-09-20, DEV519 only.
Smoke28.0706725121s + pilot1702.2846388817s =1730.3553113937s charged once;
exclude parent1738.8956809044s. Both completed/exit0; pilot666/noearlystop.
Bothboneprojections/nativeGCN/fusion updated, frozenbuffers/source/base/data/order/
LR checks passed. No runtime failure observed; GPU process now absent.

| Step | C14-R1 meanR1 | Original C14 | GCN-R1 seed42 |
|---|---:|---:|---:|
| 111 | 77.745665 | 77.842004 | 77.938343 |
| 222 | 78.131021 | 78.323699 | 78.420039 |
| 444 | 77.938343 | 78.612717 | 78.516378 |
| 666 | 78.034682 | 78.516378 | 78.612717 |

Select222mean78.1310211946, T77.263969/V78.998073; +.5780346821pprelease,
but-.4816955684pporiginalC14/R1seed42 and-.5780346821ppglobalbest78.709056.
SelectedT2VR5/10=93.063584/96.339114;V2TR5/10=93.448940/95.375723.
Projectionnorm660hand.09670927/body.05990369 (~9.24x/8.45x original), so stronger
parameterlearning occurred but all four fusedDEV points are worse than original.
This does not prove overfitting or the cause; rejects this motivatedLR contrast.
Decision: defer current C14inputprojection family after originaltie+negative
refinement; no more LR/horizon/seed sweep. Keep bothselectedbests and GCN-R1.
Newbestsha0b64b36038dc07150845264631247ca83b96b82d6b7a527baea1f34a761a7cb8.
No SOTA/novelty/generalization claim; historicalR2 replay caveat remains.
Used35060.520296s/35892s, remaining831.479704s (~13.86min), no reservation/job.
Comparable~1700spilot exceeds remaining; request boundedallocation before next
representation-level candidate, not an automatically shortened or repeated run.
V2storage34.530397GiB<36cap, disk135GiBfree; no deletion required this collection.

## C14 bone descriptors: selected tie with GCN-R1, no new incumbent

Material Passport: ARS experiment-agent/run, collected2026-09-20, DEV519 only.
Smoke22.8513324261s + pilot1713.5337910652s =1736.3851234913s charged once;
parent1743.7924284935s excluded. Complete666updates/exit0/noearlystop; bothbone
projections, nativeGCN/fusion update and frozenbuffer gates pass; recipe/base/data/
order checks against historicalR1 pass. Peak14741346816bytes (~13.729GiB).

| Step | C14 fusion meanR1 | GCN-R1 seed42 | Difference pp |
|---|---:|---:|---:|
| 111 | 77.842004 | 77.938343 | -.096339 |
| 222 | 78.323699 | 78.420039 | -.096339 |
| 444 | 78.612717 | 78.516378 | +.096339 |
| 666 | 78.516378 | 78.612717 | -.096339 |

Select444mean78.6127167630, T77.842004/V79.383430. +1.059730pprelease,
exact selected-R1seed42 tie, -.096339ppglobalbest78.709056. Do not selectonly
the favorable444 contrast or attribute all releasegain to bonefeatures.
Atselected444 T2VR5/10=92.870906/96.339114,V2TR5/10=93.641618/95.375723.
Finalposemean63.776493 equals R1finalposemean; this does not establish identical score matrices.
Bestsha f92b981dd344bb584adaf46056869ac6c5e9347198de9aaef3bb1cb664a1dec3,
retained. No promotion or new SOTA/novelty claim.
Boneprojectionnorm atstep660 hand.01046447/body.00709206, vsstep110
.00396702/.00287898; active and small weights, not proof of small featureeffect
or undertraining. Register one boneLR1e-4 contrast to test stronger learning
of zero-init projection, keep all else fixed; no further blind sweep if no gain.
Used33330.164984s, remaining2561.835016s. C14-R1 reservation2430s.

## C13-R1 lower graph LR improves C13 but not GCN-R1 — defer topology family

Material Passport: ARS experiment-agent/run, collected2026-09-20, PHDEV519.
Smoke28.2797865868s + pilot1760.4784734249s =1788.7582600117s charged once;
parent1798.9297533035s excluded. Both exit0, pilot666updates, noearlystop;
allgraph/nativeGCN/fusion updates/frozenbuffer/source/base/data/order gates pass.

| Step | C13-R1 fusion meanR1 | Original C13 | GCN-R1 seed42 |
|---|---:|---:|---:|
| 111 | 78.131021 | 77.456647 | 77.938343 |
| 222 | 78.131021 | 77.360308 | 78.420039 |
| 444 | 77.745665 | 77.745665 | 78.516378 |
| 666 | 77.842004 | 77.552987 | 78.612717 |

Select111 (222tiesmeanwithdirectiontradeoff): T77.263969/V78.998073,
mean78.1310211946. +.5780346821pprelease, +.3853564547pporiginalC13,
but-.4816955684ppR1seed42 and-.5780346821ppglobalbest78.7090558767.
Finalpose64.354528 vs originalC1364.643545; less posegain accompaniedbetter
fusion thanC13, consistentwith—but not proof of—the slowerdrift hypothesis.
At111+.192678ppoverR1 is an earlypoint signal, not a selected-model win;
later pointsunderperform. No C13 promotion, extraLR/horizon/seed sweep or
claimthat alladaptivegraphs fail. Keep both C13bests and all GCN-R1 incumbents.
C13-R1bestsha44fba4feda483339afd972fa02ccbd1fc51d34ace0b729234ff97af91d6d9f58.
Next C14 explicitbonegeometry injection ratherthan learnedtopology; methodcard
registered separately. Remaining4298.220139s; used31593.779861s of35892s.

## C13 adaptive graph: pose gain but no increment over GCN-R1

Material Passport: ARS experiment-agent/run, collected2026-09-20, fullDEV519.
Smoke2updates27.963225s; pilot666updates1808.982913s; both exit0/completed,
parent completed1843.911873s. No earlystop. Charge children1836.946137s only.
All six graph deltas updated, native GCN/fusion updates and frozen-buffer gates
passed. Base/data/batch order and recipe match historical R1 except added graph.

| Step | C13 fusion meanR1 | R1 seed42 fusion | C13 pose meanR1 | R1 pose |
|---|---:|---:|---:|---:|
| 111 | 77.456647 | 77.938343 | 62.813102 | 61.368015 |
| 222 | 77.360308 | 78.420039 | 63.294798 | 62.620424 |
| 444 | 77.745665 | 78.516378 | 63.583815 | 63.487476 |
| 666 | 77.552987 | 78.612717 | 64.643545 | 63.776493 |

C13 selects444, T76.878613/V78.612717, +.192678pp release but
-.867052pp selected R1seed42 and -.963391pp global incumbent78.709056.
Finalfusion equals release mean; finalpose+4.142582pp release and+.867052pp
R1pose. Better standalone pose is not better fused retrieval. No causal claim
that fusion lag/LR explains this; R2 replay discrepancy remains a limitation.
Best retained: seds-adaptive-graph-001/best.pt, sha256
cb35f5e98b3f009d4223c7aead5484a15500b8723bd9c5e32163c105da841817.
Decision: no promotion; one motivated C13-R1 graphLR1e-5 contrast from release,
all else fixed. Test slower topology drift, not guaranteed improvement. No further
blind LR/seed/horizon sweep if it gives no useful fused gain; keep GCN-R1 bests.
Budget used29805.021601s, remaining6086.978399s; reserve2430s for refinement.
Pilot peak17.306579GiB, source/run reports preserved, no TEST consulted.

## GCN-R2 completed training, failed exact pre-freeze comparison; no new lead

Material Passport: ARS experiment-agent/run, collected2026-09-20, fullDEV519.
Smoke4steps29.741496s and pilot666steps946.888958s both exit0/completed; no
earlystop. Parent chain FAILED its post-training exact pre-freeze metric gate.
Do not overwrite that failure or label this an exact matched causal result.

| R2 step | T2V R1/5/10 | V2T R1/5/10 | Mean R1 |
|---|---|---|---|
| 111 | 76.878613/92.870906/96.146435 | 78.998073/93.063584/95.375723 | 77.938343 |
| 222 selected, before freeze | 77.842004/93.448940/96.339114 | 78.998073/93.448940/95.375723 | 78.420039 |
| 444 | 77.842004/93.448940/96.339114 | 78.420039/93.448940/95.375723 | 78.131021 |
| 666 | 77.842004/93.448940/96.339114 | 78.227360/93.448940/95.375723 | 78.034682 |

Selected78.4200385356 is -.1926782274pp vs R1 seed42 selected78.612717,
-.2890173410pp vs globalbest78.709056, +.8670520231pp release. Final is
-.5780346821pp vs R1 final. No score gain after the freeze; defer this recipe,
retain R1 best and all seeds. Not a universal rejection of late freezing.

Implementation checks passed: transition223 kept fusion optimizer step222,
no reset; only fusion changed after transition, GCN weights and frozen buffers
remain exact through final. Same native config except output path, same base,
data/batch-order and optimizer groups. Scoped comparison of saved source shows
the freeze branch executes only after222. Nevertheless, pre-freeze equivalence
was NOT exact: R@1/5/10 and MedianR match at111/222, but T2V MeanR at111 and
V2T MeanR at222 each differ +.0019267823. Saved fusion scores differ maxabs
.006521225/.006296158, meanabs.0006513805/.0007197066. These are actual score
differences, not just JSON rounding; do not weaken the gate after seeing results.
Logged loss/gradient agree at steps1/2 and differ by10, before intervention.
Root cause remains unresolved; nondeterminism is not established. Native source
sets cuDNN deterministic=True/benchmark=False, not proof all operations are
deterministic. No broad baseline replay or automatic efficacy rerun undertaken.

Keep original child completed/parent failed reports and checkpoint sha
8c08911a8de6e929f09f4e9d82ef5c4ae953d23ab01500f9852551fb28c289b1.
Charge children976.6304543018s once, not parent988.221927s; used27968.075464s,
remaining723.924536s (~12m04s) of28692s. Disk136GiBfree,V2used33.759604GiB,
no deletion needed. No active job. A repeat R2 pilot alone observed947s exceeds
remaining allocation; do not auto-retry, relax checks or increase budget.
Current campaign lead remains GCN-R1 selected3seed mean78.612717, best78.709056;
novelty, independent generalization and SOTA remain unestablished.

## GCN-R1 three-seed results — selected gain repeats, late training is mixed

Material Passport: ARS experiment-agent/run, collected2026-09-20, fullDEV519.
Both new seeds completed666updates, no earlystop; expected updates, fixed buffers,
base/data and first-epoch order checks pass. No TEST used.

| Seed | Selected step | Selected meanR1 | Gain vs1epoch same seed | Fixed666 meanR1 |
|---|---|---|---|---|
| 42 | 666 | 78.612717 | +.578035 | 78.612717 |
| 1337 | 222 | 78.709056 | +.192678 | 78.420039 |
| 2026 | 222 | 78.516378 | +.385356 | 78.034682 |

Selected mean78.6127167630, sampleSD.0963391137; +1.0597302505pp release,
+ .3853564547pp vs1epoch mean78.227360. These are descriptive3seed statistics,
not CI/significance or independent confirmation. Fixed666 mean78.3558124599,
sampleSD.2943208539, only+.1284521516pp vs1epoch endpoints; seeds1337/2026
fixed666 are each -.096339pp vs their1epoch endpoints. Do not hide late decline.
Fixed222 under3epoch schedule mean78.5484906872, sampleSD.1471604269; extra
training is not uniformly helpful. Schedules differ from step1 and DEV repeatedly
selects checkpoints; no claim of causal extra-epoch benefit alone.

| Seed/checkpoint | T2V R1/5/10 | V2T R1/5/10 |
|---|---|---|
| 1337 selected222 | 77.456647/93.256262/96.146435 | 79.961464/93.256262/95.183044 |
| 1337 final666 | 77.842004/92.485549/95.953757 | 78.998073/93.448940/95.568401 |
| 2026 selected222 | 77.842004/93.063584/96.146435 | 79.190751/93.063584/95.375723 |
| 2026 final666 | 77.456647/92.678227/96.146435 | 78.612717/93.448940/95.568401 |

New single-model best1337step222=78.7090558767 (+1.1560693642pp release),
sha e7f0baa47343e25a28a7c38d02af65a412d6b8e53c3b8fabd18026ba625be095.
Keep all3 new seeds and all1epoch/C04 leads. Seed42 matched3epoch fusion gain
is validated separately; other historical fusion controls remain contextual,
not silently recertified by matching seed42. Novelty/SOTA/generalization unresolved.
Charge1698.5940096378+1719.8979325294=3418.4919421673s once, not parent3430.268857s.
Used26991.445009s, remaining1700.554991s of28692s. Disk136GiBfree,V2used33.421640GiB.
Next GCN-R2 freezes GCN after222, continues fusion with intact moments/schedule,
seed42 matched R1 control; hypothesis informed by late decline, not a claimed fix.

## GCN-R1 matched three-epoch fusion control supports the seed42 gain

Material Passport: ARS experiment-agent/run, collected2026-09-20, fullDEV519.
Control completed666updates646.191070s, no earlystop, only fusion updated,
frozen buffers fixed. Base/data/666batch hash/LR/schedule/selector matched.
Selected initialization77.5529865125 vs GCN-R1 selected/final78.6127167630:
paired gain+1.0597302505pp. Control final666 ties its initial mean, different T/V.

| Control step | T2V R1/5/10 | V2T R1/5/10 | Mean R1 |
|---|---|---|---|
| 111 | 76.493256/92.870906/96.146435 | 78.420039/92.870906/95.375723 | 77.456647 |
| 222 | 76.685934/92.870906/96.146435 | 78.034682/93.256262/95.375723 | 77.360308 |
| 444 | 76.685934/92.870906/96.146435 | 78.227360/93.063584/95.375723 | 77.456647 |
| 666 | 76.685934/92.870906/96.146435 | 78.420039/92.870906/95.375723 | 77.552987 |

Observed metrics also match the historical C04 fusion control at every recorded
step; retain this fact and avoid more unchanged control replays. Current-path
comparison supports native GCN adaptation at3epochs for this seed, not independent
generalization or novel architecture. One seed cannot establish robustness.
Keep GCN-R1 best and prior leads; replicate fixed recipe on1337/2026 next.
No TEST selection. Report all seeds and both selected/fixed666 outcomes, not just
the best seed. Same-seed1epoch GCN endpoints are available for horizon comparison.
Charge child646.1910696030s once, not parent651.645420s. Used23572.953067s,
remaining5119.046933s. Reserve4260s for2x2100s sequential replications;
unreserved859.046933s. Disk136GiBfree,V2used33.023229GiB before scoped cleanup.

## GCN-R1 three-epoch refinement establishes a new single-seed DEV lead

Material Passport: ARS experiment-agent/run, collected2026-09-20, actual
fullDEV519. Completed666updates1697.489469s, no early stop; all update/buffer,
base/data and first-epoch order checks passed. No TEST used.

| Step | T2V R1/5/10 | V2T R1/5/10 | Mean R1 |
|---|---|---|---|
| 111 | 76.878613/92.870906/96.146435 | 78.998073/93.063584/95.375723 | 77.938343 |
| 222 | 77.842004/93.448940/96.339114 | 78.998073/93.448940/95.375723 | 78.420039 |
| 444 | 77.842004/93.063584/96.339114 | 79.190751/93.641618/95.568401 | 78.516378 |
| 666 selected | 77.842004/92.678227/96.339114 | 79.383430/93.448940/95.568401 | 78.612717 |

Selected/final78.6127167630: +1.0597302505pp release, +.5780346821pp same-seed
one-epoch GCN, +.0963391137pp previous global best(seed1337). Both R1 directions
improve vs release; T2V R5 drops .192678pp. At common111/222, gains over the
one-epoch run are +.192678/.385356pp. Thus longer schedule already matters before
extra epochs; do not attribute all gain to late updates. Extra evaluation and
repeated DEV exposure disclosed. Cross-seed best gap is not causal evidence.
Promote as provisional checkpoint, not a novel mechanism, SOTA or independent
confirmation. Best sha08a95b5f54e6f5eabc8bd7f0f98aa2e3c72c44d24376ba038c397f78c14346e6;
retain all one-epoch GCN seeds and C04. New recipe still needs replication.

Historical C04 fusion-only3epoch control shares base/data/batch hash but uses
older runner/config, no inactive masked-control module or explicit trainable-only
optimizer partition. Selected77.552987 is contextual evidence, not certified as
an exact control. Run one fusion-only3epoch control in current GCN path; no
baseline replay, unchanged evaluator/input reused. See METHOD_GCN_HORIZON_R1.md.
Charge1697.4894688129s once (not parent1702.684896s); used22926.761997s,
remaining5765.238003s of28692s. Reserve830s next; unreserved4935.238003s.
Disk136GiBfree,V2used32.909513GiB before retiring deferred C11 last.pt only.

## C12 completed — temporal adaptation ties the matched GCN control

Material Passport: ARS experiment-agent/run, collected2026-09-20 after user
completion notification. Native fullDEV519, repeated selection, no TEST used.
Smoke2steps and pilot222steps completed; temporal weights actually updated,
frozen buffers stayed fixed, expected LR/update checks passed; no early stop.

| C12 checkpoint | T2V R1/5/10 | V2T R1/5/10 | Mean R1 |
|---|---|---|---|
| 111 | 77.456647/92.870906/96.146435 | 78.034682/93.256262/95.375723 | 77.745665 |
| 222 selected | 77.456647/93.063584/96.146435 | 78.612717/93.448940/95.375723 | 78.034682 |

Selected222mean78.0346820809 equals matched clean GCN seed42 exactly: 0pp
increment. At111 primary mean also ties that control, but V2T is -.578035pp
below reference and fails the unchanged -.5pp direction guardrail. Final passes.
R5 directional tradeoffs are not grounds to switch the selection metric.
Vs release +.481696pp, vs global incumbent1337 -.481696pp; the latter comparison
is cross-seed and cannot identify a causal method effect. Defer current C12
recipe, not the entire family. Retain clean GCN best78.516378 and all3 seeds.
No novelty, SOTA or independent generalization claim.

Pilot best sha256 c35b95dde34c34de3faee2c618492d07e2fce34cabdce01a066b4d82d2c78c1e.
Charge smoke26.8310272694+pilot607.5101563931=634.3411836624s once, excluding
parent642.965369s. V2 used21229.272529s, remaining262.727471s of21492s.
Disk137GiBfree, V2artifacts32.130671GiB <36GiB cap; no deletion needed.
No new job launched: remaining~263s cannot cover a comparable~608s pilot plus
smoke. Additional bounded local training allocation requires user direction.

## Matched fusion-only ablation supports GCN contribution across three seeds

Material Passport: ARS experiment-agent/run,2026-09-20, paired fullDEV519
results, descriptive only. All three controls completed222updates, same seed,
base/data/order,1epoch schedule and selector as corresponding clean GCN runs.
Only fusion changed; frozen buffers unchanged; exact LR groups passed.

| Seed | Fusion-only selected | GCN+fusion selected | Paired gain | Fixed222 gain |
|---|---|---|---|---|
| 42 | 77.552987 (init) | 78.034682 | +.481696 | +.481696 |
| 1337 | 77.649326 (222) | 78.516378 | +.867052 | +.867052 |
| 2026 | 77.552987 (init) | 78.131021 | +.578035 | +.770713 |

Fusion-only selected mean77.5850995504 vs GCN78.2273603083; paired mean
+ .6422607579pp, sampleSD.2005458574 across3 seeds (not CI/significance).
This supports incremental benefit of native GCN weight adaptation under this
recipe. It is not masked-reconstruction benefit, generalization or novelty/SOTA.

| Fusion-only seed/step | T2V R1/5/10 | V2T R1/5/10 |
|---|---|---|
| 42/111 | 76.685934/92.870906/96.146435 | 78.034682/92.870906/95.375723 |
| 42/222 | 76.685934/92.870906/96.146435 | 78.420039/92.870906/95.375723 |
| 1337/111 | 76.493256/92.870906/96.146435 | 78.227360/92.870906/95.375723 |
| 1337/222 | 76.493256/93.063584/96.146435 | 78.805395/92.870906/95.375723 |
| 2026/111 | 76.493256/92.870906/96.146435 | 78.420039/92.870906/95.375723 |
| 2026/222 | 76.493256/92.870906/96.146435 | 78.227360/92.870906/95.375723 |

Init retained for42/2026 under unchanged selector; do not report final instead
to inflate selected-model improvement. GCN wins fixed111 for all3 as well.
Charge214.254928+220.942733+243.751525=678.949186s; used20594.931345s,
remaining897.068655s. Disk63GiBfree,V2used29.936GiB. Keep current best78.516378.
Next C12 adapts native clip-local temporal convolution with GCN using seed42,
matched recipe; reserve880s including smoke. No automatic budget increase.

## Clean GCN replication: all three seeds improve over release

Material Passport: ARS experiment-agent/run, collected2026-09-20, PH adapted
DEV519, descriptive three-training-seed results, repeated DEV selection disclosed.

| Seed | Selected step | Selected meanR1 | Gain vs release | Final222 meanR1 |
|---|---|---|---|---|
| 42 | 222 | 78.034682 | +.481696 | 78.034682 |
| 1337 | 111 | 78.516378 | +.963391 | 78.516378 |
| 2026 | 111 | 78.131021 | +.578035 | 78.131021 |

Selected and fixed222 mean across seeds78.2273603083, sample SD.2548893363;
mean gain+.6743737958pp release. SD describes only3 training seeds, not a CI.
Selected/final equality in mean does not imply identical predictions/directions.

| Seed/checkpoint | T2V R1/5/10 | V2T R1/5/10 |
|---|---|---|
| 1337 selected111 | 76.878613/93.063584/96.146435 | 80.154143/93.063584/95.375723 |
| 1337 final222 | 77.263969/93.063584/96.146435 | 79.768786/93.256262/95.375723 |
| 2026 selected111 | 77.071291/93.063584/96.146435 | 79.190751/93.448940/95.375723 |
| 2026 final222 | 77.456647/93.063584/96.146435 | 78.805395/93.063584/95.375723 |

Seed42 full metrics in C08 table below. Both new runs completed222updates with
no early stop, expected updates/frozen buffers/LRs passed. Promote clean recipe
as repeated-seed DEV lead; new single-model best seed1337step111 at
`artifacts/slret_goal_v2/seds-gcn-clean-seed1337-001/best.pt`, sha256
c699104f8c21919fef08a3b03d03553175c11a32c00af0b7ce2427925ae62543.
Preserve all3 seeds and C04. This is an engineering adaptation, not evidence of
novelty, independent generalization or SOTA. Historical C04/fusion-only3epoch
runs differ in horizon/schedule, so GCN-specific attribution still needs control.
Next: matched1epoch fusion-only parameter ablation for same3 seeds.
Charge582.724964+576.091494=1158.816458s once. Used19915.982159s,
remaining1576.017841s; pool701.401446 + old382.616394 + unallocated492 consolidated,
total unchanged; reserve1410s for3x450s bounded ablation. Disk67GiBfree, no deletion.

## C11 completed — no incremental gain from this LoRA composition

Material Passport: ARS experiment-agent/run,2026-09-19, actual fullDEV519 outputs.
Completed222updates; selected222mean77.9383429672, +.3853564547pp release,
-.0963391137pp matched clean GCN+fusion78.034682. Historical C04best tie in
mean, different direction tradeoff and horizon. Batch order matches GCN control.

| C11 checkpoint | T2V R1/5/10 | V2T R1/5/10 | Mean R1 |
|---|---|---|---|
| 111 | 77.071291/92.870906/96.146435 | 78.612717/92.870906/95.375723 | 77.842004 |
| 222 selected | 77.456647/93.448940/96.339114 | 78.420039/93.063584/95.375723 | 77.938343 |

At111 C11 +.096339pp vs same-step GCN; at222 -.096339pp. Both checkpoints pass
direction guardrail. T2V R5/R10 increased but primary meanR1 did not beat control;
do not change metric post hoc. Actual GCN/LoRA tensors updated and frozen buffers
stayed unchanged; no implementation failure. Defer current simultaneous recipe,
preserve best7be31df0f7b4055366814ba9d0034af421a6c1df304c421d160ae58566906ebc.
No assertion that composition can never help. Current best remains clean GCN.
Charge smoke27.374265+pilot602.012399=629.386665s; used18757.165701,
remaining2734.834299s. Allocate1200s internally from unallocated1692s to current
pool660.217904→1860.217904; reserve1860 for clean GCN seeds1337/2026.
Next is actual training-seed replication of positive adaptation, not baseline
replay. Fixed config/selector and all-seed reporting; novelty/SOTA unresolved.

## C08 pair002 complete — clean GCN control becomes provisional best

Material Passport: ARS experiment-agent/run, actual artifacts,2026-09-19;
exploratory DEV selection, no independent/SOTA/novelty claim.

| Run/checkpoint | T2V R1/5/10 | V2T R1/5/10 | Mean R1 |
|---|---|---|---|
| Masked reconstruction111 (selected) | 77.071291/92.678227/96.146435 | 78.420039/93.256262/95.568401 | 77.745665 |
| Masked reconstruction222 | 77.263969/92.485549/96.146435 | 77.842004/93.063584/95.568401 | 77.552987 |
| Clean GCN+fusion111 | 76.878613/92.870906/96.146435 | 78.612717/92.870906/95.375723 | 77.745665 |
| Clean GCN+fusion222 (selected) | 77.456647/93.256262/96.146435 | 78.612717/93.063584/95.375723 | **78.034682** |

Both completed222updates, same batch-order hash3286276406a0dc50d332c97ffe9119abed089c1dcc3804083a2b1f4e4bd7b015.
Treatment-selected minus control-selected -.289017pp; fixed111 tie, fixed222
-.481696pp. Treatment finalV2T fails guardrail. Reconstruction .066876→.012372,
weighted .003344→.000619, auxiliary gradient nonzero: optimization happened,
but no retrieval benefit vs clean continuation. Defer this C08 recipe.
Control +.4816955684pp vs release and+.0963391137pp vs previousC04best.
This tiny C04 margin is one net additional hit over1038 directional queries;
keep checkpoint as provisional incumbent, not proof of superiority across seeds.
No attribution of control gain to masking/decoder (both inactive in control).
Selected checkpoint: `artifacts/slret_goal_v2/seds-masked-control-002/best.pt`,
sha256 beffcbd667e09ff29280db4170c3265a3692145557894741adce59b3cdda8fd7.
Native representation adaptation is an engineering result; novelty unresolved.
Charge smoke27.841263+treatment618.296536+control591.521164=1237.658962s once.
Used18127.779036s, remaining3364.220964s,currentpool1289.604569s. Disk73GiBfree.
Next C11 combines clean GCN adaptation and known visual LoRA at same222 horizon;
no masked objective, no score ensemble, no extra data. Card METHOD_GCN_LORA_C11.md.

## C08 attempt001: implementation failure, no efficacy result

Smoke failed in first backward (0 optimizer updates) with DDP decoder parameter
marked ready twice. Initial inference parity passed, but no training completed.
Child20.801845s charged once; remaining4601.879926s. User authorized repair and
rerun002: return auxiliary through forward, single backward, DDP regression test.
Original logs/source retained. C08 and matched-control metrics remain pending.

## C09 frozen-fusion refinement completed — defer additive geometry

Both XY and XYZ select step64 meanR1=77.7456647399, T2V=77.0712909441,
V2T=78.4200385356. Both R5: T/V93.0635838150; R10:T96.1464354528/V95.3757225434.
Delta release +.1926782274pp; prior C09 -.0963391137pp; C04 -.1926782274pp.
XYZ-minus-XY selected gain0pp. Final128: XYmean77.2639691715 (T=V77.2639691715);
XYZmean77.3603082852 (T77.2639691715/V77.4566473988). Both fail V2T guardrail.
Only geometry updated; full fusion state remained unchanged, frozen buffers,
base-dependent checkpoint roundtrip and zero-init DEV score parity all passed.
Thus fusion-drift-only explanation is not supported by this intervention;
estimated Z gain remains unestablished. This is not evidence 3D can never help.
Defer current family; retain cache/weights; proceed to C08 representation objective.
Primary reports: artifacts/slret_goal_v2/seds-geometry-frozen-{xy,xyz}-001/run.json.
Charge children341.118149s once; used16869.318229, remaining4622.681771s,
currentpool2548.065376s. Repeated DEV selection disclosed; TEST not consulted.

## C09 matched TRAIN512 fusion-only control

Completed128updates165.271246s,geometry stayedzero,onlyfusiontrainableLR1e-5.
Verified exact batch-order hash and geometry_data match both prior XY/XYZ runs.
Selector retains initialization77.552987. At64/128mean77.167630,
T2VR1=76.493256,V2TR1=77.842004. Step64R5/R10:
T92.870906/96.146435,V93.063584/95.375723; step128V R5=92.870906.
XY/XYZ selected77.842004 are+.289017vsselectedcontrol and+.674374 atfixed64.
Atfixed128 XY/XYZ77.360308 are+.192678vscontrol77.167630 but belowrelease.
This is one-seed evidence for the trainable added-geometry pathway vsfusiononly,
not proof of explicitZbenefit,3Daccuracy,generalization orSOTA. Addedcapacity and
geometry training co-vary, so input-shuffle/capacitycontrol could further test
information attribution after a strongerlead. Currentincumbent remainsC04.

Decision: C09refinement1 freezeoriginalfusion and traingeometryonly, sameXY/XYZ
pairrecipe. Rationale: fusion-onlyV2Tfalls and jointadaptationlateV2Tfalls;
freezingfusion may reduce drift, but interaction/noisecausality is unproven.
NoLRincrease or3Dre-extraction. Preserve original2D/RGB and allcurrentC09weights.
Chargecontrol165.271246s once: used16528.200080s,remaining4963.799920s;
currentpool2889.183525s,reserve2100s fornewsmoke/pair. No totaltop-up.

Primary: PH adapted full DEV519 mean(T2V R@1,V2T R@1), percentages.
Fixed reference77.5529865125; inherited incumbent77.6493256262.
All DEV gains are exploratory selection evidence;
PH/CSL TEST were exposed in V1 and are not used for V2 tuning.

| Run / step | T2V R1/R5/R10 | V2T R1/R5/R10 | MeanR1 | Delta reference |
|---|---|---|---:|---:|
| C01 aux.25 /111 |75.915222/93.063584/95.953757|77.649326/93.448940/95.568401|76.782274|−.770713|
| C01 aux.25 /222 |76.685934/92.870906/95.761079|78.227360/93.641618/95.761079|77.456647|−.096339|

Selector retained initialization77.552987; no gain, incumbent remains77.649326.
222 updates completed620.507372s; smoke13.014711s. Decision: try C02 staged
adaptation next; C01 family is not globally disproven. No TEST used.

## C02 staged fusion/upper-block adaptation

666updates completed614.156384s; stage-transition smoke13.138077s. Same pretrained
reference; no C01 loss reduction. DEV meanR1 steps111/222/444/666:
77.360308/77.552987/77.456647/77.360308. Selectorinitialization; no improvement.
FinalT2V76.685934/92.870906/96.146435, V2T78.034682/93.063584/95.375723
(R1/R5/R10). FinalV2T violates−.5ppguardrail; no score trade-off concealed.
Intermediate full metrics in `artifacts/slret_goal_v2/seds-staged-001/run.json`.
Decision: deprioritize this fixed staged recipe; move to C03 representation
intervention. No global conclusion that selective fine-tuning cannot work.

## C03 cross-articulator product interaction

222updates completed690.346252s, smoke23.400935s. New196608-parameter module
activated; zero-init fullDEV parity passed. DEV111mean77.360308,
DEV222mean77.071291 (−.481696vsreference,−.578035vsincumbent), selectorinitialization.
Final T2V76.878613/93.063584/96.146435, V2T77.263969/93.063584/96.146435
(R1/R5/R10); V2T R1−1.348748pp violates guardrail despite better mean ranks.
Decision defer this recipe; test C04 constrained adaptation next, not a proof
that hand–body information is useless. No test-based hypothesis/selection.

## C04 LoRA upper visual attention + existing fusion — exploratory lead

666updates completed580.021394s; smoke24.617811s. DEV111/222/444/666 meanR1:
77.360308/77.552987/77.938343/77.552987. Selectorstep444; +.385356vsrelease,
+.289017vsinheritedincumbent. SelectedT2V77.071291/92.870906/96.146435,
V2T78.805395/93.063584/95.568401 (R1/R5/R10). BothR1directions up, R5 unchanged,
V2TR10+.192678; no selectedR5/R10regression. Finalmean tiesinitial and finalV2T
guardrail fails; do not cherry-pick endpoint or call a one-seed selectedgain
confirmation. Best model retained; incumbent provisionally77.938343.
Next matchedfusion-only666-stepcontrol; ordinary LoRA not new algorithm/SOTA.

C04 matched fusion-only control COMPLETE584.461611s,666updates. Selectedstep0
mean77.552987; LoRA-selected444 exceeds selectedcontrol by.385356pp. Control
DEV111/222/444/666means77.456647/77.360308/77.456647/77.552987. FinalT2V
76.685934/92.870906/96.146435,V2T78.420039/92.870906/95.375723 (R1/R5/R10).
The seed42 lead survives this absent-LoRA control, but remains exploratory.
Next paired continuation seeds1337/2026, no hyperparameter changes.

### Three continuation seeds complete

| Seed | LoRA selectedmeanR1 | Fusioncontrol selectedmeanR1 | Paired delta pp |
|---|---:|---:|---:|
|42|77.938343|77.552987|+.385356|
|1337|77.745665|77.745665|0 (floating-point roundoff ignored)|
|2026|77.842004|77.552987|+.289017|

Meanpaired+.224791±.200546pp(sampleSD),2positive/1tie; not a seed-population CI.
LoRAselectedmean averagedoverseeds77.842004 vs release77.552987 (+.289017pp).
Completeinitial/selected/endpoint R1/R5/R10/ranks for everyseed andbotharms:
`artifacts/slret_goal_v2/lora-pairs-summary-001/metrics.csv`. This remains
historicallyexposedDEVselection from a shared pretrainedinitialization; not
freshgeneralization/SOTA. Fouradditionaljobs cost2387.945442s (driver not
doublecounted). Keep lead and test batch128 refinement for more real negatives.

### Batch128 refinement (C04b)

Completed168updates/3passes in564.681579s, exit0; selected168mean77.745665.
DEV28/56/112/168means77.552987/77.552987/77.649326/77.745665.
Selected T2V77.071291/92.870906/96.146435 and V2T78.420039/92.870906/95.375723
(R1/R5/R10). Mean+.192678pp vsrelease, -.192678pp vscurrentB32incumbent.
V2TR1 andR5 both-.192678pp vsrelease; do not claim uniform improvement.
PeakCUDA9.190GiB. Report `artifacts/slret_goal_v2/seds-lora-b128-001/run.json`;
selectedbest SHA c5e03f2d3192bdeac2b21bf0f22104ef9493b9ad0eeacb9a4da01eb16d3e32dd.
Runner delta_incumbent still refers to V1control77.649326; comparison above uses
actualcurrent77.938343. No promotion; matchedB128fusion control next before
attributing gain to LoRA. Three passes mean fewerupdates than B32 (168vs666).

B128fusion control COMPLETE168steps865.149460s,exit0; batchorder matchesLoRA.
DEV28/56/112/168 mean77.649326/77.360308/77.552987/77.649326.
Reportedselected168 T2V76.685934/92.870906/96.146435,
V2T78.612717/93.063584/95.375723 (R1/R5/R10).
LoRA-selected minuscontrol +.096339pp; belowB32lead, no batch-size benefit shown.
Tie caveat: step28 and168 differ by1.4e-14; rawselector took168. Correctearliest
tie is28, T2V76.493256/92.870906/96.146435,V2T78.805395/93.063584/95.375723.
Meancontrast is unchanged; oldcheckpoint/report retained, no retrospective rewrite.
FromC05 onward selector improvement requires >1e-8pp to preserve earliest ties.
Decision: deferB128recipe, retainB32incumbent, moveC05lossregularization.
V2compute throughcontrol7009.523341s; unused190.476659s oftranche1 retained.

## C05 label smoothing .05 — failed recipe

Smoke11.530890s passed finitegradients/all4groupsupdated; pilot647.172093s
completed222steps/exit0, no earlykill. DEV111mean58.574181,final60.597303,
final-16.955684ppvsrelease/-17.341040ppvscurrentincumbent77.938343.
FinalT2V61.657033/85.356455/90.751445,V2T59.537572/81.888247/87.475915
(R1/R5/R10). Selectorinitialization77.552987: no gain. PeakCUDA20.420GiB.
TRAINloss8.724178→2.179208; reducing newobjective didnot preserve retrieval.
Cause not isolated: this is epsilon.05 with full-model continuation, not proof
that all smoothing or all lossregularization fails. Drop this recipe; no repeat.
Modelbest77.938343 untouched. C04c depth3LoRA next, no C05combination.
Charge658.702983s totranche2, remaining2941.297017s; allV2used7668.226324s.

## C04c LoRA three visual blocks

Completed666steps615.138660s; smoke23.504938s; no degradation stop triggered.
DEV111/222/444/666means77.552987/77.552987/77.842004/77.745665.
Selected444 T2V77.263969/93.063584/96.146435,V2T78.420039/93.063584/95.375723.
Mean+.289017ppvsrelease andfusioncontrol, -.096339ppvsdepth1incumbent77.938343.
T2VR1+.770713ppvsrelease butV2TR1-.192678pp; no uniform directional gain.
Final666 T2V77.263969/93.063584/96.146435,V2T78.227360/93.063584/95.375723.
PeakCUDA3.038GiB. SelectedSHA06164792e393f36d367fca6a537081d66246ca610074c9f1b417156313a49772.
Deferdepth3recipe; retain depth1incumbent. Sharedtext adaptation next, not moredepth.
Charge638.643599s; V2total8306.869923s,tranche2remaining2302.653418s.

## C04d shared text LoRA

Completed666steps654.159093s; smoke18.398908s; noearlystop, all6Bfactorsactive.
DEV111/222/444/666 mean77.552987/77.649326/77.842004/77.649326.
Selected444 T2V77.071291/92.870906/96.146435,V2T78.612717/93.063584/95.568401.
Mean+.289017ppvsrelease, -.096339ppvsvisualdepth1incumbent; no incremental gain.
Final666 T2V77.071291/92.870906/96.146435,V2T78.227360/93.256262/95.568401.
PeakCUDA3.072GiB; selectedSHA0d3986cc7f08b72342c89c3d7afd0c505868f1d6a301a008ff0b30bb98220ec8.
Defertextrecipe, retainincumbent77.938343; threeC04refinements exhaustedthisround.
Cost672.558001s; totalV2 8979.427924s; tranche2remaining1630.095417s.
MoveC03b frozenencoder interaction, no moreLoRAtuning inthisround.

## C03b frozen encoder interaction

Completed666steps1414.744518s; smoke23.211164s; noearlystop.
DEV111/222/444/666 mean77.552987/77.360308/77.456647/77.552987.
Selectedinitialization77.552987, no gainvsrelease/fusioncontrol; -.385356vsincumbent.
FinalT2V76.685934/92.870906/96.146435,V2T78.420039/92.870906/95.375723.
PeakCUDA3.992GiB; moduleactivated/frozenchecks passed, but no retrievalbenefit.
Deferthisrecipe (not all spatialrelations), nextC06 temporal differences.
Cost1437.955682s; totalV2 10417.383606s; remaininglocal11074.616394s.
Bookkeeping correction: earlier unallocated10882.476659s doublecounted unused
tranche1 190.476659s; actualunallocatedbeforetranche3 10692s. No compute added.

## C06 — intentionally stopped on user request

User explicitly requested abandoning C06; verified torchrun PID1983045 received
SIGTERM. Launcher/worker/parent PIDs are absent. Last logged update419/666;
last completed DEV222mean77.263969 (-.289017ppvsrelease, -.674374vsincumbent).
DEV111mean77.456647. Selector still initialization; no candidate promoted.
This is an incomplete pilot, not proof a completed666-step run would lose.
Raw chain reports failed/exit1 and child reports stale running after SIGTERM;
the separate `c06-temporal-smoke-pilot-001/user_stop.json` records actual cause.
No retries, extension or C06 strength sweep. Preserve all artifacts.
Conservatively charge whole chain388.311097s ONCE (includes smoke22.328764s).
V2used10805.694703s; total remaining10686.305297s. Tranche3remaining3211.688903s,
earlier residual382.616394s, unallocated7092s. Incumbent77.938343 unchanged.

## C07 — completed, but fusion LR deviated from registration

Smoke24.491619s and pilot569.699737s completed222updates, exit0; no early stop.
DEV111mean77.071291; DEV222mean77.167630 (-.385356ppvsrelease,
-.770713ppvsincumbent). Selected initialization77.552987, incumbent unchanged.
Final T2V77.071291/92.870906/96.339114; V2T77.263969/93.063584/95.953757
(R1/R5/R10). V2TR1 fell1.348748pp despite gains atR10; no promotion.
Peak9.269826GiB; zero-init score parity/new-module activation/frozen checks passed.

Decision-changing recipe mismatch: native prep_optimizer gives fusion groups no
explicit LR, so BertAdam inherits args.sign_lr1e-4 (coef_lr1). Registered fusion
LR was1e-5; args.lr only controls CLIP groups. Joint LR1e-4 was as intended.
Thus these scores describe actual higher-fusion-LR C07, not the intended recipe.
Mismatch is source/config-established; it is NOT yet proven to cause degradation.
Preserve raw reports/checkpoints. Repair once via explicit named optimizer groups,
CPU update test and smoke assertion; report actual groups in new run. No other
architecture/loss/horizon/seed change, warm-start release, NOT failed endpoint.
This is a bounded implementation repair, not an unconstrained LR sweep.

Charge child times594.191356s once (chain607.799426 includes orchestration).
V2used11399.886059s, remaining10092.113941s, tranche3remaining2617.497547s.
Reserve repaired C07 chain2100s; no budget increase. User-suggested Hand4Whole++
Pose3D becomes preferred next hypothesis if correctly configured C07 has no lead.

## C07 corrected LR — small DEV gain, below incumbent

Smoke25.532193s + pilot559.612542s completed222steps/exit0; no early stop.
Actual optimizer groups joint1e-4/fusion1e-5 logged and smoke-asserted.
Selected111mean77.745665: +.192678ppvsrelease, -.192678ppvsincumbent77.938343.
T2V77.071291/92.870906/96.146435, V2T78.420039/92.870906/95.375723.
Final222mean77.552987, T2V77.071291/V2T78.034682; finalguardrail fails.
Keep best.pt SHA767d01af6eb38e9a17797525f936736968e213987efea5b0a461f4a23dfc07d7.
Not promoted; no matched control/replication, so no attributable mechanism claim.
Move to user-prioritized added Pose3D, explicitly retaining original pose2D/RGB.
Charge585.144735s childtimes once: V2used11985.030794s; remaining9506.969206s;
tranche3remaining2032.352812s, earlierresidual382.616394s, unallocated7092s.

## H4W runtime / geometry samples — feasibility only

## C09 first matched XY/XYZ pair — gain, no explicit-depth lead

Completed chain348.783473s; smoke21.503825s, XY154.318401s,
XYZ154.329949s. Charge childtimes330.152175s once, not parent+children.
Both128updates, sameTRAIN512/DEV519 and batch order, no earlystop.
FullDEVzero-init parity, geometry output updates, frozen parameter/buffer
preservation, compact state roundtrip and explicit learning rates passed.

| Model/selected step | T2V R1/R5/R10 | V2T R1/R5/R10 | meanR1 |
|---|---|---|---|
| Release |76.493256 /92.870906 /96.146435|78.612717 /93.063584 /95.375723|77.552987|
| C09 XY64 |77.071291 /93.256262 /96.146435|78.612717 /93.256262 /95.375723|77.842004|
| C09 XYZ64 |77.071291 /93.256262 /96.146435|78.612717 /93.256262 /95.375723|77.842004|

Each+.289017pp vsrelease,−.096339vsC04incumbent77.938343;XYZ−XY=0pp.
Both final128mean77.360308,T77.456647/V77.263969; finalV2Tguardrail fails.
XYfinalV2TR5/R10=93.256262/95.375723;
XYZfinalV2TR5/R10=92.870906/95.568401. Do not collapse allmetrics into equality.
Step64fusion score matrices differ(maxabs.144636,meanabs.018757); sameR1 is
not an inactiveZ-path bug. Finalgeometry outputweightnormXY.482481/XYZ.480432;
first16→last16meanlossXY.038296→.020076,XYZ.038304→.019937. These descriptive
checks do not prove optimaltraining, noise causality or depth uselessness.

Decision: retain both compactbest and cache; no promotion/SOTA/depthclaim.
Run matched fusion-only TRAIN512/128steps control before attributing+.289pp
to geometry; existing fullTRAINcontrols are not data/recipe matched.
No speculativeLRincrease, extra3Dextraction or duplicate seedpair yet.
V2used16362.928834s,remaining5129.071166s. Currentpool3054.454771s,
reserve900s forcontrol,old382.616394s+unallocated1692s unchanged.

VALID CORPUS UPDATE: extraction002 completed1031videos/60023frames,
3753.572271s,53672160bytes cache; exact intendedTRAIN512/DEV519. Original
detector invariant2classes/21joints/no trainer holds. Allcached XYZ finite and
frame/clip arrays match plan. This supports usable input plumbing, not pose
accuracy or retrieval gains. Chargeonce: V2used16032.776659s,remaining5459.223341s.
C09matched XY/XYZ training pair now registered; see METHOD_POSE3D_C09.md.

CORRECTION2026-09-19: samples001/002 are NOT valid clean inference evidence:
top-level eval recursively invoked YOLO.train(False), whose overload starts
training. Sample002 logs prove100epochs on coco8-pose and detector shape changed
21→17 and classes2→1 in memory. Retract quality/crop-effect interpretations
below; preserve historical records. This was an SLR wrapper runtime bug, not
evidence against H4W3D. Corrected inference directly sets module training flags,
no workflow calls; asserts2hand labels/21keypoints/no trainer. Clean sample003
completed24frames13.774706s with invariant passed, no trainer in log. Estimated
3D retrieval benefit remains unmeasured. Failed extraction001 stopped55.416033s,
0 saved videos. Resource total nowused12279.204388s/remaining9212.795612s.

Later user-completed sample001:24 finite TRAIN frames, wall70.350850s,
inference1.542257s. This is a later run after runtime002's missing-kornia
failure, not evidence that the failed parent succeeded. Runtime002 wholewall
62.155916s retained and charged separately; no repeated environment install.
Person-crop sample002: completed24 frames, wall68.140132s, inference1.524722s,
detector0.163816s, peak4.209842GiB, no person-detector fallback. All3 overlays
inspected by AI; visible hand/body placement offsets remain. Neither projection
coverage nor finite output establishes3D accuracy or retrieval improvement.
Decision: stop crop-only checks; C09 exact-center geometry pilot with original
pose2D/RGB retained. TRAIN512 fixedhash +DEV519,60023 extracted center frames.
See METHOD_POSE3D_C09.md for matched XY/XYZ contrast and limitations.
After bootstrap00124.335958s, runtime00262.155916s, later sample00170.350850s
and sample00268.140132s: used12210.013649s, remaining9281.986351s. These samples
are not training candidates with DEV scores and must not be ranked as such.
## C15 implementation smoke repair — no efficacy result yet

FullDEV zero-init parity passed; smoke001 failed activation gate before longpilot.
Native warmup LR0 at first update delays the zero-output branch's upstream task
gradient until backward3; initial gate wrongly required it at2. Native-optimizer
unit test confirms chronology. Repaired checks require output update2 and finite
nonzero upstream gradient/both projection updates3, without changing pilot recipe.
Failed smoke25.119405s charged once; logs/source retained. Chain002 launched for
smoke3updates→666update pilot. No C15 improvement claim or TEST access.
Repaired smoke002 passed3updates in30.920887s, including nonzero input gradient
and both projection updates; chain002 now executing the pilot in background.

## C21 canonical 3D motion — valid negative result

Retry003 completed128/128, exit0/no early stop, after two pre-training validation
repairs. All zero-init, geometry-update, roundtrip, frozen-buffer and identity
gates passed. Fused meanR1 at0/64/128 was77.552987/76.878613/76.974952;
selector retained initialization. This ties matched fusion-only control, trails
rawXYZ selected77.842004 by.289017pp and global incumbent78.709056 by1.156069pp.
Final T2V improved to77.649326 while V2T fell to76.300578, a harmful directional
tradeoff. DEFER exact anatomical-frame+motion recipe; no ablation/sweep. Wrapper
resumed UniFormerV2. Removed only unselected221789958-byte last.pt with manifest;
metrics/logs/source retained. Attempt001/002 are validation failures, not efficacy
runs, and their compute is separately accounted in STATE/BUDGET.

## C16 SignRep assets complete / transfer experiment launched

Official checkpoint strict-loaded; TRAIN512+DEV519 exact nativewindow cache complete:
1031videos/60023clips, finite768D features+latent, all starts/windows validated.
Extraction1654.872425s, peak1.95GiBVRAM, NPZ171376889bytes. No retrieval metric yet.
New hypothesis: train-only clip-level externalRGB feature supervision of existing
2DGCN; native RGB+pose2D inference unchanged. Parentc16-signrep-transfer-pair-001
launches smoke then matched160update transfer/no-transfer arms on TRAIN512,
fullDEV80/160, noTEST. Selection/init/guardrail unchanged; incumbent78.709056 retained.
This exploratory pair and pretrained usage are not a novelty or SOTA claim.

Actual launch outcome: smoke FAILED at0updates, historical step0 score equality.
RGBscores exactly equal; max pose diff.1628227234, fusion diff.0973358154.
Pose difference>1e-4 confined to7/519video rows; cause UNRESOLVED.
The auxiliary hook returns nativeRGB/pose/mask unchanged in eval, but this source
observation alone does not establish numerical equivalence. No tolerance relaxed,
no baseline campaign, no transfer/control training started, no scientific NO-GO.
Parent21.116541s chargedonce. Resume only after a scoped changed-path explanation.

Hookcheck002: exact native/hooked/unhooked output on32sameDEVitems; no modification
of inference tensors in that check. Historical R1/5/10 allmatch, fusionMeanRmatch;
poseV2TMeanRdelta1/519. Crossprocessscore cause stillunresolved, not called fixed.
Explicit validation amendment beforetraining inMETHOD_SIGNREP_C16: freshfullDEVinit
botharms, recallgates+exacthookproof, historicalscorevariance disclosed. Pair002
launched with original160step transfer/control recipe and hard1800s; no efficacy yet.

Pair002 actualoutcome: OOM intransferforward23 after22completedupdates;
smoke002 passed, matchedcontrol neverstarted. Parent207.713922s chargedonce.
No posttrainingDEV or selectedgain available; no scientific methodfailure claim.
User requested retry: CPU savedtensoroffload, sameB32/loss/LR/horizon, failedbatch
smoke thenonepilot under1700s. Offloadcontrol deferred pending resources;
no inference overhead added, hostRAM/transfer cost logged. No furtherpolling.

Offloadretry001 timedout before completedupdate1, not newOOM: smoke's fullDEV
used133.3s of150s cap. Parent151.531626s charged. Repair002 removes onlysmokeDEV;
actualpilot evaluation/recipe unchanged, no efficacy result available yet.
Retry002 thenfailedfirstbackward on efficientSDPA maskstride4225; peak2.75GiB,
notOOM. Retry003 selects mathSDPA for offloadedtraining; same formula but kernel
numerics/runtime may differ. Controlmustmatch if later run. No methodscore yet.

## C22 shared normalized-phase modulation — valid negative increment

Completed222/222, no early stop. Fused meanR1 at0/111/222 was
77.552987/77.745665/77.863865; selected final T2V77.307692 and V2T78.420039.
This is +.310879pp over release but -.170817pp versus the exact matched clean-GCN
control78.034682 and -.845190pp versus the global incumbent78.709056. All
zero-identity, separate stream-factor, shared-phase, GCN/fusion, buffer and
data-order gates passed. DEFER the exact C22 recipe without a hyperparameter or
seed sweep. Selected best retained; only redundant last.pt removed with manifest.

## C23 hierarchical graph adapters — valid negative increment

Completed222/222, no early stop. Fused meanR1 at0/111/222 was
77.552987/77.649326/77.842004; selected/final T2V76.685934 and V2T78.998073.
Against exact clean-GCN this is -.192678pp mean, with T2V-.770713pp and
V2T+.385356pp; global-incumbent gap is-.867052pp. All zero-identity, two-stage
adapter update, frozen-GCN, fusion/buffer and exact-order gates passed. The
mechanism is active but creates a harmful directional trade-off. DEFER exact C23
without rank/LR/horizon/seed sweep. Selected best retained; only redundant
last.pt removed after checksum manifest.

## Post-C23 next lead / active prerequisite

UniFormerV2-L/14@336 frozen features are being extracted independently as a
prospective complementary RGB representation. One snapshot after C23 found
TEST642 and DEV519 complete and TRAIN3037/7096 complete. No retrieval method or
gain is claimed. C24 must retain native RGB+pose2D, align the donor's 32-frame
stride1 clips to SEDS' at-most64-token interface, and separate representation
gain from fusion/extra-pretraining with a matched native-only control. Training
is not admitted from the remaining429.629546s compute allocation.
