# SLRet — current state (Goal V4; existing V2 artifact directory retained)

## Material Passport

academic-research-suite / experiment-agent, inline run, 2026-09-20.
Authority: docs/guide/ASTRA6_SLRET_RESEARCH_GOAL.md V4 + SLRET_RESEARCH_SKILLS.md + user.
Goal NOT achieved. Preserve user edits, no autopush.

Direction override, 2026-09-21: the user explicitly deferred C24, then deferred
CoSign-LI/C25 before its first GPU efficacy test. C25 is USER_DEFERRED, not an
empirical failure. The active work moved to `research/slret_dataset_first/`;
`C26_DECISION.md` records the replacement selection. No new lead has efficacy
evidence yet. C24 remains deferred. Download-only job `c26-assets-001`
COMPLETED, exit 0, 7/7 files and 3,520,971,333 verified bytes; records are
under `artifacts/slret_goal/jobs/c26-assets-001/`. Real checkpoint strict-load
and a two-example CSL DEV CPU forward pass after a float32 pose-loader repair.
This did not change the prior ~430-second GPU balance. The user subsequently
granted a new maximum five GPU-hours (20 GB VRAM) for C26. Technical attempt
`c26-unisign-csl-pilot-001` FAILED after all 19,478 frozen features and
zero-shot DEV were produced; the rectangular 1,077-video/797-text scorer bug
was repaired. Cache-only retry `c26-unisign-csl-pilot-002` COMPLETED all five
projection epochs, selected epoch4 mean DEV R@1 43.700670 (T2V 44.040151,
V2T 43.361188), versus zero-shot 5.785507 and historical CiCo CSL DEV
67.403588 (different visual/text pretraining and English caption rendering;
not a matched causal control). C26-A standalone is deferred for poor accuracy.
Both supervisors verified exact UniFormerV2 worker PID3493585/startticks190163677
resumed; at the user's status check its state was R. No C26 job is active.
Conservative total GPU wall charged 123.087s of new 18,000s grant; 17,876.913s
remain, separate from prior ~430s. No TEST loaded. The next research lead is
not yet selected; no C26 sweep or additional job is queued.

Historical background-job state at the last user-authorized check: the previously launched
UniFormerV2-L/14@336 extraction is the only active research workload: exact worker
PID3493585/startticks190163677/PGID3493571 is running after C23's wrapper verified
SIGCONT. The user-authorized check at 2026-09-21T18:24:56+07:00 found TEST642/642
and DEV519/519 complete and TRAIN3059/7096 feature+sidecar+temporal triplets
present; the exact process was state R and the workload log estimated about
23.1h remaining, with no failure observed. Do not poll it. User will report when
extraction is terminal. C24 integration is deferred by the user and has never
been admitted or trained. Adopted
job record: `artifacts/slret_goal/jobs/v4-uniformerv2-extraction-existing-001/`;
live log: `artifacts/logs/ph_uniformerv2_extraction.log`. Inspect it only after a
user completion/status report; do not automatically resume C24 afterward.

C23 terminal result: completed222/222, no early stop, supervisor wall
566.8829510211945s/trainer559.3189721107483s. Selected/final step222 meanR1=
77.84200385356455 (T2V76.6859344894027/V2T78.9980732177264), -.19267822736030pp
versus exact clean-GCN control78.03468208092485 and -.86705202312139pp versus
global incumbent. Relative to control, T2V is-.77071290944123pp and V2T is
+.38535645472062pp: a harmful directional trade-off, not a net lead. All identity,
two-stage adapter, frozen-GCN, fusion, buffer and data-order gates passed. DEFER
exact C23 without rank/LR/horizon/seed sweep. Best SHA256
d5f94f8bf02a8194fabed4579bfc78649e8c3f747bc5d0542f1289d221f76b42 retained.
Removed only redundant unselected last.pt945463878bytes after manifest
storage-prune-c23-20260921-001; metrics/logs/source/best remain.

C22 terminal result: selected/final step222 meanR1=77.86386542166889
(T2V77.3076923076923/V2T78.42003853564547), +.31087890914480pp over
release initialization but -.17081665925596pp versus exact seed42 clean-GCN
control78.03468208092485 and -.84519045501705pp versus global incumbent.
All registered identity, two-stage modulation, GCN/fusion, frozen-buffer and
data-order gates passed; this is a valid negative increment, not a runtime failure.
DEFER exact C22 without rank/LR/seed sweep. Best SHA256
3b78910317a7d503b202b02b31d30ad3884198d9d7a0019e54ef421f8b6fe392 retained.
Removed only redundant unselected last.pt1098942738bytes after manifest
storage-prune-c22-20260921-001; metrics/logs/source/best remain.

C21 attempt001 FAILED before any training update after46.37914991378784s. The
wrapper automatically resumed UniFormerV2 as designed. This was not an efficacy
or OOM result: full-DEV step0 produced exact baseline R1/R5/R10/MedianR/MeanR for
all three streams, while an old elementwise score-array gate rejected backend
floating scheduling differences (fusion max.0973358154, pose max.1628227234,
RGB0). Retry repair keeps finite/shape checks and exact all-metric parity, records
score max differences, and leaves the stricter historical gate unchanged for all
non-canonical recipes. Ten scoped tests and syntax checks pass. Attempt001 remains
retained under its unique run/job IDs; later retries do not overwrite it.

C21 retry002 FAILED before training after26.20063304901123s because the repaired
gate also required exact MeanR. All primary R1/R5/R10 and MedianR values matched;
only pose V2T MeanR shifted by exactly1/519=.0019267822736033224 from one rank
outside top10. Retry003 explicitly verifies zero output weights, finite scores,
exact primary metrics/MedianR, and bounds MeanR drift to at most one rank across
DEV519. It passed the full-DEV gate and reached step16 while UniFormerV2 remained
kernel-stopped. This was active training, not merely successful initialization.

C21 retry003 terminal evidence: supervisor wall161.26125240325928s, trainer
wall158.4188015460968s, peakCUDA3262604288bytes, all zero-init/update/roundtrip/
frozen-buffer/data-order gates passed. Selected step0 ties matched fusion-only
control, is-.28901734104046pp versus rawXYZ selected and-1.15606936416185pp
versus global incumbent. Final128 is-.57803468208093pp versus initialization,
-.38535645472061pp versus rawXYZ final, with T2V+1.15606936416185pp but
V2T-2.31213872832370pp from initialization. Generic supervisor field
`candidate_minus_control_pp=-.289017` is actually candidate-minus-rawXYZ because
C21 validator's registered anchor is rawXYZ; it is not the fusion-control delta.
No best.pt was produced because no trained checkpoint beat step0. Removed only
unselected last.pt221789958bytes permanently; all metrics/logs/source retained,
manifest storage-prune-c21-20260921-001.

v4-c20-rgb-moddrop-annealed-001 TIMED_OUT at2300s, supervisor wall
2303.211750268936s,146/666 updates, exit1. Launcher4076368/supervisor4076369/
torchrun4076373/trainer4076422 are absent. No OOM or model exception; hard timeout
sent SIGTERM. Peak owned GPU17987272704bytes, below user20GB sampled limit;
max telemetry query.104757s. Trainer run.json is stale `running` because SIGTERM
preempted its terminal write; supervisor summary is authoritative.

Only post-training DEV checkpoint111: meanR1=77.84200385356455 (T2V77.071291,
V2T78.612717), -.09633911368015pp versus historical GCN-R1 seed42 at same step
77.9383429672447, -.77071290944123pp versus its selected78.61271676300578 and
-.86705202312139pp versus global incumbent78.70905587668594. Same base/data/batch
order hash. The train-only path activated:830/4672 sampled items dropped through
step146 (17.7654%), GCN/fusion updates and frozen-buffer checks passed. However
probability remained.15639 and the registered zero-drop terminal gate was never
reached; do not call this a completed negative result. No C20 recovery: interim
fused metric is already below the common-step control, exact optimizer/RNG state
was not periodically saved, and remaining compute cannot support a valid replay.

C19-R1 retry002 COMPLETED160/160, exit0, supervisor wall1737.84938955307s;
launcher4025211, supervisor4025212 and torchrun4025213 are absent. Selected/final
meanR1=78.22736030828517 (T2V77.263969/V2T79.190751), -.09633911368015902pp
versus matched native subset control78.32369942196533 and also -.096339114pp
versus completed bidirectional C19. Curve0/80/160=77.552987/78.131021/78.227360.
All identity, direction-specific trainability, output/gradient, GCN/fusion,
frozen-buffer and delta-roundtrip gates passed. The adapter was active, but the
directed contrast did not recover a net gain. DEFER the global-exchange family;
no LR/rank/slot/within-stream sweep and no novelty/SOTA claim.

C19-R1 attempt001 FAILED after2 completed updates, exit1, wall261.3941104412079s;
launcher/supervisor/torchrun all absent. No post-training DEV result or scientific
efficacy conclusion. Full DEV step0 matched release; output projection changed at
step2 and native update/buffer gates passed. Failure at backward3 was caused by
generic fusion policy re-enabling the deliberately inactive RGB-to-pose direction:
optimizer recorded262656 adapter params instead of registered131328, then the
all-gradient gate correctly rejected unused reverse parameters. This is a scoped
trainability-order bug, not evidence against directed exchange.

Repair: reapply directional trainability after generic fusion configuration and
before optimizer construction; pose-to-RGB remains the only active direction.
Smoke length now includes3 updates so the upstream-gradient gate is reachable.
18 targeted CPU tests pass, including regression for fusion-policy re-enable and
all asymmetric trainable gradients after output activation; syntax checks pass.
Retry002 uses a unique run/job ID and otherwise identical scientific recipe.

C19 recovery COMPLETED160/160, exit0, wall971.9356958866119s; all PIDs absent.
Resume restored model/optimizer/RNG at saved112 and recomputed113..121 before
continuing to160; first-resumed-update and all original identity/gradient/output/
GCN/fusion/frozen-buffer/delta gates passed. Full-GPU bitwise equivalence is not
claimed. Selected160mean78.32369942196532, exact mean tie (floating roundoff only)
with matched native control78.32369942196533. C19 T2V77.263969 vs77.071291 control
(+.192678pp), while V2T79.383430 vs79.576108 (-.192678pp). Curve0/80/160:
77.552987/78.131021/78.323699 vs control77.552987/78.034682/78.323699.
Therefore no net improvement, no promotion and no within-stream capacity control.
Best SHA2565a349fc571d530d6ecf9136cff2256a5ab7a50661a59cbbdff515d7c650703c8
retained. Peak owned GPU9927917568bytes; max query.125862s.

Directed C19 retained only pose-to-RGB global messages, preserving native pose-local
inputs. Same seed42/TRAIN512/B32/10epochs/160updates/native loss/GCN1e-6/
fusion1e-5/adapter1e-4/eval0,80,160. Total schema262656 adapter parameters;
131328 trainable. The negative result rejects the registered contamination
explanation for this recipe, without claiming all cross-modal exchange is useless.

CPU C19/C19-resume tests17/17 pass; broader unittest ran76 executable tests and
only three unrelated imports failed because pytest is absent in the seds env.
Syntax checks pass. No TEST access or baseline replay.

C19-R1 retry002 supervisor1737.84938955307s remains charged exactly once in the
cumulative accounting below.
Budget after charging C23 supervisor wall566.8829510211945s once: used
60662.370454086995s of61092s; remaining429.629545913005s. C23's950s reserve is
released. This cannot fund the measured ~567s exact one-epoch full-TRAIN recipe,
so no shortened or unmatched C24 training is admitted. User20GBdecimal remains
binding; C23 peak owned17.968GB and trainer peakCUDA15.105GB were below it.
After permanently deleting only the two completed C19 recovery `last.pt` states
(908681322bytes), selected bests/logs/data retained, V2 artifacts42273282447bytes;
free133790625792bytes/floor15GiB. Manifest storage-prune-c19-recovery-20260920-001.
Directed C19-R1 unused `last.pt`453287783bytes also permanently removed after
hash/size recording; best/logs/scores/source retained. Manifest
storage-prune-c19-r1-20260921-001. Global incumbent full-TRAIN seed1337 step222
remains78.70905587668594.
For C20 headroom, permanently removed only unselected seed1337/2026 GCN-R1
`last.pt` states2197815324bytes; all selected bests/logs/metrics/source retained.
V2 artifacts40441406611bytes; free135469223936bytes; manifest
storage-prune-gcn-r1-last-20260921-001. C20 planned4GiB fits42GiB cap/floor15GiB.
C20 added one retained common-step best checkpoint804973524bytes (SHA256
8a87050d2987f08cc1ac0090fe74d0339021d33a1ece98920319808e4b9ca2c1), logs and
scores; no last.pt exists. V2 artifacts41316001640bytes; free134241001472bytes.

C21 retry003 completed recipe: same C09 TRAIN512/cache/order, seed42,B32,
8epochs/128updates, branch1e-4/fusion1e-5,DEV0/64/128. Selected initialization;
trained steps degraded the fused selector. Ten scoped tests and all runtime gates
pass, so this is a valid negative result rather than an implementation failure.

C23 completed and is deferred as reported above. The prospective C24 lead uses
the ongoing frozen UniFormerV2 representation as an additional RGB source while
retaining native RGB and pose2D; it must explicitly align 32-frame stride1 donor
clips to SEDS' 64-token contract and use a native-only matched control. It is not
ready until TRAIN7096/7096 extraction validates, and remaining allocated compute
cannot fund a fair full-TRAIN pilot. No C24 efficacy/novelty claim exists.
For42GiB cap headroom, permanently removed only two redundant unselected last.pt
states2197769244bytes from GCN-R1seed42 and one-epochGCNseed1337; both selected
best.pt and all metrics/logs retained. Manifest storage-prune-c22-admission-20260921-001.

Incumbent78.709056 unchanged; nativeB32subsetcontrol78.323699. C18 B64 deferred
(selected78.227360/final78.131021, charge1563.841474s already accounted).
C16 directSignRep transfer deferred. C06/C17 USER-CLOSED; no DCL/recovery reopening.

## Reference / best artifacts

- Native SEDS release: `third_party/SEDS/ckpt/ph_best_model.bin`.
- TRAIN7096 `artifacts/slret_goal/seds-adapted-train-001`; DEV519
  `artifacts/slret_goal/seds-adapted-dev-001`; no re-extraction needed.
- Fixed native full-DEV mean(T2V R1,V2T R1)=77.5529865125.
- NEW provisional best: GCN-R1 three-epoch schedule, seed1337, step222,
  mean78.7090558767; T77.4566473988/V79.9614643545; +1.1560693642pp release.
  `artifacts/slret_goal_v2/seds-gcn-horizon3-seed1337-001/best.pt`, sha256
  e7f0baa47343e25a28a7c38d02af65a412d6b8e53c3b8fabd18026ba625be095.
  R1 selected42/1337/2026=78.612717/78.709056/78.516378, mean78.612717,
  sampleSD.096339; paired mean+.385356pp over1epoch. All selected improve.
  Fixed666 mean78.355812, SD.294321; two seeds decline late. Keep ALL bests.
- Previous one-epoch clean GCN+fusion best, seed1337, step111, retained:
  mean78.5163776493; T76.8786127168/V80.1541425819; +.9633911368pp release.
  `artifacts/slret_goal_v2/seds-gcn-clean-seed1337-001/best.pt`.
  Source recipe is C08 control, NOT a masked-reconstruction win.
- Previous C04 LoRA best77.9383429672, T77.071291/V78.805395, retained at
  `artifacts/slret_goal_v2/seds-lora-001/best.pt` with replication artifacts.
- Three clean GCN seeds42/1337/2026 select78.034682/78.516378/78.131021;
  mean78.227360, sampleSD.254889, mean gain+.674374pp over release. Final222
  meanR1 equals selected for each seed, with different directional tradeoffs.
  Keep ALL seeds and C04; repeated DEV selection is not independent confirmation.
- PH/CSL TEST exposed historically; not used for current selection/hypotheses.

## Earlier negative pilots
C08 masked reconstruction selected77.745665 vs cleanGCN78.034682, defer.
C11 simultaneousGCN+LoRA selected77.938343 below matched cleanGCN, defer.
Implementation gates passed; no primary-metric increment. Full RESULTS.md.

## Completed clean GCN replication / matched ablation

Seeds1337/2026 completed222updates582.724964/576.091494s; charge1158.816458s.
All3 seeds above release both selected and fixed222. New best1337 retained;
selected step111 ties step222 mean but T/V tradeoffs differ. No early stopping,
all expected LR/update/frozen-buffer checks passed. Means are descriptive only.
New matched1epoch fusion-only controls complete222 each, exact seed/data/order
and only-fusion update checks passed. Selected42/1337/2026 meanR1:
77.552987/77.649326/77.552987; paired GCN gains+.481696/+.867052/+.578035pp,
mean+.642261 (sampleSD.200546). Fixed222 gains+.481696/+.867052/+.770713pp.
GCN adaptation improves all3 matched controls under this recipe; no independent
generalization or SOTA claim. C04 historical horizon remains unmatched.
Charge214.254928+220.942733+243.751525=678.949186s once; all artifacts retained.

## Completed C12: no increment from clip-local temporal adaptation
C12 selected222mean78.034682, exacttie matchedGCNseed42; temporalupdates and
frozenbufferchecks pass. Defer exactrecipe, keep provenance; full RESULTS.md.

## Resource accounting / retention

Latest user grants about5hours more and max20GB VRAM: +18000s, ceiling61092s.
Used49768.476597s; available11323.523403s; reserve2450s, unreserved8873.523403s.
Charge each new supervisor wall once on collection, not child wall again.
R2 children29.741496+946.888958=976.630454s once; exclude parent988.221927s.
C13 charge27.963225+1808.982913=1836.946137s once; exclude parent1843.911873s.
C13-R1 charge28.279787+1760.478473=1788.758260s, not parent1798.929753s.
C14 charge22.851332+1713.533791=1736.385123s once, not parent1743.792428s.
C14-R1 charge28.070673+1702.284639=1730.355311s, not parent1738.895681s.
C15-R1charge2143.777292s once; parent2154.044604s excluded; faileddownload120s charged.
Prior charges remain consolidated in BUDGET_TOPUP_20260920.md; no cloud or unlimited grant.
Before C13: deleted3unusedlast states2986610544bytes/2.781498GiB permanently;
R2freeze, fusionseed2026, oneepochfusionablation1337 only. All selected models
retained; manifest storage-prune-adaptive-graph-20260920-001. V2~30.98GiB,
disk~139GiBfree; C13planned4GiB+smoke128MiB fits36GiBcap/15GiBfloor.
Deleted old fusion-control42/1337 last states only (1.757311GiB, permanent), selected release/best retained;
manifest storage-prune-gcn-freeze-20260920-001, all GCN models/data/logs preserved.
Removed3933372004bytes (3.663238GiB) for two sequential4GiB reservations.
Targets: C08masked-pose002 last, horizon3fusioncontrol last, one-epoch fusion
ablation42/2026 last only. Preserve ALL selected checkpoints/logs/data.
Manifest storage-prune-gcn-replication-20260920-001; permanent/no trash.
free floor15GiB. Deleted only deferred C11 last.pt (1098902278bytes /1.023432GiB),
preserved its best/logs; permanent/no trash, storage-prune-c11-last-20260920-001.
Deleted only C12 last.pt (1344307134bytes /1.251984GiB), retained its best and
all logs. Permanent removal, no trash; storage-prune-c12-last-20260920-001 manifest.
User authorizes old unused checkpoints cleanup; prior cleanup removed19.179GiB
with manifest `artifacts/slret_goal_v2/storage-prune-20260919-001/manifest.json`.
Keep release, both incumbent candidates, C04 replications, H4W corrected cache
`h4w-ph-center-pilot-002` (51MiB) and C09 compact weights. Keep datasets/features.
C06 user-closed. C07 corrected attention and current C09 additive geometry deferred.

## Completed GCN-R1 / next actual user return

Registered METHOD_GCN_HORIZON_R1.md: positive GCN lead, one-pass curve has no
late decline; test3epochs666steps seed42 from release. Same GCN1e-6/fusion1e-5,
native losses/BN/dropout/B32/data. No C11/C12 branch, no novel-method claim.
DEV0/111/222/444/666, same selector and guardrails. Horizon stretches scheduler
from the start; compare both common steps and selected, disclose extra selection.
Reuse native clean smoke; real first2-update checks retained, no baseline replay.
Driver --long-gcn-pilot, chain gcn-horizon3-pilot-001, child seds-gcn-horizon3-001.
Completed666updates, all gates passed, no earlystop. Mean111/222/444/666:
77.938343/78.420039/78.516378/78.612717. Promote provisional lead; RESULTS.md.
Matched3epoch fusion-only control completed666 with exact seed42/base/data/order,
LR/schedule/selector, only fusion updated/frozen buffers fixed. Selectedinit and
finalmean77.552987, paired GCN gain+1.059730pp. Historical C04 control metrics
also match all recorded steps; do not replay unchanged controls again.
Replication completed both666steps, all gates passed/noearlystop. Best1337 at222,
2026 also peaks222; final means78.420039/78.034682. See RESULTS.md for all seeds,
selected/fixed comparisons and late-decline limitation, no cherry-picking.
GCN-R2 train completed666 with transition223/no optimizer reset and exact frozen
GCN final; selected222mean78.420039, final78.034682. No gain; defer current recipe.
Parent chain FAILED exact pre-freeze comparison: R1/5/10 match at111/222 but
MeanR differs1/519, score maxabs .00652/.00630; loss/grad differ by10 despite
identical1/2. Same native config/base/data/order/optimizer groups; cause unresolved.
Do not label it exact causal evidence, change failed status or relax the gate.
Scoped saved-output/source checks only; no baseline replay/automatic retraining.
Keep all R1 leads. Details RESULTS.md and METHOD_GCN_FREEZE_R2.md.
COMPLETED C13: all666updates, allgatespassed, noearlystop. Best444mean77.745665,
T76.878613/V78.612717; +.192678pprelease but-.867052ppR1seed42 and-.963391pp
globalbest. Finalfusion77.552987, finalpose64.643545 vsrelease60.500963/R1pose63.776493.
No promotion. Bestsha cb35f5e98b3f009d4223c7aead5484a15500b8723bd9c5e32163c105da841817;
keep best/logs; remove only unusedC13last1098984006bytes (~1.024GiB), permanent,
manifest storage-prune-c13-last-20260920-001. Data/allincumbents retained.
COMPLETED C13-R1 graphLR1e-5:666updates/allgatespass/noearlystop. Best111mean78.131021,
T77.263969/V78.998073; +.385356ppC13 but-.481696ppR1seed42/-.578035ppglobalbest.
Finalfusion77.842004. Defer C13topologyfamily now, no furtherLR/horizon/seed sweep.
Keep selectedbest44fba4feda483339afd972fa02ccbd1fc51d34ace0b729234ff97af91d6d9f58.
COMPLETED C14: all666updates/gatespass/noearlystop; best444mean78.612717,
T77.842004/V79.383430, exacttieR1seed42, -.096339ppglobalbest. Final78.516378.
Bestsha f92b981dd344bb584adaf46056869ac6c5e9347198de9aaef3bb1cb664a1dec3 kept.
Boneprojectionnorm660hand.01046447/body.00709206; not proof of undertraining.
COMPLETED C14-R1 boneLR1e-4, all666updates/gatespass/noearlystop. Best222mean78.131021,
T77.263969/V78.998073, -.481696pporiginalC14/R1seed42, -.578035ppglobalbest.
Final78.034682; allfourDEVpoints beloworiginalC14. Projectionnorms increased
~9.24xhand/8.45xbody without gain; causeunproven. Defer currentC14family,
no additionalLR/horizon/seed sweep. Bestsha retained:
0b64b36038dc07150845264631247ca83b96b82d6b7a527baea1f34a761a7cb8.
Cleanup onlyC13-R1last+originalC04last2043322862bytes~1.903GiB permanent;
allselectedbests/data/logsretained, storage-prune-c14-20260920-001 manifest.
For refinement retireonlyC14last+C04seed1337last2043265302bytes~1.903GiB,
permanent; selectedbests kept, storage-prune-c14-refinement-20260920-001 manifest.
COMPLETED C15all666/gatespass/noearlystop; selected666mean78.420039,
-.192678ppR1seed42/-.289017ppglobalbest. No promotion; retain best93518a5d...719e6.
COMPLETED C15-R1 all666/gatespass, selected444mean78.612717/finalsame; tieR1seed42,
-.096339ppbest. Deferpoolingfamily, noextrasweep. Best75166231...b1d816e6 retained.
RetireonlyC15last+olderoneepochGCN42last2198712420bytes~2.048GiB permanently;
selectedbests/data/logs/allR1stateskept, storage-prune-c15-refinement-20260920-001.
CURRENT C16 train-only SignRep-to-pose supervision, METHOD_SIGNREP_C16.md.
Extraction completed1031videos/60023clips in1654.872425s, chargedonce; allcachesvalid.
COMPLETED C16offload003:160steps, selected78.034682; no promotion, matchedcontrol now funded.
Pair001failedbeforetrain; hook002exact on32DEVitems; historicalscorevariance unresolved.
Cleanup unusedC15-R1last+oneepochGCN2026last2198712484bytes permanent; allbestskept.
Offload001 timeout;002 SDPAbackwardmaskalignment failure26.178253s chargedonce.
