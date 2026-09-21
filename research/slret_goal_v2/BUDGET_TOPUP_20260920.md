# Local GPU allocation update — 2026-09-20

## C23 collected / remaining allocation

No new grant. C23 completed222/222; charge supervisor wall566.8829510211945s
once (trainer559.3189721107483s is included): used60662.370454086995s of61092s,
remaining429.629545913005s. Release C23's950s reservation. This is below the
measured566.883s needed for the same full-TRAIN one-epoch recipe, so no shortened
or unmatched next training run is admitted. User20GB cap remains binding.

C23 peak owned process bytes17968398336 and trainer peakCUDA15104944640 stayed
below limits. Permanently removed only redundant unselected C23 last.pt
945463878bytes after checksum manifest storage-prune-c23-20260921-001; selected
best/logs/metrics/source retained. V2 artifacts41490056242bytes after cleanup.
C23 wrapper return0 resumed exact UniFormerV2 PID3493585 and verified state `R`.

## C22 collected / C23 admitted

No new grant. C22 completed222/222; charge supervisor wall611.9488961696625s
once (trainer604.0990860462189s is included): used60095.4875030658s of61092s,
remaining996.5124969342s. Release C22's1200s reservation. C23 reserves950s outer
bound (hard900s), leaving46.5124969342s unreserved. User20GB cap remains binding.
The same exact UniFormerV2 PID/start/cmd contract will stop only its worker and
SIGCONT it in `finally`. C22 wrapper already resumed it and verified state `R`.

C22 peak owned process bytes17989369856 and trainer peakCUDA14727352832 stayed
below limits. Permanently removed only redundant unselected C22 last.pt
1098942738bytes after checksum manifest storage-prune-c22-20260921-001; selected
best/logs/metrics/source retained. V2 artifacts40477110844bytes before C23.

## C22 admitted / UniFormerV2 temporarily paused

No new grant. Before launch used59483.53860689614s of61092s, remaining
1608.46139310386s. Reserve1200s for C22 outer bound (hard1150s), leaving
408.46139310386s unreserved. User20GB VRAM cap remains binding. User explicitly
authorizes pausing exact UniFormerV2 whenever a method needs GPU; wrapper173924
validated PID3493585/startticks190163677/cmd, SIGSTOP verified `Tl+`, and will
SIGCONT in finally. C22 passed full-DEV identity plus update3 gates and reached
step7; startup owned snapshot13.147GB below watchdog19GB. No further polling.
Charge supervisor wall once on collection and release reserve.

Storage admission permanently deleted only redundant unselected GCN-R1seed42 and
one-epochGCNseed1337 last.pt states2197769244bytes; selected bests/logs/metrics
remain. Manifest storage-prune-c22-admission-20260921-001. V2 artifacts after
cleanup39464552439bytes; free~127GiB, floor15GiB.

## C21 retry003 collected / canonical-motion recipe deferred

C21 retry003 COMPLETED128/128, supervisor161.26125240325928s; trainer
158.4188015460968s is included and not charged again. Charge supervisor once:
used59483.53860689614s of61092s, remaining1608.46139310386s. Release1050s reserve;
no active SLRet allocation. Wrapper return0 resumed UniFormerV2 and verified
state `R`. Candidate selected initialization and is deferred. Removed only its
unselected last.pt221789958bytes; all metrics/logs/source retained.

## C21 launch with user-authorized extractor pause

No new grant. Attempt001 failed before update0 after46.37914991378784s due an
elementwise score parity gate, not OOM or efficacy; its wrapper resumed the
extractor. Retry002 then failed pre-training after26.20063304901123s because pose
V2T MeanR moved exactly one rank/DEV519 while all R1/R5/R10/MedianR values matched.
Charge both once: used59322.27735449288s of61092s, remaining1769.72264550712s.
Retry003 reserves1050s (hard1000s plus wrapper headroom), leaving719.72264550712s
unreserved. User20GB VRAM limit remains binding.
User explicitly requested temporary pause of exact UniFormerV2 extractor
PID3493585/startticks190163677 and automatic resume after C21. Wrapper PID151330
validates its identity, runs C21 detached, and resumes it on completion, failure
or timeout. Retry003 uses PID-scoped stop/continue; monitoring through the repaired
full-DEV gate verified extractor state `T` and C21 running at16 updates. The stopped process retains8640MiB CUDA context
and therefore remains listed by nvidia-smi, but does not execute. No further
polling; charge actual retry supervisor wall once on user-reported completion and
release the reservation.

## C19-R1 retry002 collected / family closed — 2026-09-21

Charge v4-c19-r1-pose-to-rgb-002 supervisor1737.84938955307s ONCE; child trainer
wall1731.7395939826965s is not added. Used56946.48582126114s of61092s,
remaining4145.514178738857s. Release3650s reservation; then reserve2350s for
registered C20, leaving1795.514178738857s unreserved. C20 estimate1850s,
hard2300/outer2350. COMPLETED160/160, selected78.227360 versus control78.323699;
defer family. Observed ownedGPU10057940992bytes/CUDApeak6835746816bytes, below
user20GB at sampled checks. Permanently removed only its unused last.pt453287783
bytes after hash recording; best/logs/scores/source retained. Manifest
storage-prune-c19-r1-20260921-001. No TEST, download or extraction.
For C20's4GiB save headroom within42GiB cap, permanently remove two unselected
GCN-R1 last states2197815324bytes; all selected bests/provenance retained. V2
artifacts40441406611bytes, free135469223936bytes, floor15GiB; manifest
storage-prune-gcn-r1-last-20260921-001.
C20 launched detached at Unix1789927798.3877852. Sole startup check found
launcher4076368/supervisor4076369/torchrun4076373/trainer4076422 alive, status
RUNNING during native initialization with no immediate error. No polling; await
actual user return before collection.

C20 collection: TIMED_OUT at hard2300s after146/666; charge supervisor wall
2303.211750268936s ONCE, not trainer2285.1319556236267s. Used
59249.69757153008s of61092s; remaining1842.302428469923s. Release2350s reserve;
no new reserve. No OOM/model exception; peak ownedGPU17987272704bytes. Common-step
111 mean77.842004 is-.096339pp versus historical control111. No exact resume or
automatic replay. V2 artifacts41316001640bytes, free134241001472bytes; retain
step111 best/logs/scores, no last.pt. One later GPU snapshot showed an unowned
8520MiB/100%-util process; record only, never signal or manage it.

## C19-R1 technical failure / repair retry allocation — 2026-09-21

Charge v4-c19-r1-pose-to-rgb-001 supervisor261.3941104412079s ONCE; no child
double charge. Used55208.63643170807s of61092s, remaining5883.36356829193s.
Release3650s; reserve3650s for unchanged-recipe technical retry002; unreserved
2233.36356829193s. No new grant. Attempt001 ended after2 updates with no checkpoint
or post-train DEV and is not scientific evidence. Retry retains hard3600/outer3650,
16GiB allocator,19GB trip,20GB user cap and20s query deadline. V2 artifacts
42273282447bytes; free133790625792bytes/floor15GiB; no cleanup/download/TEST.

## C19 recovery collected / directed C19-R1 reservation

Charge v4-c19-resume-001 supervisor971.9356958866119s ONCE; inherited112 updates
and original C19 wall are not charged again. Used54947.24232126686s of61092s,
remaining6144.75767873314s. Release1850s; reserve3650s for
v4-c19-r1-pose-to-rgb-001; unreserved2494.75767873314s. No new grant.
Estimate3200s, hard3600s/outer3650s, B32/160updates. Keep16GiB allocator,
19GB owned-descendant trip, user20GB decimal cap and20s telemetry deadline.
Permanently removed only completed C19 parent/resume `last.pt` states908681322bytes;
selected bests/logs/data retained and hash-recorded. V2 artifacts42203858473bytes,
free134253056000bytes,15GiB floor. No download/extraction/TEST.

## C19 timeout collection / resume allocation

Charge v4-c19-global-exchange-001 supervisor2403.396547317505s ONCE; childwall
excluded. Used53975.30662538025s of61092s, remaining7116.6933746197465s.
Release2450s; reserve1850s for v4-c19-resume-001; unreserved5266.6933746197465s.
No new grant. Original reached121, last checkpoint112; resume113..160, recompute
9 unsavedupdates (included in new wall charge), no epoch/schedule extension.
Median original step17.452807s:48steps~838s plus finalDEV/startup; estimate18–25min,
hard1800s/outer1850s. Initial/80DEV inherited, not replayed. No automatic nextjob.
MeasuredownedGPU10057940992bytes below20GB at sampled checks;16GiBallocator/
19GBtrip/query20s retained. Artifacts42430335679bytes+1.25GiB reserve <42GiB;
free134074048512bytes/floor15GiB. No deletion/download or re-extraction.

## C18-002 completed / C19 pilot reservation

Charge supervisor1563.841474056244s ONCE (not child); C18-001 already charged.
Used51571.91007806275s of61092s; remaining9520.089921937251s. Release2450s.
Reserve2450s C19 v4-c19-global-exchange-001; unreserved7070.089921937251s.
No new grant. Estimate25–35min, hard2400s/outer2450s; one TRAIN512/B32 pilot.
C18 observedGPU17628659712bytes, allocated12224351744bytes; peak query7.457s.
C19 retains16GiB allocator/19GB sampled trip/user20GBdecimal and querydeadline20s.
Storage41679130629bytes+1.25GiB reserve exceeds40GiB by~68MiB; reallocate agent-only
subcap40->42GiB under GoalV4§8. Free134879973376bytes/floor15GiB. No deletion;
preserve historical/user assets. C19 compact adapter adds~1MiB weights.

## C18 infrastructure failure collection / single retry reservation

Charge v4-c18-batch64-001 supervisor239.5920069217682s ONCE; child wall not added.
Used50008.068604006505s of61092s; remaining11083.931395993495s. Release2450s.
Reserve2450s for v4-c18-batch64-002; unreserved8633.931395993495s. No new grant.
Same native B64/80updates recipe, hard2400s/outer2450s, estimate25–35min.
Infrastructure repair only: GPU query deadline5->20s, still failclosed on timeout
or unavailable telemetry;16GiB allocator/19GB sampled trip/user20GB unchanged.
Artifacts40863137187bytes +1.25GiB planned headroom <40GiB subcap;
free135714144256bytes >15GiB floor. Retain failed evidence; no deletion/download.

## C17-R1 timeout collection / user closes C17 / C18 reservation

Charge supervisor1803.0584270954132s once; stale childwall1672.012262s not added.
Used49768.47659708474s of61092s; remaining11323.523402915263s. Release1850s.
Prepared recovery never launched; charge0, reserve0 for C17 recovery.
Reserve2450s for v4-c18-batch64-001; unreserved8873.523402915263s. No new grant.
One pilot, native loss/B64,80steps at same5120TRAIN exposures. Estimate25–35min,
hard2400s/outer2450s; extra eval margin after prior1800s timeout.
User20GBdecimal,16GiB allocator/19GB owned-process watchdog retained. B64 memory
not yet measured; failclosed on OOM/trip, no automatic batch-size retry.
Current artifacts40793786658bytes (~37.99GiB)+1.25GiB reserve <40GiB;
free135811424256bytes (~126.49GiB), floor15GiB. No deletion/download/extraction.

## C17 collection / C17-R1 reservation

v4-c17-dcl-001 COMPLETED160; charge supervisor1447.346301317215s once, not
child1439.1033766269684s. Used47965.41816998932s of61092s; remaining13126.581830010677s.
Release old1850s; reserve1850s for v4-c17-dcl-blend005-001;
unreserved11276.581830010677s. No new grant. Estimate23–27min/hard1800s/outer1850s.
Observed ownedGPU10051649536bytes, allocated6835113472bytes for collected C17.
Retain16GiB allocator/19GB sampled watchdog/user20GB decimal maximum.
Agent-set storage subcap38->40GiB under GoalV4 section8 resource reallocation:
measured37.296432GiB +1.25GiB saveheadroom=38.546432GiB fits40GiB. Free127.211262GiB,
floor15GiB. Preserve all historical/user assets; no deletion, download or extraction.

## C16-R2 collection / C17 reservation

v4-c16-relation-weight1-001 COMPLETED160, supervisor1187.2980473041534s chargedonce;
child wall excluded. Used46518.07186867211s of61092s; remaining14573.928131327892s.
Release old1850s; reserve1850s for v4-c17-dcl-001, unreserved12723.928131327892s.
No additional grant. Observed processGPU9.943GB/allocated6.386GiB on R2;
C17 keeps16GiB allocator,19GB local safetytrip, user20GB decimal maximum.
Storage36.536414GiB +1.25GiB saveheadroom <38GiB; free127.996GiB >15GiB floor.
No historical deletion or new download. One pilot160steps, estimate20–25min,
hard1800s/outer1850s; native teacher-free dataset, first2update gates inside job.

## C16-R1 collection / final C16-R2 reservation

v4-c16-relation-001 COMPLETED; supervisor1222.4315004348755s charged once.
Used45330.773821367955s of61092s; remaining15761.226178632045s.
Release old1850s, reserve1850s for v4-c16-relation-weight1-001;
unreserved13911.226178632045s. No new grant, no double-counted child wall.
Observed owned-processGPU9942597632bytes, allocator6857040384bytes;
16GiB allocator/19GB trip/20GB user cap retained. Local tracker, no agent polling.
Storage35.758776GiB +1.25GiB headroom <38GiB; free128.792397GiB >15GiB floor.
No deletion/download. Estimate20–25min, hard1800s plus outer1850s. One run only.

## C16 control collection / C16-R1 reservation

v4-c16-control-001 terminalCOMPLETED, supervisor1382.3132772445679s chargedonce;
child1374.7376325130463s excluded. Used44108.34232093308s of61092s;
remaining16983.65767906692s. Release old1850s; reserve new1850s for
v4-c16-relation-001, unreserved15133.65767906692s. No grant added this turn.
Control allocatedGPUpeak6.370GiB; old process-group watchdog0 was invalid telemetry
because torchrun workers start a new session. Preserve rawlogs, no totalGPU claim.
New tracker follows descendants across sessions, retains start ticks to avoid PID
reuse, fails on unavailable memory and requires nonzeroGPU observation for success.
Allocator16GiB/trip19GB/user20GB remain. No other user's GPU process touched.
Estimated23–27min/hard1800s+outer1850s; one pilot, not automatic tuning.
Reallocate agent-set storage allowance36->38GiB under V4§8 rather than delete
historical/user assets. Current34.977GiB +1.25GiB measured-save headroom fits;
free129.97GiB, floor15GiB unchanged; no deletion/download/feature extraction.

## Latest explicit five-hour grant — after C16 collection

User: "cấp thêm cho bạn tầm 5 tiếng compute, tối đa 20gb vram".
Add18000s once to43092s: ceiling61092s. Used42726.02904368851s unchanged;
available18365.97095631149s. Reserve1850s for v4-c16-control-001;
unreserved16515.97095631149s. Charge actual supervisor wall once on collection,
including failures; do not also add child wall. This supersedes the pending
1800s request, not an additional second grant. No automatic top-up.

Interpret max20GB conservatively as20,000,000,000bytes for our jobs (not other
user processes). Set PyTorch allocator16GiB and local process-group watchdog
trip19,000,000,000bytes, check5s; reserved margin for non-tensor CUDA memory.
Not a hardware-enforced instantaneous total-memory partition. Training timeout
1800s; detached outer1850s. Known control only, no automatic tuning/new method.
V4 run.log/status.json/summary.json; one startup check then WAITING_FOR_USER.
Current V2 artifacts34.2083GiB, free130.77GiB; measured compact checkpoints
justify1.25GiB save reservation (best162MB + last465MB + temporary465MB plus
headroom); within36GiB cap and15GiB free floor. No deletion this launch.

User explicitly accepted the proposed additional2hours local GPU with "oke".
No cloud, external service, or unlimited resource authorization.

- Previous V2 ceiling21492s; actual charged21229.272528662415s.
- Previous residual262.7274713375855s, carried forward once.
- Added7200s; new total ceiling28692s; available7462.7274713375855s.
- First reservation2130s for GCN-R1 three-epoch pilot; unreserved5332.727471s.
- Charge actual child runtime upon collection, release unused reservation; do not charge parent wall twice. Failed jobs count. No new automatic top-up.
- Storage cap36GiB, free-space floor15GiB unchanged; retain datasets/features and all incumbents. User-authorized unused-run cleanup recorded separately.
- Most new compute remains for measured model improvement and contingent matched controls; no full audit or polling campaign.

## GCN-R1 collection

Child1697.4894688129425s charged once; parent1702.684896s not added.
Used22926.761997475358s; remaining5765.238002524642s. Reservation2130s released.
Reserve830s for current-path three-epoch fusion-only control; unreserved4935.238003s.

## Three-epoch control collection

Charge child646.1910696029663s once; exclude parent651.645420s.
Used23572.953067078324s; remaining5119.046932921676s; previous830s reservation released.
Reserve4260s for fixed GCN-R1 seeds1337/2026; unreserved859.046933s.

## Replication collection / second motivated refinement

Children1698.5940096378326+1719.8979325294495=3418.491942167282s charged once;
parent3430.268857s excluded. Used26991.445009245606s; remaining1700.554990754394s.
Reserve1400s GCN-R2 smoke/pilot, unreserved300.554991s. No automatic top-up.

## GCN-R2 collection (training complete, chain comparison failed)

Charge29.741496086120605+946.8889582157135=976.6304543018341s once;
exclude parent988.221927s. Used27968.07546354744s; remaining723.92453645256s.
Release1400s reservation. No active job or new reservation. A repeat of observed
947s pilot exceeds remaining budget; no automatic top-up or retry.

## Second explicit two-hour top-up — latest user "oke", 2026-09-20

This is a new approval AFTER R2 collection, not a recount of the earlier top-up.
Add7200s to28692s: total ceiling35892s. Used27968.07546354744s unchanged;
available7923.92453645256s. Reserve2430s for C13 smoke/pilot parent hard bound;
unreserved5493.92453645256s. Charge actual child runtime once on collection,
including failed jobs, release unused reservation; exclude parent wall duplication.
No automatic subsequent family/seed launches while waiting for user return.
Storage36GiB/freefloor15GiB unchanged. External public repo/dependency/checkpoint
downloads additionally permitted by user, still within storage/local compute limits.

## C13 collection / first motivated refinement

Charge smoke27.96322464942932 + pilot1808.9829125404358 =1836.946137189865s
once, exclude parent1843.911872625351s. Used29805.021600737306s;
remaining6086.978399262694s. Release previous2430s reservation; reserve2430s
for C13-R1 graphLR1e-5 smoke/pilot, unreserved3656.978399262694s.
No automatic further configs while waiting. Storage estimate revised from4 to3GiB
for adaptive-graph output after observed~2.1GiB total, including temporary save
headroom. Delete only C13 unusedlast~1.024GiB, retain best/provenance; no cap lift.

## C13-R1 collection / C14 allocation

Charge28.279786586761475+1760.4784734249115=1788.758260011673s once;
exclude parent1798.9297533035278s. Used31593.77986074898s;
remaining4298.2201392510215s. Release2430s reservation; reserve2430s C14,
unreserved1868.2201392510215s. No automatic C13 refinements or top-up.
C14 uses native2D geometry/no new assets, estimated18GiBVRAM and3GiBoutput
plus128MiBsmoke. Cap36GiB/floor15GiB unchanged; retireunusedC13-R1last and
originalC04last only, selectedbests retained. Manifest records permanentdeletion.

## C14 collection / one projection-LR contrast

Charge22.851332426071167+1713.533791065216=1736.3851234912872s once;
exclude parent1743.7924284934998s. Used33330.16498424027s,
remaining2561.8350157597342s. Releaseprevious2430s; reserve2430s for C14-R1
smoke+pilot; unreserved131.8350157597342s. No automatictop-up.
Hardcaps90+2300s with parent2430s; estimate~29min from original1713.53s.
Storage3GiB+128MiBsmoke, cap36GiB/floor15GiB unchanged. Retireonlyunused
C14last and C04seed1337last, preserving their selectedbests and allprovenance.

## C14-R1 collected — allocation boundary

Charge28.070672512054443+1702.2846388816833=1730.3553113937378s once;
exclude parent1738.8956809043884s. Used35060.52029563401s;
remaining831.4797043659964s. Release2430s reservation; no activejob/reservation.
No furtherGPUallocation approved. Comparablepilot~1700s cannot fit remaining;
do not auto-top-up or shorten a recipe solely to exhaust budget.
V2artifacts34.530397GiB, disk135GiBfree. No deletion this collection; retainbest.

## Third explicit two-hour top-up — latest user "oke", after C14-R1

New approval, not recounting either previous2026-09-20 top-up. Add7200s to35892s:
ceiling43092s; used35060.52029563401s unchanged; available8031.4797043659964s.
Reserve2660s for C15 joint-bilinear smoke120s/pilot2500s plus launch overhead;
unreserved5371.4797043659964s. Actual child wall charged on collection, including
failed jobs; exclude duplicate parent wall. No subsequent automatic budgetincrease.
New representation-level candidate, not additional C13/C14 tuning. Expected~30–40min,
<=24GiBVRAM,3GiBpilot+128MiBsmoke. Cap36GiB/freefloor15GiB unchanged.
Retire explicit C14-R1last/C04seed2026last only,2,043,265,366bytes (~1.903GiB),
selectedbests/data/provenance retained. Background launch and yield, no polling.

## C15 smoke-gate repair

Chain001 failed before longpilot. Charge smoke00125.119404554367065s once,
exclude parent31.179850101470947s. Used35085.639700188374s;
remaining8006.360299811629s. Release previous2660s, reserve2660s again for
explicit repaired chain002 (smoke3updates, pilot unchanged), unreserved5346.360300s.
No result/gain from failedsmoke. Preserve original report/logs/source and decision.

## C15 collected / structural covariance contrast

Charge successfulsmoke30.920886754989624+pilot1771.4528727531433=
1802.373759508133s once. Parent1813.8293826580048s excluded; failedsmoke001
alreadycharged earlier. Used36888.01345969651s; remaining6203.986540303493s.
Release2660s reservation; reserve2660s for C15-R1 covariance smoke/pilot,
unreserved3543.986540303493s. User's newest message is method-choice permission,
NOT a furtherGPUbudget increase. Cap43092s unchanged.
Cleanup C15unusedlast and older oneepochGCNseed42last only, bests/data retained;
storage36GiB/freefloor15GiB. Expected~30min/<=20GiBVRAM/3GiB+128MiBsmoke.

## C15-R1 collection / SignRep asset admission

Charge27.894715547561646+2115.882576942444=2143.7772924900055s once;
exclude parent2154.0446043014526s. Used39031.79075218651s,
remaining4060.2092478134873s. Prior2660s reservation released.
Conservatively also charge120s failed initialcheckpointdownload (noGPU, curlhard120s):
used39151.79075218651s; remaining3940.2092478134873s. No hidden compute top-up.
Reserve3600s parent wall for C16 download/smoke/conditionalextraction, remaining
unreserved340.2092478134873s. Charge this whole asset-parent wall ONCE on return,
not its childwalls again. Includes network/CPUtime conservatively inlocalallocation.
No automatic retrievaltraining. Download1500s cap, smoke180s, extraction onlyif
conservative smoke-based estimate fits remaining parent. Unadmitted is not success
of full extraction; report prepared_not_admitted. Currentstorage34.84GiB+~.22GiB
source/checkpoint+256MiBcache <36GiB; freefloor15GiB. No deletion thisturn.

## C16 assets collected / explicit extraction admission

Parent prepared_not_admitted terminal; download and strict-load smoke succeeded.
Charge parent242.94235181808472s once, do not add its235.778023s download or
5.882174s smoke separately. Used39394.733104004594s;
remaining3697.266895995403s. Release3600s reservation; reserve3500s for direct
native-window extraction, unreserved197.266895995403s. No compute top-up.
Initial estimate4203s repeated one-time startup/model-load cost across videos.
Use recorded per-video end-to-end timings1.802368s/41clips and2.464748s/64clips:
60s startup allowance + max(per-video seconds/clip)*60023*1.25 =3358.278354s.
This fits3500s. Only2timing samples and other-user GPUload imply uncertainty;
hard timeout remains and complete per-video caches retained for explicit resume.
No timing-only rerun, no model/preprocessing/selector change. Actual extraction
wall charged on collection once; do not charge completed assetparent again.

## C16 extraction collected / paired transfer pilot

Extraction completed1031videos/60023clips, wall1654.8724250793457s chargedonce.
Used41049.60552908394s; remaining2042.3944709160573s. Release3500s reservation.
Cache171376889bytes, finite features/latents and all exact windows/starts verified.
Reserve1800s for one parent: smoke120s, transfer800s, matchedcontrol800s,
80s orchestration headroom. Charge parent once on collection, not childwalls.
No new top-up. TRAIN512 tenepochs160updates/arm, DEV80/160, noTEST.
Expected~8–20minutes combined from historicalGCN step timings, contention caveat;
hard1800s. VRAM estimated<12GiB; compact deltas reserve1GiB/arm, smoke128MiB.
Retire only unused C15-R1last and older oneepochGCN2026last2198712484bytes,
keep selected bests, all three GCN-R1 best/last states, features/data/pretrained.

Pairedparent001 FAILED before training: initial historical score parity gate.
Charge parent21.116541385650635s once (includes smoke17.821748971939087s).
Used41070.72207046959s; remaining2021.2779295304067s; release1800s reservation.
Both160step training arms NOT STARTED. Do not count asset success as model gain.
Logs/source retained. No gate tolerance relaxation or automatic retry launched.

## C16 scoped hook check and explicit pair002 admission

Hookcheck001 failed5.500586271286011s from using TRAIN input unpacker on DEV;
fixed diagnostic uses native DEV visualinputs. Hookcheck002 completed5.97008204460144s:
native repeated / before-after allocation / hooked-unhooked all bitwise identical
on first32DEVitems including one historically differing row. Charge11.470668315887451s.
Used41082.19273878548s; remaining2009.8072612145193s. Reserve1800s pair002,
unreserved209.8072612145193s. No extra top-up, no extra cleanup.
Historical three-stream R1/5/10 all unchanged in failedsmoke; poseMeanR differs1/519.
Cross-process score cause remains unresolved, NOT claimed fixed. V2 permits narrow
changed-path proof rather than mandatory historical fullscore parity. Pair002 uses
fresh fullDEVstep0 in BOTHarms, requires historical recall equality and records score
maxdifferences. Explicit contract revision, not a raised score tolerance or efficacy
result. Same model/loss/LR/subset/160update design; failed logs remain available.
Two longarms still800s each, smoke120s,parent1800s; charge parent once on return.

## User-authorized C16 OOM retry

Pair002 terminalfailed: smoke24.683150s succeeded, pilot171.782354s OOM during
step23 after22complete updates, controlneverstarted. Charge parent207.7139220237732s
ONCE includingchildren/launch overhead. Used41289.906660809254s;
remaining1802.093339190746s; release1800s reservation. No retrievalresult.
User explicitly requests rerun. Reserve1700s:150s failedbatchsmoke +1500s pilot
+50s orchestration. Unreserved102.093339s, no new topup/control allocation assumed.
CPU savedtensor offload keepsB32/native negatives/objective/LR160steps, costs hostRAM
and transfers. Hostavailable49GiB, GPUfree~39GiB atadmission, another extractoractive.
Old firstbatchsmoke peak14.75GB(~13.74GiB), but failedbatchpeak unrecorded;
OOM root resourcecompetition vs batchpeak not proven. New smoke tests exactfailed
batchindices22 twice and gatespeak<=16GiB. Log allocated/reserved/peak/RSSperstep.
Plan runtime~10–25minutes with uncertain contention/offload cost, hard1700s.
V2storage33.366GiB, reserve2GiB for atomic lastcheckpoint replacement, cap36GiB;
no deletion. Save compactresumablelast every16steps. Failclosed, no retryloop.
## Offload smoke timeout repair

Offloadretry001 terminalfailed151.53162622451782s; childtimeout124 at150s.
FullDEV alone took133.293906s; no completedtrainingstep, no OOM observed.
Childrun.json stale running but PIDabsent andparentexit124 are terminalevidence.
Charge parentonce: used41441.43828703377s, remaining1650.5617129662282s.
Release1700s, reserve1600s forretry002: TRAIN-only smoke150s +pilot1400s +50s.
No new topup. Keep allfailedlogs. Remove only redundantfullDEV from memorysmoke;
pilot stillfreshfullDEV0/80/160. Model/loss/batch/LR/selector unchanged.
No polling after startup. Onfailure stop; no autonomousretryloop.
## Offload attention-layout repair

Retry002 failedduringfirstbackward with attentionmask stride4225 notmultiple4;
peakCUDA2953653760bytes, no OOM. Charge parent26.178252935409546s once.
Used41467.61653996918s; remaining1624.3834600308187s. Release1600s; reserve1600s
retry003 (same150s smoke/1400s pilot/50s headroom), unreserved24.383460s.
Select mathSDPA insideoffloadedtraining to avoid fusedmask-layout restriction;
no architecture/objective/negativebatch change; kernelnumerics caveat explicit.
## C16 completed / budget collection under GoalV4

Charge offloadretry003parent1258.4125037193298s ONCE; children46.776684522628784s
smoke and1199.7817311286926s training are included, not addedagain.
Used42726.02904368851s; remaining365.97095631148886s; ceiling43092s unchanged.
Release1600s reservation. No activejob/no newallocation. Userreportedcompletion.
Nextcontrol estimated~1200s from measuredsame-recipepilot; request additional1800s,
pending userapproval. Do not claim request as allocated. No deletion thiscollection.
