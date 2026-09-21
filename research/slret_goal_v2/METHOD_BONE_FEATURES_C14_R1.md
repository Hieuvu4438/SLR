# C14-R1 — stronger learning of the zero-init bone projection

- Material Passport: ARS experiment-agent/run, prospective2026-09-20; user V2 authority. No TEST/historicaltesterror access.
- Evidence: C14 complete666, best78.612717 tiesR1seed42; fixed444+.096339pp, otherthreepoints-.096339pp. Boneprojectionnorm at660hand.01046447/body.00709206, active but limitedweights. No proof of undertraining or beneficialgeometry.
- Hypothesis: initialboneLR1e-5 may constrain zero-init feature learning; increase only thisLR to1e-4. This is one motivated contrast, not a newmethod or knownsolution.
- Same rawbonevector/length/direction5→128 operator, sameparentmaps/masks, nativeXY/RGB/fixedgraphs retained. No C13edges, C06temporalbranch, teacher/gate/scorer.
- Releaseinit/seed42/B32/TRAIN7096/3epochs666, GCN1e-6/fusion1e-5, native losses/FP32moments/BN/dropoutfixed. NewboneLR1e-4 only; do not resume C14best or optimizer.
- DEV0/111/222/444/666, same meanR1selector includinginit/directional-.5ppguardrail, earlymean-drop>2pp. Compare selected/fixed to originalC14 and R1seed42; globalbest78.709056 retained.
- Unit tests verify newLRpartition, original geometry/gradient/serialization tests retained. Freshtwo-stepGPU checks actualbone/GCN/fusion updates and frozenbuffers; fullDEVzero-init once in smoke because runnerchanged; pilotinherits exactproof.
- All historicalcontrols reused, not replayed. R2 reproducibility caveat remains; anypositivelead requires replication and capacitymatchedrawXYprojection before geometry-specific attribution, subjecttoadditionalbudget.
- Expected~29min/<18GiBVRAM, caps90ssmoke+2300spilot, parent2430s. Remaining2561.835016s, unreserved131.835016s; no automatictop-up/cloud.
- Storage3GiBpilot+128MiBsmoke after retirement of unusedC14last/C04seed1337last; allselectedbests/data/logs kept. Cap36GiB/freefloor15GiB unchanged.
- Command: `python research/slret_goal_v2/tools/run_adaptive_graph.py --bone-features --faster-bone`; chain `c14-bone-features-lr1e4-pilot-001`, children `seds-bone-features-lr1e4-smoke-001`, `seds-bone-features-lr1e4-001`.
- No promotion for release-onlygain/tie. If no usefulincrement, defercurrentC14specification, no extraLR/horizon sweep. Improvement/novelty/generalization/SOTA unresolved.
- Backgroundpersistentlogs, verify startup once then yield; user signalscompletion, no monitoringloop or automaticnextjob.

## Startup outcome

12 scopedCPUtests pass. Smokecompleted2steps28.0706725121s: fullDEVzero-init
parity, boneLR1e-4/1280params, bothbone/nativeGCN/fusion updates and fixedbuffers
pass. Pilot verifiedalive atstep5, PID3219568; inheritedproof/gatespass, noerror.
Bothchildren to be charged once on userreturn; smoke not yet in33330.164984s.
Cleanupdeleted2043265302bytes/1.902939GiB permanently; bothselectedbests retained.
Log `artifacts/slret_goal_v2/c14-bone-features-lr1e4-pilot-001/seds-bone-features-lr1e4-001.log`.

## Collected — defer

Completed666steps1702.284639s, allgatespass/noearlystop; selected222mean78.131021,
T77.263969/V78.998073. OriginalC14/R1seed42 selected78.612717, globalbest78.709056.
All four fusedDEV points worse than originalC14. Norm660hand.09670927/body.05990369
confirms stronger weightlearning, not improvedretrieval. No furtherLR/horizon sweep.
Keep bothC14bests/provenance, no promotion. Children1730.355311s charged once,
remaining831.479704s, no newjob/reservation. Full numbers in RESULTS.md.
