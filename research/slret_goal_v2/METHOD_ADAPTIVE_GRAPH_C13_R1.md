# C13-R1 — slower topology adaptation, registered2026-09-20

- Material Passport: ARS experiment-agent/run; V2 goal plus user continuation.
- Evidence: C13 standalonepose rises60.501→64.644, but bestfusion77.746 trails matchedrecipeR1seed42 by.867pp; finalfusion77.553 returns to release. Finite gradients and all6B updates, no runtime defect observed.
- Hypothesis: learned topology may move too quickly relative to pretrained fusion adaptation. Lower graphLR could retain native useful relationships while still learning; source of mismatch unproven.
- Single change: graph B LR1e-4→1e-5 (first refinement). Keep GCN1e-6/fusion1e-5, native losses and all trainable subsets, no freezing or new gates.
- Release initialization, not C13 or bestseed checkpoint; seed42/B32/TRAIN7096/epochs3/666nativeupdates, onlinepose+cachedRGB, BN/dropoutfixed; original2D/RGB retained.
- DEV0/111/222/444/666; selector includesinit, meanR1 primary, eachdirection>=release-.5pp; earlymean drop>2pp. No TEST use.
- Compare selected and fixed steps with originalC13 and historicalR1seed42; retain globalbest78.709056. Historical replay mismatch caveat remains; one seed exploratory.
- Reuse same adaptive graph operator. Scoped CPU LR partition/regression tests plus two-stepGPU LR/update smoke; one fullDEV check in freshsmoke because trainer/rate plumbing changed, pilotinherits exactsource/base/data/configproof.
- Resource: estimated~31min, smoke90/pilot2300s caps, parent2430s. Remaining6086.978399s before run; unreserved3656.978399s. EstimatedVRAM18GiB based on C13actual17.307GiB.
- Storage: 3GiB pilot +128MiBsmoke after actualC13~2.1GiB; remove onlyC13last1.024GiB, keep selectedbest/logs. Cap36GiB/floor15GiB unchanged.
- Positive lead vs matchedR1 merits replication; release-only gain does not. If no useful increment, defer this LR/topology recipe and move family, not another blind LR/horizon sweep.
- Command: `python research/slret_goal_v2/tools/run_adaptive_graph.py --slower-graph`; chain `c13-adaptive-graph-lr1e5-pilot-001`; children `seds-adaptive-graph-lr1e5-smoke-001` then `seds-adaptive-graph-lr1e5-001`.
- Background with persistent logs; verify startup then yield until user returns. No model download or external spend required. Borrowed adaptive graph, novelty unresolved.

## Startup

Nine CPU tests pass. Smoke completed2steps28.2797865868s: fullDEVzero-init
parity, graphLR1e-5, allgraph/nativeGCN/fusion updates and frozenbuffers pass.
Pilot observedalive atstep4, PID2918875, inheritedparity passed, noerror.
Both child runtimes to be charged together on user return; smoke not yet counted.
Log `artifacts/slret_goal_v2/c13-adaptive-graph-lr1e5-pilot-001/seds-adaptive-graph-lr1e5-001.log`.

## Collected / deferred

Completed666steps1760.478473s, noearlystop, allgatespass. Selected111mean78.131021
(222ties), final77.842004. +.385356pporiginalC13 but-.481696ppR1seed42 and
-.578035ppglobalbest; no promotion. Defer topology after this one refinement,
no extraLR/horizon sweep. Best retained; unusedlast deleted with C14manifest.
Children1788.758260s charged once, remaining4298.220139s. See RESULTS.md.
