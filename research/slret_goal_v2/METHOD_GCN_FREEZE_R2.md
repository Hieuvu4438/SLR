# GCN-R2 — freeze learned GCN after one pass, continue fusion

## Material Passport / prospective card — 2026-09-20

ARS experiment-agent/run, inline; refinement2 in current GCN allocation.

- Evidence: GCN-R1 selected improves all3 seeds over1epoch;1337/2026 peak222 then lose .289017/.481696pp by666, while42 gains .192678pp after222. Late drift is a hypothesis, not established cause.
- Test: same release/seed42/fullTRAIN7096/B32/3epoch666 schedule, GCN1e-6 and fusion1e-5 through222; then freeze GCN weights, train only fusion through666. Fixed BN/dropout, same native loss and pose2D/RGB; no new branch/augmentation/teacher/3D extraction.
- Rewrap native DDP at223, retain the SAME optimizer and fusion moments/step counters/schedule. Frozen GCN gradients cleared to None; skip frozen updates without resetting fusion or changing RNG/data order.
- Matched control: completed GCN-R1 seed42, mean78.612717 final666, same initial222 training. Globalbest1337mean78.709056 also reported, not a causal cross-seed comparator.
- DEV0(reuse)/111/222/444/666, original primary meanR1/selector/directional reference-.5pp guardrail/earlystopdrop2pp. Initial222 metrics should match R1; later comparisons test the freeze schedule, not a novel module.
- Preserve current six GCN checkpoints and all prior leads. If no increment, defer this freeze schedule; no blind freeze-step/LR sweep. If promising, further matched-seed confirmation requires enough remaining allocation.
- Distinct from closed RPCA teacher/protected gradients and C02 fusion-first/unfreeze-upper: no projection, distillation or residual scorer. Freeze already-improved native GCN late; ordinary optimization refinement, novelty unresolved.
- Targeted CPU DDP2+2steps test verifies transition, unchanged GCN and preserved/advancing fusion moments. GPU smoke4steps switches after2; fullDEV smoke parity checks new orchestration, then real run requires transition and final immutability gates.
- Driver `tools/run_masked_pair.py --freeze-gcn-pilot`; chain gcn-freeze-pilot-001, children seds-gcn-freeze-smoke-001 / seds-gcn-freeze-001.
- Estimated~1050–1200s including smoke, peak~15GiB VRAM, ~2.1GiB outputs with4GiB conservative reservation. Smoke cap60s, pilot1300s, parent1400s <= remaining1700.554991s.
- Retire only two old fusion-control last.pt states (selected release/best remain), manifest storage-prune-gcn-freeze-20260920-001; cap36GiB/free floor15GiB unchanged. No selected models or data removed.
- Source hashes fixed; no retries or automatic next pilot. Background/log/initial smoke check/yield to user. Repeated DEV exposure, no TEST loaded; no SOTA/independent-generalization claim.

Execution: CPU DDP2+2 transition test passed0.416s; runners compile.
Launched timeoutPID2819816, parent1400s. Deleted the two scoped old fusion-control
last states1.757311GiB permanently; all selected checkpoints and logs retained.
GPU smoke and pilot results pending at launch; charge both upon collection.
GPU smoke completed4steps29.741496s; transition preserved fusion step counters,
only fusion updated after freeze, GCN stayed exact. Pilot child2820709 started.
Do not treat smoke as efficacy; background handoff, collect on user completion.

## Collected result / decision — 2026-09-20

Pilot completed666steps946.888958s; smoke29.741496s. Parent FAILED post-training
exact pre-freeze comparison, although both children exit0 and all actual
freeze/update/frozen-buffer checks pass. Preserve failure/provenance unchanged.
Selected222mean78.420039 occurs before intervention; final66678.034682.
No new lead: selected -.192678pp vs R1 seed42 and -.289017pp globalbest.
Defer the recipe; no proof that freezing alone caused the loss because exact
pre-intervention equivalence failed. Full metric table and caveat in RESULTS.md.

Scoped saved-output comparison:111/222 R1/5/10 match, but MeanR differs +1/519
for T2V at111 and V2T at222. Fusion score maxabs differences .006521225 and
.006296158; not mere rounding. Loss/grad identical at1/2, differ by10. Same
native recipe except output_dir, batch/data/base/optimizer groups; source freeze
path not active until223. Cause unresolved, no invented nondeterminism diagnosis,
no relaxed post-hoc tolerance or baseline audit campaign. Failure is a limitation
of exact attribution, not a training crash or a positive result needing rescue.
Total charge976.630454s; remaining723.924536s cannot cover repeat947s pilot.
No new job/automatic rerun. Keep R1 best and all seeds; additional bounded local
allocation needed for a comparable new refinement or broader method tranche.
