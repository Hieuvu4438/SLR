# GCN-R1 — three-epoch refinement of the clean GCN lead

## Material Passport / prospective card — 2026-09-20

ARS experiment-agent/run, inline. User approved +7200s local GPU with "oke".

- Evidence: clean GCN beats matched fusion-only for all3 seeds; selected/final222 means agree per seed. Seed42 improves from111 to222. Added C11/C12 capacity did not add primary-metric gain.
- Hypothesis: the proven GCN adaptation has learning headroom beyond one pass before its schedule decays; this is unproven, not an underfitting diagnosis.
- Intervention: change native horizon from1 to3epochs (222 to666updates), with native warmup/cosine stretched accordingly. No new module/loss/input, no C11/C12 combination. Refinement1 in this new allocation, not a novel family.
- Release initialization, seed42 (not selected-best1337), TRAIN7096/B32; GCN1e-6/fusion1e-5, native objectives/FP32 moments; all other weights and encoder BN/dropout frozen. Original pose2D/RGB retained, no new 3D extraction.
- Reuse unchanged clean GCN smoke and inference anchor; first2 actual updates still check expected parameter changes and fixed buffers. Only orchestration/labels changed; no baseline replay.
- DEV519 at0(reused),111,222,444,666; unchanged primary mean bidirectionalR1, each direction>=reference-.5pp, initialization eligible, earlystop mean-drop>2pp.
- Compare reference77.552987, prior seed42 recipe78.034682, globalbest1337 78.516378. Report fixed common steps and all new evaluations; longer schedule differs from step1, and extra selection opportunities prevent attributing differences solely to extra late updates.
- If lead, use a matched3epoch fusion-only control (reuse historical one only if exact recipe/data/order matches), then seed replication as budget permits. Do not call this a new mechanism or SOTA.
- If no increment or late deterioration, retain current incumbent and move to another hypothesis; no blind horizon/LR sweep. At most3 motivated refinements per family/tranche.
- Collision: ordinary native GCN weight adaptation already has a positive measured lead, unlike frozen residual-readout horizon rescue or augmentation changes in [V1 registry](../slret_goal/NO_GO_REGISTRY.md). No teacher, residual scorer, lexical support, C06 or TEST tuning.
- Estimate~1650–1850s, peak~15GiB VRAM; child hard2100s,parent2130s. Reserve2130 of7462.727471s; cap36GiB and free15GiB floor unchanged. Retire only deferred C12 last.pt, preserving its best and logs, before4GiB output reservation.
- Command: `tools/run_masked_pair.py --long-gcn-pilot` through bounded launcher. Chain `gcn-horizon3-pilot-001`, child `seds-gcn-horizon3-001`. Source hashes fixed throughout; failures stop, no automatic retry/next family.
- Run background with persistent logs, verify initial updates once, yield until actual user completion. TEST historically exposed but not used; DEV repeatedly selected, novelty unresolved.

Execution: seven focused existing policy/gradient/LR/DDP tests passed0.461s,
changed runners compile and chain CLI parses. Launched timeoutPID2533478 with
2130s cap; main log `artifacts/slret_goal_v2/gcn-horizon3-pilot-001/seds-gcn-horizon3-001.log`.
C12 last.pt permanently removed1.251984GiB, best and logs retained; manifest
`artifacts/slret_goal_v2/storage-prune-c12-last-20260920-001/manifest.json`.
Initial live check:5updates, expected parameter updates/frozen-buffer gates
passed. Retrieval outcome pending; no monitoring loop after handoff.

## Collected outcome / matched control registration — 2026-09-20

Completed666updates1697.489469s, all gates passed. MeanR1 at111/222/444/666:
77.938343/78.420039/78.516378/78.612717. Selected666 T77.842004/V79.383430,
+1.059730pp release, +.578035pp same-seed1epoch, +.096339pp old globalbest.
Provisional new best; retain all prior leads. Gain partly appears before222 under
the stretched schedule; no isolated extra-update effect or independent/SOTA claim.

Next registered control: same release/fullTRAIN7096/seed42/B32/3epochs666,
native loss, FP32 moments, inactive masked-control attachment, fusion1e-5,
same native schedule/evaluation111/222/444/666, selector and earlystopdrop2pp.
Only change: GCN weights frozen; all BN/dropout remain eval. No C11/C12 module.
Reuse existing clean-fusion update tests/smoke and reference; actual step2 must
show only fusion updated and all frozen buffers unchanged. Exact666batch hash
and base/data must match GCN-R1. No fresh input extraction or TEST use.
Historical C04 control has matching batch hash/base/data but older optimizer
partition and attachment/config; do not silently certify full equivalence.
This current-path control tests whether GCN still adds value at3epochs, not
whether three epochs are an independently novel technique.
Driver --long-fusion-control, chain gcn-horizon3-control-001,
child seds-gcn-horizon3-fusion-control-001; estimate600–700s, child cap800s,
parent830s ofremaining5765.238003s. <=4GiB reserved storage, peak<8GiB VRAM.
Retire only deferred C11 last.pt first; preserve its best, all incumbents/logs.
Positive contrast -> freeze recipe and replicate seeds1337/2026 as budget allows;
no gain -> retain measured checkpoint but reassess attribution before further work.
Background/log/initial check/yield; no automatic next jobs or monitoring loop.

Control launched timeoutPID2575110, parent830s; targeted fusion-only update test
passed0.436s. C11 last.pt permanently removed1.023432GiB, best/logs retained;
manifest artifacts/slret_goal_v2/storage-prune-c11-last-20260920-001/manifest.json.
Initial check: running15updates, only_fusion_updated and masked_update_checks
passed; incumbent78.612717 correctly recorded. Yield until user completion.

## Control collected / fixed-recipe replication registration — 2026-09-20

Control completed666updates646.191070s; selected init77.552987, final same mean,
T76.685934/V78.420039. GCN-R1 +1.059730pp selected and fixed666. All matching
and update/buffer checks passed, no early stop. Historical control metrics match
all recorded steps; no more unchanged control replay needed. Full table RESULTS.md.

Register GCN-R1 seeds1337/2026, each release initialization/3epochs666/B32,
GCN1e-6/fusion1e-5/native loss and FP32 moments, same frozen BN/dropout,
inactive masked-control attachment. No new hyperparameters/modules/horizon tuning.
DEV0(reused)/111/222/444/666, unchanged selector/guardrail/earlystopdrop2pp;
reuse smoke, first2 actual updates verify GCN/fusion changes and fixed buffers.
Compare all3 selected and fixed666 scores versus release and same-seed1epoch
GCN; disclose unequal horizons and repeated DEV. Do not pair1337/2026 with
seed42 fusion control for a causal claim. Later matched controls only if needed
and within budget; no TEST used. Novelty/generalization/SOTA remain unresolved.
Driver --long-gcn-replication, chain gcn-horizon3-replication-001, children
seds-gcn-horizon3-seed1337-001 / seds-gcn-horizon3-seed2026-001. Source hashes
fixed between runs, each cap2100s, parent4260s ofremaining5119.046933s.
Expected3300–3600s total, peak~15GiB VRAM, about4.1GiB new outputs; keep4GiB
per-run conservative reservation. Retire four explicit unused last.pt states
from C08 treatment and completed fusion controls before launch, preserving ALL
selected checkpoints/data/logs. Cleanup manifest is separate. Cap36GiB and
free floor15GiB remain; estimated post-cleanup29.4GiB fits both sequential jobs.
Run both registered seeds regardless of first seed DEV score unless runtime
gates fail. No automatic retry/tuning/third job. Background/initial check/yield.

Replication launched timeoutPID2597273 with4260s cap. Existing clean-GCN/LR
test passed0.350s; orchestration compiles/CLI parses. Four listed last.pt states
permanently removed3.663238GiB; all selected checkpoints retained, manifest
artifacts/slret_goal_v2/storage-prune-gcn-replication-20260920-001/manifest.json.
Initial live check: seed1337 at3updates, update/frozen-buffer gates passed;
seed2026 queued. No further polling until actual user completion notification.

## Replication collected — 2026-09-20

Both new seeds completed666, no earlystop, all gates passed. Selected1337step222
78.709056, selected2026step22278.516378; final66678.420039/78.034682.
All3 selected mean78.612717, sampleSD.096339, mean+.385356pp vs1epoch.
Fixed666mean78.355812, two seeds below their1epoch endpoints; do not claim
uniform benefit from late updates. Full metrics/provenance in RESULTS.md.
New best1337step222 retained along with all prior leads. Charge3418.491942s
once, remaining1700.554991s. Next motivated refinement2 is GCN freeze-after222,
not longer training or a blind sweep; METHOD_GCN_FREEZE_R2.md. No TEST tuning.
