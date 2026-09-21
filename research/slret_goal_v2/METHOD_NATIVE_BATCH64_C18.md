# C18 — native contrastive batch-size contrast

LATEST: retry002 COMPLETED80. Selected40mean78.227360 vsB32control78.323699,
final80mean78.131021. Gates passed, observed17.629GB. DEFER current B64 recipe;
not a universal negative-count finding (optimizer steps differ). RESULTS.md has
directional metrics/hash/resources. No new incumbent, no additional B64 run queued.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline
- Origin Mode: run, GoalV4 implementation authority
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED (prospective pilot)
- Version Label: c18_plan_v1

## Infrastructure retry002 (same scientific plan)

001 FAILED: nvidia-smi query timed out at5s; supervisor cleanup stopped workers.
Initial DEV completed but no train-step record or resumable checkpoint exists.
No evidence of OOM or a negative retrieval outcome. Charge239.5920069217682s once.
Retry v4-c18-batch64-002 / seds-native-batch64-offload-002 starts from release;
only run IDs and GPU-query deadline5->20s change. Missing telemetry remains fatal,
no stale/zero fallback or automatic retry loop. Query times/last sample recorded;
20s blind window is possible, sampled watchdog is not a hardware memory partition.
User20GBdecimal, allocator16GiB/19GB trip unchanged. No C17 reopening.
Command: /home/haipd/miniconda3/envs/seds/bin/python -u
research/slret_goal_v2/tools/run_c18_batch64_retry_v4.py --launch
Hard2400s/outer2450s, reserve2450s of11083.931395993495s; estimate25–35min.
Artifacts40863137187bytes+1.25GiB reserve <40GiB; free135714144256bytes/floor15GiB.
14 scoped CPU tests pass, including exact scientific recipe equivalence to001,
resource ownership and timeout/N/A failclosed. Syntax compilation passes.
One startup check then WAITING_FOR_USER; user return authorizes next collection.
Retry002 launched startUnix1789916170.5989223; supervisor3834909 and torchrun3834911
alive at sole startup check, initialDEV519 entered, no immediate error. Training
updates/peak not yet validated. STATE=WAITING_FOR_USER; no polling after handoff.

## Original plan and launch001 history

User explicitly closes C17. No DCL, convex DCL blend, final-eval recovery or
teacher supervision in this job. Existing native CE and native RGB/pose2D retained.

Hypothesis: weak contrastive pressure may partly reflect the random negative pool
being too small, rather than a need to remove CE's positive-dependent attenuation.
Increase negatives by changing which examples compete under the same native CE.
Prior native B32 fusedCE first10mean.000613 is a motivation, not proof of a
hard-negative deficit or a semantic error. C17's large gradients did not improve
fusedR1 and its family is user-closed. No attempt to rescue it under a new name.

Intervention: physical TRAIN batch32->64, native loss all components unchanged.
No hard mining, duplicate relabeling, gallery memory/risk, cross-batch stale queue,
new score/gate, teacher or gradient surgery. No collision with PMGR/SSSC/ELSC or
closed CiCo scalar pooling/temporal variants: only native batch composition changes.
Native independent-query inference/architecture/parameters unchanged.

Keep release checkpoint, TRAIN512 exact IDs/order, fullDEV519, seed42/tenepochs,
GCN1e-6/fusion1e-5, FP32moments, BN/dropout eval except fusion. CPU activation
offload/mathSDPA reuses C16 resource implementation; NOT GradCache. Increasing
batch size is a known engineering control, not a novelty contribution. Prior
GradCache paper remains an alternate implementation lead, not adopted code here.

Same5120sample exposures, but80updates instead of160. Batch64 is exactly the
concatenation of two consecutive historical B32 batches. Thus each sample has63
other in-batch examples instead of31; supervision/positive mapping unchanged.
Candidate eval0/40/80 aligns in data exposure with control0/80/160. Native schedule
spans80optimizer steps rather than160; LR base unchanged. Number of updates,
gradient averaging and schedule discretization are explicit coupled differences.
Do not claim isolated negative-count causality from this run. A promising result
requires stronger update/exposure-matched confirmation, including search budget.

Control: existing native noaux B32 selected/final78.323699. Its unused teacherhead/
cache reader was already omitted by the native_subset interface; no external
features loaded. Global fullTRAIN incumbent78.709056 remains, not subset-matched.
No full baseline audit or re-extraction. No TEST load/tuning; repeatedDEV selector.

Select max meanR1 among0/40/80 under existing per-direction guardrail; early stop
if mean falls>2pp below release. Full initialDEV recalls and source/base/data order
checked; rawscore historical numeric uncertainty remains disclosed. First2steps
check finite gradients/GCN+fusion updates and frozen buffers. Checkpoint roundtrip
verified with nativeGCN+fusion compact delta. Failure/OOM is a resource failure,
not an empirical no-gain finding; no automatic retry/new batch size.

PROMOTE if worthwhile matched-exposure gain/trend supports fuller-data confirmation;
REFINE only from actual loss/dev/memory evidence; DROP if adequate learning shows
no useful signal. This tests baseline headroom, not sufficient novelty or SOTA.
Separate longer-term 3D/hand representation leads remain in queue; no broad donor
download merely because transfer failed. Do not rename this GradCache or a new loss.

Run v4-c18-batch64-001 -> seds-native-batch64-offload-001.
Command: /home/haipd/miniconda3/envs/seds/bin/python -u
research/slret_goal_v2/tools/run_c18_batch64_v4.py --launch
Estimate25–35min, hard2400s/outer2450s, reserve2450 from11323.523403s remaining.
20GBdecimal user cap;16GiB allocator,19GB sampled owned-descendant safetytrip.
B64 peak not established in advance; first updates/resource monitor failclosed.
Storage~37.99GiB+1.25GiB reserve <40GiB; free~126.49GiB/floor15GiB, no deletion.
Keep best/last/log/config/source IDs; resume state retained, general training resume
entrypoint not implemented. Logs/status/summary local, no model monitoring.
11 scoped CPU tests pass: batch-order/exposure including actual control artifact,
process ownership/memory failure checks and existing recipe rejection tests.
py_compile passes. One startup check then WAITING_FOR_USER; no adaptive nextjob.

Launched startUnix1789915311.2623327; supervisor3815823/torchrun3815824/trainer3815882
alive at sole startup check. InitialDEV519 entered, native/B64/signrepNone config
confirmed. No immediate error. InitialGPU0 precedes sampling; B64 training peak
not yet measured. WAITING_FOR_USER, no model polling. C17 recovery never launched.
