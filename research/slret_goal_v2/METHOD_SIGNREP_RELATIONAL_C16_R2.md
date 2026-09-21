# C16-R2 — final relational-loss strength contrast

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run, inline; user GoalV4 authorizes implementation
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED (new run not yet measured; no independent rerun)
- Version Label: c16_r2_plan_v1

Question: does the RKD-D constraint improve retrieval when given more weight,
or is the transferred geometry unhelpful to native pose retrieval in this setting?

Evidence: R1 completed all160 updates/gates, mean78.227360 <noaux78.323699,
while pointwise78.034682 is lower still. R1 first/last10 relationloss means
.023042823/.014856976, last10 weighted=.001485698 vs native=.047221867 (~3.15%).
The fixed.1 coefficient was carried from pointwise loss whose final10mean was
.546527. These are different objectives/scales; equal coefficients do not imply
equal influence. Auxiliary edgegradient is nonzero (~4.98e-5 last10), but native
gradient on that edge has not been measured. Do not compare it to global parameter
gradientnorm, or claim established interference/underweighting. Reducing auxiliary
loss alone is not retrieval evidence. Lack of useful teacher geometry is a strong
alternative explanation; projective head absorption is also not ruled out.

Intervention: native7 losses + 1.0*RKD-D instead of +.1*RKD-D. This is the only
optimization change. At identical tensors, auxiliary edgegradient scales by10;
later optimizer/model trajectories need not. No claimed 10x encoder update or
novel method. Known RKD-D donor, independent equation implementation, license/
source/read scope and NO-GO collision remain in METHOD_SIGNREP_RELATIONAL_C16_R1.md.
No new source download, weights, annotations or auxiliary inference stream.

Fixed recipe: native SEDS release; TRAIN512 same cached IDs/windows, fullDEV519;
seed42, B32, tenepochs160updates, encoder1e-6/fusion1e-5/head1e-4,
FP32moments, activationoffload/mathSDPA; RGB+pose2D inference unchanged.
Fresh initialization (not continue R1), evaluate/select0/80/160 with unchanged
guardrails; also compare fixed160. Existing noaux78.323699 and R1.1weight78.227360
are matched recipe controls. Global incumbent78.709056 is fullTRAIN, not matched.
Historical score-parity uncertainty remains; initial fullDEV recalls checked,
not an assertion of bitwise rawscore equality. TEST not loaded or used.

Prediction: if constraint strength was limiting useful transfer, weight1 may
produce separation above noaux; if irrelevant/conflicting, loss may still drop
without ranking benefit or with degradation. A new gain remains exploratory and
search-conditioned, requiring confirmation before attribution/generalization.
PROMOTE only for meaningful matched gain/trend, not merely above initialization.
If tie/lower at the selection and final metrics, DEFER direct C16 pointwise/RKD-D
recipes in this tranche, no further coefficient/LR/horizon sweep. Preserve data
and pretrained for substantively different hypotheses. This is second scientific
refinement after structural R1, not an automatic adaptive queue.

Run: v4-c16-relation-weight1-001 -> seds-signrep-relational-weight1-offload-001.
Command: /home/haipd/miniconda3/envs/seds/bin/python -u
research/slret_goal_v2/tools/run_c16_relation_weight_v4.py --launch
Estimate20–25min, hard1800s/outer1850s, reserve1850 from15761.226179s remaining.
20GB decimal user cap;16GiB allocator, sampled19GB safetytrip on owned descendants.
Storage35.759GiB +1.25GiB <38GiB, free128.79GiB >15GiB; no cleanup needed.
Keep best/last/config/log/snapshot; compact last contains optimizer/RNG but a
tested resume entrypoint is not implemented. No promise of automatic recovery.

14 scoped CPU tests pass (loss/mask/gradient/pointwise regression, process-tree,
registered config/missing checkpoint/update rejection); py_compile passes.
Real gradient/update/buffer gates run inside training. Successful synthetic tests
do not establish retrieval efficacy. Launch, one startup check, WAITING_FOR_USER.

Launched startUnix1789909234.0524302; supervisor3688727, torchrun3688728,
trainer3688784 all alive at sole startup check. CandidateC16_R2 and weight1.0
present in trainer report; initial fullDEV entered, no immediate error.
Initial status precedes GPU sample (0 not used as measured peak). Watchdog will
record owned-process GPU memory and require nonzero telemetry before success.
WAITING_FOR_USER; no adaptive follow-on or agent polling.

Collected after user return2026-09-20: COMPLETED160, selected/final78.323699,
tie matched noaux78.323699; R5/10 unchanged. R2 improves one T2V hit and loses
one V2T hit relative to noaux. Gates pass; loss decreased and auxiliary edge
gradient is ~10x R1, but no primary-metric increment. DEFER direct C16transfer
recipes in this tranche as registered; no extra strength/LR/horizon sweep.
Incumbent78.709056 unchanged. RESULTS.md and budget ledger contain full collection.
