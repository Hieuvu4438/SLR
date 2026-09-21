# C17-R1 — moderate decoupling during warm-start adaptation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline
- Origin Mode: run, under user GoalV4 authority
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED (prospective retrieval outcome)
- Version Label: c17_r1_plan_v1

Evidence: C17 pureDCL completed160/allgates, selected77.649326/final77.456647,
below noaux78.323699. Preclip totalgradient median9.314 vscontrol.528; clipped
160/160 vs57/160 at fixed norm1. Pose-only finalmean+2.312139pp, fusedR1 declines.
NPC qmean first/last10=.000520674/.006721790. DCL decreases; numerical failure
is not observed. Neither small q nor clipping proves a causal retrieval bottleneck.

Hypothesis: full decoupling increases pressure on already easy positive pairs and
changes the balance with native branch objectives too much for warm-start GCN/
fusion adaptation. A modest mixture can retain hard-pair CE emphasis and a small
nonvanishing discrimination signal. Alternative: any extra pressure simply overfits
the same training pairs and will not improve full-gallery retrieval.

Only intervention from C17: fused loss = .95*CE+.05*DCL instead of DCL alone;
all pose/RGB/matching losses unchanged. Native4direction/dual_mix=.5 retained.
Weight.05 is a rounded heuristic near historical median totalgradient ratio
.528/9.314=.0567, NOT a common-parameter component-gradient measurement or guarantee
that new gradients/clipping will match control. No automatic alpha search.
At identical row logits, mixture gradient is [.05+.95*q] times DCL gradient,
q=1-p_positive; not a scalar equivalent to reducing the entire learning rate.
Log native CE, pureDCL, effective fusedloss, weight, q distribution/effective q,
and existing preclip norm. A negative total loss is valid and not a stop criterion.

Known DCL is credited in METHOD_DECOUPLED_FUSION_C17.md/LITERATURE.md. Convex mixing
is an engineering refinement, no new novelty/SOTA claim. No teacher, score anchor,
gradient surgery, inference reranking/gate or changes to positives/splits. Same
NO-GO collision scope as C17; not resurrecting a closed transfer recipe.

Fresh native release, not continuing C17. Same native-only TRAIN512 IDs/batches,
fullDEV519, seed42/B32/tenepochs160updates, GCN1e-6/fusion1e-5,
FP32moments, evalBN/dropout except fusion, CPUactivationoffload/mathSDPA.
RGB+pose2D inference unchanged; no new parameters or external features.
Compare matched noaux78.323699 and pureDCL77.649326; global incumbent78.709056
remains fullTRAIN reference, not matched subset evidence. RepeatedDEV one-seed
selection remains exploratory; historical rawscore numeric caveat unchanged.
Evaluate/select0/80/160 under same guardrails, stop mean drop>2pp from release.

Prediction: less clipping and less disruption while preserving some useful DCL
signal. Lower clipping by itself does not validate the method. PROMOTE only for
useful matched ranking gain/trend, then further-data/new-code control confirmation.
If no matched gain with reasonable update scales, DEFER this fused-DCL recipe
family in this tranche. If severe technical/scaling issues persist, inspect the
specific evidence before any further refinement; no automatic sweep/queue.

22 scoped CPU tests pass, including blend endpoints, row-gradient formula and
native branch preservation; py_compile passes. Actual native CE reconstruction,
GCN/fusion update, frozen buffers and checkpoint gates run inside pilot.
Code hashes include shared C17 validator/config as well as blend entrypoint.

Run v4-c17-dcl-blend005-001 -> seds-fused-dcl-blend005-offload-001.
Command: /home/haipd/miniconda3/envs/seds/bin/python -u
research/slret_goal_v2/tools/run_c17_dcl_blend_v4.py --launch
Estimate23–27min, hard1800s/outer1850s. Reserve1850 of13126.581830s remaining.
User20GB decimal;16GiB allocator/19GB owned-descendant sampled safetytrip.
Storage37.296GiB+1.25GiB saveheadroom; selfsubcap38->40GiB perGoalV4section8,
free127.21GiB/floor15GiB. No deletion/download/re-extraction.
Retain compact best/last, optimizer/RNG/config/logs/source IDs; no tested resume
entrypoint yet. One startup check then WAITING_FOR_USER; no model polling.

Launched startUnix1789913035.7929072; supervisor3767293/torchrun3767301/trainer3767345
alive at sole startup check. InitialDEV519 entered, weight.05/native_subset/signrepNone
recorded, no immediate error. Initial telemetry0 is before first GPU sample, not a
validated peak. WAITING_FOR_USER, no agent polling or automatic next experiment.

Collected2026-09-20: TIMED_OUT during finalDEV, not a training-step failure.
last.pt contains next_batch_index160/all160 logged updates; stale report159.
Intermediate80mean77.649326, finalDEV missing. User then explicitly closed C17.
Do not run recovery or more refinements. Recovery source was prepared but never
launched, costs0 and has no active reservation. Retain all evidence/weights;
no scientific final-performance claim from incomplete evaluation. Next C18 nativeB64.
