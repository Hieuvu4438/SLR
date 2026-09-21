# C17 — fused-only decoupled contrastive adaptation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline
- Origin Mode: run; GoalV4 authorizes candidate implementation
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED (prospective retrieval outcome)
- Version Label: c17_plan_v1

## Hypothesis and evidence

After C16 transfer is deferred, target optimization rather than another feature
supervisor. Matched native TRAIN512 control first/last10 fused CEmeans are
.0006133417/.0092848049 despite meanDEV78.323699 < incumbent78.709056. Low training
loss may reflect easy in-batch classification while full-gallery confusers remain.
It may also reflect an already good initialization for which further discrimination
overfits. TRAIN-only aggregate losses do not identify linguistic errors or prove
that attenuation causes DEV errors. No new error audit or test inspection.

Claim under test: under warm-start, small-batch SEDS adaptation, removing the
positive-dependent attenuation from fused retrieval training may yield more useful
ranking updates, without another representation or inference cost. If stronger
updates merely overfit/reshape the same ranking, abandon or refine based on data.

## Borrowed mechanism, adaptation and scope

Yeh et al., Decoupled Contrastive Learning, ECCV2022; primary method Eq4–5,
experiments and limitations read from arXiv2110.06848v2 (2021-10-23), venue
confirmed by ECVA final paper. See LITERATURE.md for URLs/search scope.
Independent equation implementation, no copied third-party code/checkpoint.
The inspected raminnakhli repository is a third-party implementation, NOT
verified official author code; no clone/import. No donor dependencies needed.

For square paired native logits S, DCL = mean_i[logsumexp_{j!=i} S_ij - S_ii].
Replace only fused CE terms in native modeling.py forward, preserving dual_mix
and both I2T/T2I logit constructions/transposes. Keep pose CE, RGB CE and matching
term exactly; no DCLW, new temperature, hard mining, relevance relabeling or queue.
Unlike donor two-view SimCLR, negatives here are existing cross-modal batch pairs.
This is an adaptation of known DCL, not a new loss or established novelty/SOTA.

Native scores already have temperature/logit_scale. Do not apply a second one.
For each row, grad(CE) = (1-p_positive)*grad(DCL). Log q=1-p_positive stably using
expm1(log_softmax diagonal); model dual_mix=.5, equal four-direction summaries.
Record q mean/quantiles plus native fused CE and DCL each step. Negative DCL values
are valid; clipping/finite-gradient and DEV checks, not loss sign, govern safety.
No manual gradient surgery, score-reference anchor or baseline protection.

Intervention is training-only: a forward wrapper captures existing fused logits,
replaces loss components and clears graph references. Inference methods return
native outputs; no added parameters. This does not replace RGB/pose2D with 3D.
NO-GO collision: not RPCA (no teacher/protection), PMGR (no grouped/all-gallery risk),
SSSC/ELSC (no selected support), C05label smoothing, CiCo scalar inference pooling,
augmentation rescue or closed temporal variants. Changes native TRAIN objective.

## Pilot / control / decision

Native SEDS release; exact historical TRAIN512 IDs, batches and fullDEV519; seed42,
B32, tenepochs160updates; GCN1e-6/fusion1e-5, FP32moments, evalBN/dropout except
fusion, activationoffload/mathSDPA. Only GCN+fusion train; text/RGB encoders frozen.
NativeSubset reads ID metadata from C16 noaux report, NOT SignRep features/targets;
unused transfer head omitted. Both produce same native input order; no extra data.
Compare existing noaux78.323699 selected/final160 and curve0/80/160. This reuses
semantically matched native recipe, with the documented removal of an unused
frozen head/cache reader. New-code controls/seeds still required for a strong
attribution claim if promising. Global incumbent78.709056 is not subset-matched.

Evaluate0/80/160 fullDEV. Step0 recalls compared across streams/directions;
historical rawscore numeric uncertainty remains, no bitwise parity claim.
Selection unchanged meanR1, directional guardrails; stop if mean drops>2pp from
release. Initial checkpoint stays eligible. CPU tests cover exact native inference,
branch preservation, DCL equation/gradient identity and independent native subset.
First real steps include finite-gradient, nativeCE reconstruction and step2GCN/
fusion updates with frozen-buffer checks; failures stop the bounded job.

PROMOTE for meaningful matched gain/trend, not decreasing DCL loss alone; follow
with fuller-data and matched confirmation only after user returns. REFINE if
curve/q/gradient shows a specific correctable limitation, at most motivated scope.
DROP if learnable objective lacks useful ranking signal after suitable test.
No claim of significance, independent generalization or sufficient novelty from
one repeatedly selected DEV pilot. No TEST loading/tuning.

## Execution and retention

Run v4-c17-dcl-001 -> seds-fused-dcl-offload-001.
Command: /home/haipd/miniconda3/envs/seds/bin/python -u
research/slret_goal_v2/tools/run_c17_dcl_v4.py --launch
Estimate20–25min; hard1800s plus outer1850s. Reserve1850 of14573.928131s remaining.
16GiB allocator; owned descendant watcher19GB safetytrip; user20GB decimal limit.
Storage36.536GiB +1.25GiB <38GiB, free~128GiB >15GiB; no deletion or download.
Compact nativeGCN+fusion best/last plus logs/config/source/optimizer/RNG. Last state
retained, but resume entrypoint remains unimplemented; no automatic retry promise.
Local status/heartbeat/exit/summary only. One startup check then WAITING_FOR_USER.

Launched startUnix1789911230.5672507. Sole startup check: supervisor3730004,
torchrun3730005, trainer3730038 alive; initialDEV519 entered, expected native_subset/
dcl/signrep=None config recorded. First status precedes GPU sample (0 not interpreted
as peak). 20 scoped CPU tests and py_compile passed. No retrieval outcome yet.
WAITING_FOR_USER, no agent polling or automatic follow-on research.

Collected2026-09-20: full160, all gates passed. Selected80mean77.649326 vs
noaux78.323699; final77.456647. No promotion. PureDCL clipped160/160steps,
median total norm9.313722 vscontrol.528228 (57/160clipped). Pose-only mean improves
2.312139pp but fused R1 does not. Loss/scales/recalls in RESULTS.md.
R1 tests a modest convex blend .95CE+.05DCL at unchanged native branch weights,
same warm start/data/LR/horizon; not a claim that clipping is the sole cause.
