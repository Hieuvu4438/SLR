# C19 — global bottleneck exchange before native local fusion

## Family decision: deferred after bidirectional and directed contrasts

C19-R1 attempt001 FAILED technically after2 updates, before any post-training DEV.
Generic fusion configuration re-enabled frozen `to_pose`, producing262656 optimizer
params rather than131328; the step3 gate then rejected gradients for that unused
direction. Repair reapplies direction-specific trainability immediately before
optimizer creation and extends global-exchange smoke to3 updates. 18 targeted
tests pass. Retry002 changes no data, loss, LR, horizon, seed, module forward or
selection rule; do not interpret attempt001 as method performance.

Retry002 COMPLETED160/160, exit0, supervisor wall1737.84938955307s. Selected and
final meanR1=78.22736030828517, -.09633911368015902pp versus matched control and
-.096339114pp versus bidirectional C19. T2V R1=77.26396917148362; V2T R1=
79.1907514450867. Curve0/80/160=77.552987/78.131021/78.227360. All registered
identity, output, all-gradient, update, buffer and delta-roundtrip gates passed;
131328 direction-specific adapter parameters trained. Best SHA256
2962647ac1a8c26750f6dba5622827c2b49802c07420b3b70174b37ca09f020d retained.
The adapter was active but did not improve retrieval. Defer this family without
an LR/rank/slot/within-stream sweep; this is neither a novelty nor SOTA result.

C19 recovery COMPLETED160/160. Selected meanR1=78.32369942196532, effectively
identical to matched native control78.32369942196533. It trades +.192678pp T2V
for -.192678pp V2T, so there is no net gain and no promotion. All activation,
gradient, update, delta and recovery gates passed. Full details are in RESULTS.md.
No within-stream control is warranted without a positive signal.

C19-R1 tested one evidence-driven direction change: only pose-to-RGB messages are
active; RGB-to-pose is frozen and pose input to native CGAF is bitwise unchanged.
This follows the SEDS local-pose/global-RGB division and tests whether contaminating
the pose-local stream caused the observed directional tradeoff. Total checkpoint
schema stays262656 parameters for compatibility, but only131328 adapter parameters
train. Everything else matches C19/control: seed42,TRAIN512,B32,10epochs,160updates,
native loss, GCN1e-6,fusion1e-5,adapter1e-4,DEV0/80/160. This is an exploratory
refinement, not a novelty/SOTA claim. It did not beat control or show a useful
endpoint trend, so the preregistered DEFER rule is now applied.

Repair driver: `research/slret_goal_v2/tools/run_c19_pose_to_rgb_repair_v4.py --launch`.
Hard3600s/outer3650s; actual supervisor wall1737.84938955307s; user20GB safeguards held.
18 targeted tests pass; broader unittest has76 executable passes and three
unrelated missing-pytest imports. No TEST access or baseline replay.

Attempt001 launch record and failure remain in job/training artifacts.

## Historical recovery design

Original v4-c19-global-exchange-001 TIMED_OUT at2400s, wall2403.396547317505s.
121updates logged, last.pt at112 (all optimizer state/RNG/batch order retained).
DEV80mean78.131021, +.096339pp vscontrol80 but-.192678pp vscontrolselected160.
No final160 metric, no promotion or negative final conclusion. All initial/new
module gates passed; observed10.058GB. Original runtime estimate was too low;
median17.452807s/update, actual48remainingupdates+eval estimated18–25min.

Recovery job v4-c19-resume-001 -> seds-global-exchange-cross-resume-001.
Pinned checkpoint SHA256f62c84e34d2568beafcde31a22a35bbdf9ae5be55084aadb3c91a2fbf96fb81f.
Trainer accepts --resume-from only for C19cross; helper checks exact recipe,
parent snapshots/base/checkpoint hashes and complete batch plan. Source change
limited to reviewed resume wiring in train_seds_extended.py plus new helper;
model/loss/optimizer/scorer source unchanged. Restore compact learnedweights,
FP32 BertAdam moments without casting through parameter dtype, step112 schedule
and Python/NumPy/CPU/CUDA RNG immediately before update113. No LR/warmup reset.
Original112log rows copied with provenance;9 unsaved rows113..121 discarded from
resumed trajectory and recomputed, parent log retained. Firstresumedupdate must
change adapter outputs/GCN/fusion without frozenweight/buffer drift. End160 unchanged.
Inherit DEV0/80 and selectedbest80, with source paths; do not rerun them. New DEV160
uses original selector. Final selectedcheckpoint may remain parentbest80.
29 tests passed including CPU BertAdam uninterrupted-vs-resumed nextupdate exact
equality, FP32 state on FP16parameters, schedule/recipe rejection and inherited
metric provenance. This does not claim full GPU bitwise uninterrupted equivalence.

Command: /home/haipd/miniconda3/envs/seds/bin/python -u
research/slret_goal_v2/tools/run_c19_resume_v4.py --launch
Hard1800s/outer1850s; reserve1850 of7116.693375s; no new compute grant.
20GBdecimal/16GiBallocator/19GBtrip/query20s retained. Storage42.430GB+1.25GiB
headroom <42GiB, free134.074GB/floor15GiB. No deletion. One startupcheck thenwait.
No automatic retry/refinement, no C17 recovery or DCL run.
Recovery launched startUnix1789921518.9217765; supervisor3948969/torchrun3948995
alive at sole startup check, model initialization in progress, no immediate error.
Actual resumed-update gates not yet observed. STATE=WAITING_FOR_USER; no polling.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline run
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED retrieval efficacy; scoped CPU checks passed
- Version Label: c19_pilot_v1
- Authority: user GoalV4 §§1,2,4,6,8; no external uploads or autonomous polling

## Hypothesis and counterargument

Sparse cross-stream sampling may miss useful video-wide interactions even though
each encoder already carries temporal context. A small clip-to-slot-to-clip
exchange might add useful cross-modal information at low parameter cost.
This is a hypothesis, not a diagnosed absence of global semantics. Native learned
offsets are not mathematically confined to nearby frames. SEDS explicitly reports
that unrestricted cross-attention can be worse because of irrelevant distant
features (paper fusion ablation); preserving CGAF and restricting added messages
to four slots is a testable response, not a guarantee against that failure.

C18 provided no gain from B64; C17 is USER-CLOSED. Neither proves architecture is
the bottleneck. C19 changes an information path instead of repeating loss scaling.
No novelty/SOTA claim. Core method claim remains prospective until matched controls.

## Borrowed component and implementation

Borrow attention bottleneck principle from Nagrani et al., NeurIPS2021, Sec3.2.3:
https://papers.neurips.cc/paper_files/paper/2021/file/76ba9f564ebbc35b1014ac498fafadd0-Paper.pdf
Primary methods and relevant fusion ablations read online; no local PDF anchors.
Independent implementation, not copied code/full MBT or imported pretrained model.
MBT's audiovisual classification results do not establish SLRet efficacy.

Inputs: native contextual RGB and pose tensors [B,65,512] plus padding mask.
Before native model.fusion, two directions separately collect four rank64 slots
from all valid opposite-stream clips, then let each target clip read those slots.
Zero-initialized output projections preserve exact initial inputs. Native CLS and
padding are unmodified by this adapter; original downstream mask behavior stays.
Added parameters262656; no dropout/newloss/labels/teacher/new modality. Native CGAF
still executes, so this is NOT a claim all fusion goes through bottlenecks. Original
pose/RGB auxiliary logits remain computed from unmodified streams. TrainableGCN
and originalfusion retained. The new adapter is under fusion.global_exchange,
included in compact checkpoints. Inference needs attach_global_exchange before
loading delta over release. No gallery/text-conditioned state; per-video encoding.
Extra projected attention costs scale with clips×4×64; actual runtime measured
in pilot. This is not a speedup over native SEDS because it adds computation.

Collision: C07 joint exchange is pre-pool anatomical256-d joint messaging; C19 is
post-encoder cross-modal512-d clip messaging. C06 adjacent temporal differences,
C03 pooled part products, C13 graph topology and closed scalar score/rival/gating
mechanisms are absent. No new confidence gate, teacher support, graph over gallery,
relevance change, temporal-order objective, decoder protection or C17/DCL revival.
An attention primitive overlap alone is not identical mechanism under GoalV4§5.

## Registered experiment

Run v4-c19-global-exchange-001 -> seds-global-exchange-cross-001.
Release checkpoint6f07e5ab...95f6af5; same TRAIN512 IDs and B32 batches as C16
noaux control,10epochs/160updates, seed42. DEV519 eval0/80/160. No TEST load.
GCN1e-6/originalfusion1e-5/newadapter1e-4, FP32 native moments, normclip1,
activation offload/mathSDPA; frozen native BN/dropout except fusion like control.
Native loss components unchanged. Sample exposures/schedule/control recipes match.
Same maxmeanR1 selector, earliesttie1e-8pp, per-direction reference-.5pp guard,
earlystop>2ppmean drop. Reference77.552987, matchedcontrol78.323699, globalincumbent
78.709056 (different TRAIN exposure, not matched). RepeatedDEV exploratory only.

CPU tests cover identity, masks/CLS/emptyclips, opposite-source dependency,
within-source isolation, independent-video batch behavior, all gradients after
activation, actual attachment, fresh-base learned-delta reconstruction, LR groups.
23 combined scoped tests pass. In-pilot first2updates must activate both outputs
and GCN/fusion, preserve frozenweights/buffers; step3 requires all adapter gradients
finite/nonzero and traineddelta roundtrip. FullDEV0 and exact adapter input identity
are relevant new-path checks, not wholebaseline audit. Stop job if gates fail.

PROMOTE only with worthwhile gain/trend above matched control; then within-stream
sameparameter control (already implemented mode), fullTRAIN/seeds as justified.
REFINE at most one factor from actual scale/curve evidence; no arbitrary module
stack. DROP/defer if adequate learning has no useful signal. Opposite versus within
comparison is required before claiming cross-modal mechanism, not just capacity.

Command: /home/haipd/miniconda3/envs/seds/bin/python -u
research/slret_goal_v2/tools/run_c19_global_exchange_v4.py --launch
Expected25–35min; hard2400s/outer2450s, one pilot only. Reserve2450s outof9520.089922s.
User20GBdecimal:16GiB allocator/19GB owned-descendant trip/querytimeout20s; fatal on
unknown telemetry. No hardware memory partition. Save best/last/log/config/source.
Storage41.679GB+1.25GiB reserve: agent subcap40->42GiB underGoalV4§8; free134.88GB,
15GiBfloor. No deletion/download or re-extraction. One startupcheck then WAITING.

Launched startUnix1789918567.248251; supervisor3884040/torchrun3884041 alive at
sole startupcheck, initial fullDEV519 entered, no immediate error. Actual new
module update/memory/efficacy gates not yet collected. STATE=WAITING_FOR_USER.
