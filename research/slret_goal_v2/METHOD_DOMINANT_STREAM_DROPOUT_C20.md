# C20 — annealed dominant-stream dropout during fusion training

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline run
- Origin Date: 2026-09-21
- Verification Status: implementation verified by scoped CPU tests; retrieval efficacy unverified
- Version Label: c20_pilot_v1
- Authority: GoalV4 and user; no TEST access, external upload or autonomous polling

## Evidence and hypothesis

On the fixed release DEV readout, RGB is materially stronger than pose (about
73.8 versus61 R1 depending on direction), while the best reproducible gain so far
comes from adapting the pose GCN with native fusion. C19 added an inference-time
exchange path but did not improve the matched control. A plausible remaining
failure is training-time reliance on the stronger RGB stream, limiting how much
the newly adapted pose representation affects fused retrieval. This is a
hypothesis, not a proven diagnosis.

Borrow ModDrop's principle of randomly omitting modality channels during fusion:
Neverova et al., TPAMI2016 ([primary paper](https://doi.org/10.1109/TPAMI.2015.2461544)).
Counterevidence matters: Dai et al., CVPR2024, find that common dropout can induce
modality bias and hurt complete-input AVSR ([primary paper](https://openaccess.thecvf.com/content/CVPR2024/html/Dai_A_Study_of_Dropout-Induced_Modality_Bias_on_Robustness_to_Missing_CVPR_2024_paper.html)).
Neither result establishes SLRet efficacy or novelty.

## Registered mechanism

Only the input to native CGAF changes during training. At update1, independently
for each batch item, the RGB fusion tensor is zeroed with probability.2. The
probability decays linearly to zero at update666; the last update therefore uses
complete RGB+pose inputs. Pose is never dropped. Native pose/RGB auxiliary losses
still see their original streams. There is no rescaling, new loss, teacher,
checkpoint, parameter, inference module or retrieval-score correction. A private
seeded generator prevents the new Bernoulli draws from shifting native dropout
RNG. Evaluation disables the intervention exactly, and its parameter-free module
adds no state-dict keys; selected weights load into the unmodified inference model.

The one fixed start probability.2 is deliberately low and averages about.1 over
training. Annealing is the synthesis prompted by the complete-input counterevidence,
not a tuned schedule. C08's failed masked-pose reconstruction altered coordinates
and added a decoder/loss; C20 leaves the encoders, inputs and losses intact and
regularizes only fusion exposure. It does not reopen closed confidence gating,
gradient surgery, DCL/C17, global exchange/C19 or score-correction families.

## Registered experiment and decision rule

Run `v4-c20-rgb-moddrop-annealed-001` -> `seds-rgb-moddrop-annealed-001`.
Release checkpoint, full TRAIN7096, seed42, B32, three epochs/666 updates, native
loss, pose GCN1e-6 and fusion1e-5. Same batch order, scheduler, DEV519 evaluations
at0/111/222/444/666, selector, per-direction guard and >2pp early-stop rule as
historical GCN-R1 seed42. That control selected78.61271676300578; global incumbent
seed1337 is78.70905587668594. Repeated DEV selection remains exploratory.

Runtime estimate~1850s; hard2300s/outer2350s. User maximum20GB decimal;16GiB
allocator and19GB owned-process watchdog, query deadline20s. Reserve2350s from
remaining4145.514178738857s. Storage40,441,406,611bytes +4GiB planned fits42GiB
agent cap and15GiB free floor after removing only two unselected GCN-R1 `last.pt`
states; all selected bests and provenance remain (manifest
`storage-prune-gcn-r1-last-20260921-001`).

PROMOTE only if the selected/full curve materially exceeds matched seed42 and has
a credible comparison to78.709056; replicate a positive lead before attribution.
DEFER if it ties/loses with valid activation and complete-input terminal gates.
At most one refinement may follow only from the observed dropout/count/curve
evidence; no blind probability or schedule sweep. Command:

`/home/haipd/miniconda3/envs/seds/bin/python -u research/slret_goal_v2/tools/run_c20_moddrop_v4.py --launch`

Launched detached at Unix1789927798.3877852. Sole startup check found launcher
4076368, supervisor4076369, torchrun4076373 and trainer4076422 alive; status
RUNNING during native initialization, no immediate error. No autonomous polling.

## Outcome: incomplete timeout, park without final verdict

The bounded job timed out after146/666 updates (supervisor2303.211750268936s),
with all PIDs absent and no OOM/model error. Step111 meanR1=77.84200385356455,
-.09633911368015pp versus the historical seed42 control at the identical step.
The intervention and native update gates passed, but annealing had only reached
.15639 rather than0, so the registered terminal condition is missing. Exact
optimizer/RNG resume is unavailable. Because common-step fused evidence is already
negative and remaining compute is insufficient for valid replay, park C20; do not
interpret this as a completed family rejection. Full evidence is in RESULTS.md.
