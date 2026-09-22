# Dataset-first SLRet track — active state

## Current operational state — recovery verified (2026-09-22)

The user sent `xong`. C27 recovery is terminal COMPLETED, exit0, with519 DEV
and7096 TRAIN artifacts passing the scoped CPU integrity audit. Terminal
records confirm that the paused worker was resumed; its current state was
not polled. Charge6339.263507 supervisor seconds once, leaving11,537.649493
seconds of the new grant. No new GPU job or retrieval training was launched.

All13 TRAIN examples failing the CTC length condition were confirmed through
the native loader. The unchanged all-TRAIN CTC recipe is therefore technically
ineligible; do not skip examples or suppress losses silently. C27 remains OPEN,
not SUPPORTED-FOR-PILOT, independently of successful recovery. The authoritative
[verification report](../../docs/codex_slret_research/evidence/C27_RECOVERY_RESULT.md)
contains scope, hashes, counts, limitations and tests. Human annotation remains
excluded. This update supersedes older waiting/automatic-pilot language below.

## Historical launch state — WAITING_FOR_USER (2026-09-21)

C26-A is terminal and weak (selected CSL DEV mean R@1 43.700670); do not
relaunch it. A distinct C27 gloss-sequence/CTC plus native contrastive lead is
registered in `C27_DECISION.md`. PH TRAIN gloss coverage and three focused CPU
CTC tests passed. No C27 retrieval training has started.

Detached prerequisite job **`c27-ph-feature-recovery-001`** was launched at
2026-09-21 22:51 local with supervisor PID 1166311. Its one startup check
verified a live supervisor and the official RTMPose checkpoint download stage;
the GPU extraction stage has not yet been observed by this agent. The finite
queue is checkpoint download/checksum → adapted PH DEV519 extraction → adapted
PH TRAIN7096 extraction. The parent records progress and terminal outcome at
`artifacts/slret_dataset_first/jobs/c27-ph-feature-recovery-001/{run.log,status.json,summary.json,launch.json}`.
The supervisor will verify and SIGSTOP only the exact UniFormerV2 GPU worker
before extraction, then SIGCONT it in `finally`; an independent local watchdog
covers supervisor loss. Hard wall cap 7,200 seconds, owned VRAM cap 20 GB,
disk floor 20 GiB, prior 17,876.913-second new-grant balance. Do not poll this
job or UniFormerV2 until the user says “xong” or asks status. On return, inspect
one terminal status/log/outputs check; if successful, complete the C27 matched
control/candidate training implementation and launch that bounded pilot.

At ~22:45 local, old `artifacts/slret_goal/` contents, including C26 job logs and
adapted SEDS feature caches, were observed absent while free disk rose by
~49 GiB. This was not a C27 deletion. The C26 result remains recorded in
`C26_DECISION.md`; old artifact links must not be assumed live. The C27 job
writes to a new, explicit artifact root and does not replay old audit work.

## Decision

On 2026-09-21 the user deferred C24 and requested a new research direction that
does not default to adding another component to SEDS. CoSign-LI/C25 was prepared
but the user then deferred it before any GPU efficacy test. This is USER_DEFERRED,
not an empirical negative. C26-A has since been measured and deferred as a
standalone accuracy recipe, recorded in `C26_DECISION.md`; no replacement has
been promoted to an efficacy-tested lead.

Operational status: **C26-A pilot measured; no C26 job running**. The user's
status query found technical attempt `c26-unisign-csl-pilot-001` FAILED after
all 19,478/19,478 frozen features and zero-shot DEV were produced: the
projection evaluator erroneously required equal video/text counts, but the
CSL gallery has 1,077 videos and 797 caption groups. The wrapper resumed the
exact UniFormerV2 worker. Rectangular scoring was repaired and regression-
tested (17 track tests pass); `c26-unisign-csl-pilot-002` reused the complete
feature cache and COMPLETED, exit 0, all five head epochs. Its selected epoch
4 is T2V R@1 44.040151, V2T R@1 43.361188, mean R@1 43.700670; zero-shot
mean R@1 5.785507. The existing CiCo CSL DEV reference mean is 67.403588
under the same 1,077/797 gallery and metric, but it uses different features
and English caption rendering, so it is not a matched causal control. C26-A
is a poor standalone accuracy candidate; defer this exact frozen-global-head
recipe, not every Uni-Sign-based method. No TEST was loaded. Both supervisors
verified UniFormerV2 resumed; the user-authorized check saw its worker state R.
Attempt outputs: `artifacts/slret_goal/jobs/c26-unisign-csl-pilot-{001,002}/`.
Combined conservative supervisor wall charged is 123.087 seconds of the new
18,000-second grant, leaving 17,876.913 seconds of that tranche. C26 peak
reserved VRAM was 3,210,739,712 bytes in extraction and 56,623,104 bytes
in cached-head training, each below 20 GB.

Asset status: on the user's
2026-09-21 status request, the one-shot check of detached download
`c26-assets-001` found terminal COMPLETED, exit code 0, all 7/7 files and
3,520,971,333/3,520,971,333 expected bytes; the job's summary records the
verified per-file sizes and SHA256 hashes. It started at 13:18:32Z and ended
at 14:04:44Z. Job records:
`artifacts/slret_goal/jobs/c26-assets-001/{run.log,status.json,summary.json,launch.json}`.
The download used no GPU. The real checkpoint now strict-loads on CPU (627
state tensors, zero missing/unexpected keys). A two-example real CSL DEV
forward produces finite video/text features of shape [2,768]. This smoke
revealed and repaired a pose normalization float64-to-float32 mismatch; eight
focused CPU tests pass. Before the current pilot, no C26 feature extraction
or training had started.
The user granted the numeric GPU compute. The measured C26 pilot has ended;
unused compute does not justify an unmotivated retry or sweep.

UniFormerV2 was automatically resumed by both C26 supervisors and was running
at the last user-authorized check. It is not a C26 prerequisite and must not
be independently polled. C24 remains deferred.

## Evidence completed in this track

- Annotation-only structure analysis implemented in
  `tools/analyze_structure.py`; synthetic tests pass.
- First complete run saved under the ignored artifact
  `artifacts/slret_goal/dataset-first-structure-001/report.json`.
- Direct multi-vector scorer implemented in
  `methods/sl_mvr/late_interaction.py`; all scorer and analysis tests pass.
- Focused primary-source review and no-go collision screen completed in
  `LITERATURE.md`.
- C25 hypothesis, controls, gates, and compute sequence retained as history in
  `METHOD_COSIGN_LI.md` but not authorized for launch.
- C26 donor bridge, deterministic CSL pose loader, and reusable frozen-feature
  cache/projection head implemented in `methods/translation_retrieval/`;
  all 17 track CPU tests pass. Official CSL
  TRAIN/DEV labels match local labels byte-for-byte, and all 18,401/1,077 named
  133-point pose files exist. One real DEV clip's four normalized parts match
  the donor pose loader within 2e-7 max absolute error. Donor `models.py` imports
  under base Python but not usefully under the `seds` torch/Transformers mismatch;
  the pinned public checkpoint and mT5 initialization assets are now downloaded
  and verified. Strict checkpoint loading and a two-example real-data CPU
  forward now pass after the float32 loader repair.

## Current boundary

C25 has no GPU efficacy test. C26-A now has a valid exploratory result well
below the existing CSL DEV reference; no C26 run is active and no new method
has been promoted. The previous ~430-second balance is retained separately
from the remaining 17,876.913 seconds of the user's new grant. Do not launch
another C26 configuration or score TEST merely to rescue this weak recipe.
Select a distinct evidence-based lead before the next GPU job.

## 2026-09-22 scientific admission update (operational handoff unchanged)

The autonomous research audit has not checked or modified the outstanding C27
recovery job. Its `xong` boundary still applies. Saved-score analysis now rejects
the narrow exact-gloss-multiset order-confusion motivation: zero strictly
outranking such confusers among persistent errors across three GCN seeds.
Broader C27 is OPEN, not SUPPORTED-FOR-PILOT. This scientific update supersedes
any earlier implication that successful recovery automatically triggers a C27
trainer/run. After `xong`, verify recovery and account for resources; require
method admission and the same-gloss order-free control before new training.
See [C27 decision](C27_DECISION.md) and
[current research state](../../docs/codex_slret_research/00_RESEARCH_STATE.md).
