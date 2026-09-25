# Research log

## Cycle 1

Date / commit: 2026-09-22 / `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.

Research question: Is the inherited C27 ordered-gloss lead sufficiently
supported and distinct to justify a new retrieval method pilot?

Evidence inspected: user attachment/guide; current dataset-first and V2 states;
proposal7 and V1 exclusion registries; original 12 archived gate files; retained
GCN-R1 run and selected metrics; CiCo/SEDS scoring and PH loader; C27 CTC module;
official PH TRAIN gloss CSV; CVT-SLR and CSLR² primary method/ablation passages;
September 2026 SignSeek task definition. Detailed limits in audit files.

Hypothesis: ordered gloss supervision provides rank-relevant evidence beyond
generic gloss content. Alternative: extra annotations or ordinary adaptation
explain any future gain.

Experiment / diagnostic:

```bash
/home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/audit_research_state.py --output docs/codex_slret_research/evidence/state_audit_20260922.json
/home/haipd/miniconda3/bin/python -m pytest -q docs/codex_slret_research/tools/test_audit_research_state.py research/slret_dataset_first/tests/test_c27_gloss_ctc.py
```

Result: 12/12 original gate hashes match. PH TRAIN7096, unique ordered gloss
sequences6900, unique multisets6874, 1085 types/55247 tokens. Distinct-order
multiset groups23/140 rows; groups also containing distinct exact translation22/
138 rows. Adjacent repeated glosses occur in56 rows; maximum formal CTC steps30.
Five focused CPU tests pass. No GPU, model training or DEV/TEST annotation reads
in the new diagnostic; no active-job checks.

Interpretation: original negative evidence remains usable. Ordered annotation
exists, but the exact-multiset contrast is a small and linguistically unvalidated
slice. It does not establish a retrieval bottleneck. Prior art makes the generic
joint-supervision story insufficient; a same-gloss order-free control is needed.

Alternative explanation: CTC could help broad lexical recognition rather than
order; the exact-multiset census would not capture that mechanism. Distinct
translations can be paraphrases. Formal CTC length feasibility does not ensure
visual evidence survives compression.

Decision: **redesign diagnostic and controls; continue research**. C27 stays
OPEN, not SUPPORTED-FOR-PILOT. No new training behavior changed. Preserve the
existing feature-recovery handoff until explicit user completion/status input.

New evidence labels: [M] current hash/annotation counts and fixtures; [V] source
paths, retained GCN metrics and primary-paper passages; [A] external ablations;
[H] C27 retrieval improvement remains unmeasured.

Closed mechanism(s): no newly measured method closure. Reject the unsupported
claim that “CTC plus contrastive learning” alone establishes novelty. All prior
closed families remain closed in their recorded scope.

Next highest-information branch: complete remaining history/baseline/question
audit independently; make an order-versus-content diagnostic decisive before
implementing a C27 trainer. After `xong`, verify recovery once, without launching
training merely because features are available.

Verification before handoff: all local Markdown links in the ten new research
documents resolve; every source hash recorded by the new diagnostic matches
current bytes; `git diff --check` passes. These checks validate artifact
integrity only. The requirement audit explicitly leaves the full history,
30-question map, candidate screen, primary proposal and controlled experiments
unfinished; the goal is not marked complete or blocked.

## Cycle 2

Date / commit: 2026-09-22 / `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.

Research question: Do persistent errors of the strongest retained GCN controls
support exact gloss-order confusion, and where should the next diagnostic focus?

Evidence inspected: complete proposal7 main report, Phase2 bottleneck report,
Q01–Q37 feasibility audit; completed static GCN runs for seeds42/1337/2026;
their selected fusion/pose/RGB matrices, metric files and official PH DEV CSV.
No active-job status, process, checkpoint forward or TEST was accessed.

Hypothesis: exact gloss-multiset/different-order competitors explain a material
share (predeclared ≥10% in either direction) of persistent errors. Alternatives:
lexical/representation or scoring defects, shared training bias, singleton
relevance ambiguity. Descriptive branch scores do not identify causality.

Diagnostic, with protocol written before execution:

```bash
/home/haipd/miniconda3/bin/python -m pytest -q docs/codex_slret_research/tools/test_gcn_residuals.py
/home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_gcn_residuals.py --output docs/codex_slret_research/evidence/gcn_residuals_20260922.json
```

Result: three fixtures passed. Nine matrix replays match all stored directional
metrics and query ranks exactly. Persistent errors: T2V113/V2T98. Both branches
fail in every seed for88/74 of those respectively. Same-gloss/different-order
strict confusers: zero in both directions. Common strict confusers across seeds:
113/96, which cannot be repaired by nonnegative mixtures of these fixed seed
scores. This algebraic bound is not an ensemble result. Selected mean R1 across
seeds78.612717; sampleSD0.096339pp. Full counts/limits in the
[result](evidence/GCN_RESIDUAL_RESULT.md).

Interpretation: reject only the narrow exact-order explanation. No demonstration
of absent visual evidence, linguistic equivalence, or CTC efficacy follows.
Most residuals are stable, so seed-score mixtures cannot address their common
strict competitors. Candidate-dependent/new-model mechanisms are outside this
bound. Shorter gloss sequences have higher observed error rates, noncausally.

Decision: continue with Q01/Q22 diagnostic design and Q23 as an alternative;
do not launch C27 on prerequisite availability. C27 remains OPEN. Wrote38
question records with all required fields and seven-component priorities;
validated component ranges/sums and record count. Literature verification of
each inherited nearest-prior claim remains unfinished. Files09–12 remain pending.

New evidence labels: [M] saved-score replay, residual counts and tests;
[V] inspected historical reports; [H] causal explanations and method benefit.
Closed mechanism: exact-order-confuser material-bottleneck story in this DEV
slice only; existing-score nonnegative mixtures have the stated per-query bound.
No new method family is globally closed or admitted.

Next highest-information branch: finish mandatory history/source audit and
predeclare an adequate distinct representation-versus-scoring diagnostic before
the 5–10-candidate screen. User `xong` is still required for the existing job
check. Goal remains active; no SOTA claim or exhausted-search claim is made.

Handoff verification: all31 recorded source hashes across the two diagnostics
still match, all25 local Markdown links in the audit tree resolve, and all38
question records satisfy required fields, component ranges and priority sums.
Combined focused suite: **8 passed in0.80s**. `git diff --check` passes.
No model/training source or existing recovery-job state was changed; the
dataset-first state and C27 decision received additive scientific admission
notes to avoid an automatic pilot after prerequisite recovery.

## Cycle 3

Date / commit: 2026-09-22 / `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.

Previous goal turn: **PROGRESS**, from saved-score replay, residual measurements
and the completed question map. Current cycle: source/history audit progress,
not a new retrieval experiment or verified wait.

Research question: What implemented mechanisms and comparison limits must the
next candidate screen account for, and is the proposed Q01/Q22 next step new?

Evidence inspected: attachment/current guide; all remaining mandatory proposal7
history, including full1,985-line state,1,344-line loop and481-line literature
log; complete Q01 positive-control and Q02/Q22 function-class notes; historical
C²RL derivative/task gates; current UPRet scorer/trainer, SAN model/data/loss/
evaluation/recipe and four complete CMCM component files. Exact inspected paths,
source hashes and paper-reading scope are in
[BASELINE_MECHANISM_AUDIT.md](evidence/BASELINE_MECHANISM_AUDIT.md).

Hypothesis under audit: baseline names or historical pins may hide materially
different implementation/resource contracts, and the generic proposed probe
design may repeat completed local analysis.

Diagnostic: read-only source tracing and git-root/HEAD resolution; primary UPRet
§3.6, C²RL §§III-B/C and TablesVI/VII, SAN method/Table1 and ACL record;
publisher CMCM search content with failed direct open disclosed. No model,
dataset-content, active-job, GPU, feature/checkpoint or new TEST access.

Result: vendor git queries resolve to parent SLR, not independent upstream
checkouts; historical pins cannot be freshly certified by those commands.
UPRet's transport auxiliary is intentionally training-only; its current native
entry point selects via test. SAN consumes an external negative table, adds a
video→hard-caption CE, uses German BERT/mBART-form encoders, and selects through
its configured test loader. C²RL pretraining already combines contrastive and
translation objectives; derivative SLT code is not its trained retrieval release.
CMCM components do not establish an integrated trained comparator. These mostly
corroborate historical findings rather than invent new source defects.

Interpretation: the six named baseline mechanisms now have bounded comparison
coverage sufficient for screening; exact paper reproduction is not necessary.
The proposed generic Q01/Q22 next action repeats completed adequacy/expressivity
analysis. Corrected that assumption and added explicit hypothesis contracts for
the five high-priority questions, retaining missing definition/resource gates.

Alternative explanation: private/different author code could resolve public
release gaps. Current source behavior alone cannot invalidate reported outcomes.
Poor probe performance still cannot establish absent information. Historical
scopes and instructions are not current active-job commands.

Decision: **continue**, moving to the bounded5–10-mechanism admission screen,
not repeating broad audits, generic probes, repair certificates or availability
lookups. No candidate promoted; C27 remains OPEN. Original negatives unchanged.
No implementation, training launch, dependency polling or status mutation.

New evidence labels: [V] current source paths/hashes/git-root and primary passages;
[A] external benchmark reports; [I] admission/prioritization consequences;
[U] complete trained external comparators and final frontier certification.
Closed mechanisms: no new empirical family closure; preserve existing reduction,
mining, calibration, pooling, auxiliary/context and resource-route decisions.

Next highest-information branch: use the completed source/history contracts to
screen materially distinct candidate mechanisms and identify a genuinely new
separating diagnostic. A candidate remains OPEN or REJECTED unless its admission
criteria can be evidenced. Files09–12 remain unfinished; no primary method or
SOTA claim, no global exhaustion/blocked claim. Recovery still waits for `xong`.

Verification: all12 vendor source hashes in the new report match current bytes;
all33 local Markdown links in the research tree resolve; `git diff --check`
passes. Documentation-only cycle: no new test-suite result or runtime baseline
verification is claimed.

## Cycle 4

Date / commit: 2026-09-22 / `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.
Previous turn: **PROGRESS** in bounded source/history reconstruction. This turn:
**PROGRESS** in candidate admission, not new efficacy or verified job waiting.

Research question: does any of six distinct intervention routes survive the
measured-bottleneck, local-separation, prior-art and fair-pilot gates?

Evidence: completed baseline/history/question/hypothesis records; retained GCN
residual result; current blacklist; proposal7 candidate screen; dataset-first
literature map; fresh primary arXiv abstracts and primary CVF indexed excerpts for
Thinking Fast and Slow and TokenBinder; m-RNN training equations and retrieval
section. Reading scope and direct403 failures are recorded in file08. The main
academic-research-suite workflow and skeptical-review instructions were followed
inline. No independent reviewer, external model or linguistic validation occurred.

Hypothesis under review: one of ordered gloss supervision, joint pair attention,
articulator relation binding, decoder likelihood, raw visual acquisition or
candidate-set comparison supplies an admissible distinct explanation/intervention.

Diagnostic: semantic mechanism collision screen with input/output, parameter and
ranking effects; explicit six-baseline comparisons; controls, signatures and kill
criteria; four sequential skeptical perspectives. No new model fitting, dataset
content access, feature/checkpoint load, TEST evaluation or GPU use.

Result: file09 now contains six candidate records. P1 (inherited C27) stays OPEN
because its broad hypothesis lacks a demonstrated bottleneck and matched efficacy;
its narrow exact-order story already failed. P2–P6 are REJECTED as proposed for
specified local/prior collisions and missing mechanism-specific evidence. m-RNN
provides a direct precedent for P4's caption-likelihood/prior score. Analytically,
the text-only prior cannot alter T2V ordering. This is not a new empirical result
or an assertion that a decoder equals the incumbent scorer.

Interpretation: the requested bounded candidate-screen artifact is now present;
no supported primary method emerged. Six routes are not six novel discoveries.
Available near-miss counts do not justify generic cross-attention, relations,
generative scoring, acquisition or candidate-set architectures by themselves.

Alternative explanation: a specific natural contrast and distinct intervention
could still justify a future candidate; none is identified here. Generic mechanism
precedents do not prove every sign-specific extension equivalent, and failed
admission does not prove impossibility of retrieval improvement.

Decision: **continue** research without promoting or implementing any route.
Do not populate files10–12 with a fictitious survivor. The next useful contribution
must be a new discriminating observation or genuinely separate causal mechanism,
not another broad audit or a renamed rejected candidate. No cheap empirical test
is currently admitted. This is not a global NO-GO, goal completion or blocked status.

New evidence labels: [V] bounded primary-source mechanisms; [I] algebraic/admission
consequences; [H] route predictions; [U] causal bottlenecks and all candidate gains.
No new empirical family closure, resource charge, launch or job status mutation.
The existing recovery remains behind the user's `xong` handoff; neither liveness
nor its output has been checked, and recovery cannot automatically admit C27.

Verification: six candidate sections contain all12 common record fields; the
separate matrix supplies all six named-baseline comparisons; six reviewer rows
cover the four perspectives. All42 local links across the12 top-level research
Markdown documents resolve. `git diff --check` passes. Documentation-only work;
no new runtime or statistical test result is claimed.

## Cycle 5 — annotation branch declined; automated scope preserved

2026-09-22; base HEAD unchanged. Previous goal turn: **PROGRESS**, completing the
six-route admission screen. Current turn's contribution is scope clarification
and a bounded metadata check, **not method or efficacy progress**.

Re-read the attachment, current state, relevant guide gates, skill workflow and
historical annotation-compatibility/semantic-scope reports. Four official DEV
annotation/fusion-metric hashes referenced by the retained residual report still
match. A metadata-only check found111/96 persistent T2V/V2T queries with stable
top non-paired candidates and381/373 all-seed rank0 queries with stable top
non-paired candidates. These are model-output strata, not human-valid controls;
no new semantic interpretation follows. No footage, features, checkpoint, TEST,
GPU or active-job state was accessed.

Prepared a draft optional video-relevance review protocol and asked whether the
user could provide review or existing compatible judgments. The user explicitly
answered **“Continue without human annotation.”** This is controlling authority:
the branch is stopped. The protocol is marked DECLINED; the newly created,
unexecuted preparation script was removed. No sampling manifest or review packet
was generated, no judgments collected, no reviewer contacted, and no upload made.
The retained protocol records the abandoned option, not a pending dependency.

The academic skill's consent/governance boundaries limited this branch to planning;
the user's answer now excludes its execution entirely. Do not reopen historical
gloss/text or incompatible mouthing-label searches to simulate the missing visual
judgments, and do not infer signed meaning with an LLM as a replacement reviewer.

Independent source checks revisited pose window selection, RGB feature loading
and the exact token scorer. They did not establish a new bottleneck. In particular,
slot-count agreement alone would not prove physical-time alignment, but missing
timestamp proof is not evidence of actual misalignment. C22's negative phase
modulation, closed sampling/support methods and prior scalar-score/positive-control
arguments remain binding. No synchronized-stream or pooling repair is admitted.

Decision: continue automated research under the original method and SOTA gates.
No candidate promotion, new training or recovery follow-up. The `xong` boundary
is unchanged. Human annotation is not treated as a global blocker. A specific
new automated diagnostic still needs an untested decision consequence; do not
label routine source rereading as one. Goal completion remains unproven.

## Cycle 6 — existing-annotation lead checked without relabeling

2026-09-22; base HEAD unchanged. Continued a bounded primary-source literature
scan under the no-human-annotation constraint. The academic-research-suite
verification discipline keeps source facts, local inferences and unavailable
resources separate; all phases ran inline, without delegated reviewers.

New source verification: [ASL-MTP scope record](evidence/ASL_MTP_DIAGNOSTIC_SCOPE.md).
It is a separately published diagnostic lead, not compatible relevance judgments
for the current PHOENIX confusers. The inspected sources and exact-name release
searches did not locate a download; global unavailability is not established.
No attempt was made to recreate its labels, transfer them to DGS, or contact
authors. SignMatch was also encountered but was already screened in proposal7;
it is not counted as a new discovery. This is not an exhaustive literature review.

Decision: record the source's diagnostic/prior-art scope, with no candidate
promotion, new method, empirical result, or SOTA claim. No runtime code was
changed, training launched, data downloaded, TEST accessed, or active job polled.
The existing `xong` handoff boundary remains intact. Human annotation is not a
global blocker, and this unreleased/unlocated lead is not a new dependency.

## Operational checkpoint after Cycle6 — first exhausted-work handoff

2026-09-22; HEAD `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.
Previous turn: bounded source-verification progress, not method progress or a
verified process wait. This continuation re-read the controlling attachment and
guide, the current38-question map, C27 admission decision and research state.
No new causal evidence or empirical result was obtained; do not count this
checkpoint as a scientific research cycle.

The available question contracts currently require either a new separating
mechanism, unavailable compatible evidence, or a later confirmation stage.
None specifies an unexecuted, admitted independent experiment now. Repeating
the existing tests, generic literature searches or resource-availability checks
would not move the method decision forward. Human annotation remains excluded.

The next concrete operational action is the recorded recovery verification,
which requires `xong` under guide§37. No such completion message has arrived.
The job's process, logs, status and output have not been inspected; this is an
instruction-gated handoff, **not a verified live wait**. Successful recovery
would establish asset readiness only and would not admit C27 automatically.

This is the first explicit exhausted-independent-work checkpoint for the present
handoff condition, not a retroactive claim that earlier source-verification
turns were blocked. Goal remains active and incomplete; no blocked/paused/
complete status update is made. Stop here rather than manufacture another
audit or bypass the handoff. After `xong`, perform the one targeted completion
check already specified in `00_RESEARCH_STATE.md`, then reassess admission from
the actual evidence. No training launch is authorized by this note.

### Handoff recurrence audit — third consecutive turn

2026-09-22. The following two automatic continuations supplied no `xong` or
equivalent completion message. The second turn revalidated the boundary without
scientific progress; the third confirmed the same instruction-gated dependency.
No job was polled, declared stopped, restarted or inferred live. No useful
independent action has been identified beyond the exhausted screen.

The three-turn blocked threshold is now met. `update_goal(status="blocked")`
returned **blocked**. This is not goal completion or a scientific impossibility
claim. Resume with `xong` for the single recorded recovery verification; preserve
the no-human-annotation constraint and the separate method-admission gate.

## Cycle7 — recovery accepted; all-TRAIN CTC feasibility assumption fails

Date / commit:2026-09-22 / `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.
Previous goal turn: no progress, instruction-gated block, not a verified live
wait. User `xong` now releases that dependency; this cycle makes concrete
artifact-validation and falsification progress, not efficacy progress.

Research question: are the recovered inputs complete and structurally eligible
for the already implemented CTC head? Evidence: terminal launch/status/summary,
checkpoint/source hashes, all saved TRAIN/DEV features, native label dictionaries,
official CSV IDs, native loader and head implementation. Hypothesis: every
TRAIN sequence has sufficient valid visual windows. Diagnostic: exhaustive
file/metadata checks, necessary CTC inequality, then native loading of every
failing case. [Result and scope](evidence/C27_RECOVERY_RESULT.md): recovery
complete;13/7096 TRAIN sequences violate the condition. Four DEV and16 TRAIN
native checks pass, including all13 failures. Seven focused CPU fixtures pass.

Interpretation: the unchanged all-TRAIN CTC recipe would raise, independently
of its already missing scientific admission. Alternative concerns about an
audit-only count error are reduced by native loading of all13 cases. No claim
that short windows cause retrieval failure follows. An intermediate pickle-byte
comparison false alarm was corrected to exact label-value comparison and
documented; underlying labels were not changed.

Decision: accept recovered artifacts for diagnostics; do not launch or repair
the C27 method into a pilot. Charge6339.263507 supervisor seconds once, leaving
11,537.649493 seconds of the grant. New evidence labels:[V] terminal/source
contracts,[M] integrity and eligibility,[I] unchanged-head failure. Closed
assumption: all-TRAIN length eligibility, not all gloss/CTC supervision.
Next scientific gate remains a measured, distinct mechanism with a separating
test; feature recovery and an engineering repair cannot substitute for it.
No human annotation, TEST access, extraction restart or new GPU job.

## Cycle8 — separate frame coverage from CTC temporal resolution

Date / commit:2026-09-22 / `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.
Previous turn: PROGRESS, recovered-input validation and13 confirmed CTC failures.
Research question: does the64-window cap omit retained frames, and is frame
removal responsible for those13 failures? Newly available evidence: recovered
original-frame/window indices, with inventory hashes matching Cycle7, plus
the native source. This is not a repeat of missing-provenance speculation or
the older CiCo pseudo-clip/sampler intervention.

Hypothesis: selected windows omit retained input frames. Diagnostic: registered
exact stage decomposition R→G→K→U before aggregate counts; no threshold sweep.
[Result](evidence/FRAME_RETENTION_RESULT.md): zero K-minus-U frames in7615
examples; all55,775 decoded DEV frames covered. Only23 TRAIN clips undergo
initial temporal subsampling, and four other TRAIN clips lose one support frame
each. All13 CTC failures retain their full16/17 decoded frames.

Interpretation: window cap is not the alleged frame-omission mechanism here;
CTC failures instead follow from only1/2 windows versus2–5 required timesteps.
Alternative explanation retained: encoding/pooling may lose information despite
complete input coverage; supplied MP4s are not proven original camera streams.
Historical feature equivalence and semantic/retrieval effects remain unknown.

Decision: close the exact cap-omission and CTC-support-filter explanations;
do not add a sampler, increase window cap, skip labels or implement a CTC rescue.
New evidence labels:[M] census,[V] source/provenance,[I] coverage argument.
The next highest-information branch must identify information loss at a different
interface with a distinct separating test; no such pilot is admitted by this
measurement. No new annotation, GPU use, TEST access or active-job polling.

## Cycle9 — existing DGS linguistic annotations, with a permission boundary

Date / commit:2026-09-22 / unchanged base HEAD. Previous turn: PROGRESS,
bounded frame-omission falsification. Research question: is there an existing
DGS linguistic annotation resource that could supply a better specified natural
distinction than PHOENIX gloss/text disagreement, without new annotation?

Evidence inspected: official Public DGS Corpus conventions, data-statement
annotation sections, a Release4 transcript and posted English/German terms.
[Source record and reading limits](evidence/PUBLIC_DGS_RESOURCE_SCOPE.md).
Result:[V] existing parent/subtype and mouthing annotations are documented;
the portal is real. Important counterevidence: mouth targets are not literal
lip transcripts, intervals are not exact articulation times, and same-parent
tokens are not verified mouth-only minimal pairs. No model failure is measured.

Decision: conditional resource feasibility lead for Q17, not a method. Posted
linguistic-research terms do not explicitly settle this project's use. Ask about
existing project permission before acquisition/model use; do not send external
messages or download datasets. This is not another DGS reviewer request or
new human annotation. Scientific admission remains unchanged.

Next bounded action if use is cleared: inspect a small unchanged annotation
export for original IDs, tier links and usable held-out grouping. Otherwise
leave this resource branch inactive; its permission issue does not establish
a global research barrier. No training, GPU, TEST or active-job checks.

## Cycle10 — falsify raw-score drift as a rank-harm indicator

Date / commit:2026-09-22 / `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.
Previous goal turn: NO PROGRESS—the permission check restated the existing
Cycle9 boundary without new method evidence. That is not a verified wait or a
global blocker. Re-read the attachment and proposal9 guide, current hypothesis
map and blacklist. Bounded current-paper queries mostly rediscovered existing
SAN/CMCM/SCL-SLT/SignMatch leads; no new frontier or novelty claim.

Research question: can raw unpaired-score movement identify harmful ranking
drift (Q15/Q16)? Hypothesis under test: competitor score increases suffice to
indicate worsening competition. Alternative: paired score changes faster,
or scores move together without changing ranks. This differs from Cycle2's
cross-seed selected-snapshot persistence count; it uses fixed within-run times.

Diagnostic: [pre-count protocol](evidence/SCORE_DRIFT_PROTOCOL.md), six retained
fusion matrices at111/666, three seeds, official DEV IDs, both directions.
Read-only CPU analysis; no model inference or labels added. All metrics/ranks
replay exactly. [Result](evidence/SCORE_DRIFT_RESULT.md):128–167 initial maximum
competitors per seed/direction increase their score while paired margin improves;
15–38 decrease while margin deteriorates. An algebraic global-shift control
also preserves both directional probabilities/ranks. No claim of encoder-wide
shift realizability or a refutation of SCL-SLT's algorithm follows.

Decision: close the raw-drift sufficiency inference, not all contrastive
optimization. No miner, calibration, gradient protection or pilot admitted.
New evidence:[M] endpoint quadrants, [R] exact saved-score replay and deterministic
analysis replay, [V] score-level algebra, [I] limits on harmful-negative labels.
Six dedicated unit tests pass; two full analyses produce byte-identical JSON.
Combined focused suite:22/22 pass;20/20 recorded source hashes revalidated.
One initial attempt to display a large single-line metrics JSON was truncated;
analysis uses complete programmatic parsing and exact rank checks instead.

Next scientific requirement: a distinct causal intervention with a separating
test; raw trajectories cannot satisfy it alone. MY DGS acquisition remains
conditional, but did not block this diagnostic. No human annotation, GPU,
TEST access, job polling, method implementation or SOTA claim. Goal stays active.

## Cycle11 — independently check an existing PH handshape resource

Date / commit:2026-09-22 / unchanged base HEAD.
Previous turn: NO PROGRESS, permission question repeated without a new finding.
The duplicate automatic continuation messages do not count as separate completed
goal turns. MY DGS uncertainty is not sufficient to declare the whole goal blocked.

Research question: can existing handshape labels provide an independent natural
visual distinction for Q04/Q17 without new annotation? Bounded primary-source
search led from an August2026 handshape preprint to the2023 PHOENIX14T-HS paper.
[Evidence and reading limits](evidence/HANDSHAPE_RESOURCE_SCOPE.md).

Hypothesis screened: a PH-compatible handshape resource establishes a usable
video-level cue label. Counterevidence: documented annotation limitations defeat
that assumption; source availability and video-level validity are separate gates.
No corpus payload, checkpoint or raw video was downloaded, and no labels were
generated or collected. Release landing-link failure is recorded as a browser
observation, not proof that the resource cannot exist elsewhere.

Decision: close this specific independent-cue shortcut; retain the resource as
weak-supervision prior art with its original limits. No generic extra hand stream,
CTC head, dictionary teacher or relational module is admitted. Existing closures
are unchanged; this is not a universal handshape-information impossibility claim.
New evidence:[V] documented provenance and adjacent prior,[I] inability to use
that evidence as the requested independent diagnostic truth. No efficacy measured.

Next requirement remains a genuinely separating causal test under permitted
evidence, not another availability search for the same weak labels. Goal remains
active and incomplete. No GPU, TEST, deferred-job polling or external contact.

## Cycle12 — remove an overbroad self-imposed admission condition

Date / commit:2026-09-22 / unchanged base HEAD.
Previous completed cycle: PROGRESS in bounded source verification, not efficacy.
Current question: did the local research notes accidentally make independent
semantic labels a prerequisite for all further experiments?

Inline academic-research-suite adversarial review compared the next-action
language with user-guide sections1/23/41. [Audit](evidence/ADMISSION_SCOPE_AUDIT.md).
Finding: files00/09 overgeneralized a validated-natural-contrast requirement;
Q24 also needed a scope qualification to avoid a universal PH-first reading.
The original guide allows computational/statistical bottleneck hypotheses,
with appropriate evidence and controlled tests, without establishing video
meaning. Its six pilot gates remain intact.

Action: corrected00/06/07/09 and documented the original error, counterargument
and claim-specific evidence boundaries. Rechecked P1–P6 against the original
gates; no status changes or implementations. Independent semantic support is
still needed for those particular semantic attributions, and existing labels
must not be recast as video relevance truth. No new human annotation or AI
semantic-label substitute. MY DGS permission stays resource-specific.

Result: PROGRESS as a protocol correction only; no new measured mechanism or
efficacy evidence. No CPU experiment, training, GPU, TEST, deferred-job check,
external contact or new literature verification. No next empirical test is
claimed specified or executed. Goal remains active and incomplete.

## Cycle13 — bounded computational-mechanism screen, no admitted lead

Date / commit:2026-09-22 / unchanged base HEAD.
Previous turn: PROGRESS through the admission-scope correction, not efficacy.
Current work reread the attachment, guide's mechanism/gate sections, question
map, closure registry, relevant SEDS source and targeted historical records.
No human annotation or new semantic judgments were requested or generated.

The anatomical-coordinate, pooling and channel-objective routes examined here
encounter already documented designs/controls. A bounded primary-source search
added a [pooling-prior qualification](evidence/POOLING_PRIOR_TRANSFER_SCREEN.md),
including a displayed-equation caveat checked algebraically against the rendered
text. No new experiment or author-code failure is claimed. Broad discovery
also rediscovered existing SLRet/dictionary-retrieval sources; it did not reveal
a verified stronger fair sentence-retrieval comparator.

Current classification: **NO PROGRESS toward mechanism admission**. This is
not a verified wait; no process was checked. No CPU model experiment, GPU,
training, asset download, deferred-job polling or TEST evaluation. Historical
TEST asset metadata was encountered in records, not used for selection; scope
is disclosed in the source note. No code change or numerical test run.

Primary method, implementation and controlled efficacy remain outstanding.
The goal is incomplete and active. Difficulty formulating a defensible new
mechanism is not itself a user-input blocker or a reason to mark completion.

## Cycle14 — paired TRAIN-anchor construction screened before extraction

Date / commit:2026-09-22 / unchanged base HEAD.
Previous turn: NO PROGRESS toward mechanism admission. The automatic continuation
received during this turn does not create another completed turn or blocker count.
Attachment reread; guide section41 and current state rechecked. Inline skill
review scoped a concrete paired-bank hypothesis and challenged both its alleged
novelty and an overbroad candidate-offset dismissal.

[Screen](evidence/PAIRED_ANCHOR_SCREEN.md): linear anchor coordinates reduce to
a fixed bilinear map; a two-dimensional counterexample excludes universal
additive-offset reduction. Primary RELIT method text supplies a direct generic
prior collision. Neither a PH neighborhood-correspondence bottleneck nor a
distinct causal intervention is established. Reject this specification before
feature extraction; no new semantic labels, pseudo pairs or pilot.

Classification: bounded screening PROGRESS, not empirical/method progress.
The next action changes: do not recreate representations to try this generic
formula or an anchor/kernel sweep. `artifacts/proposal7/phase2` was checked and
is absent; this operational observation is not the scientific rejection reason
or a global blocker. No GPU, model execution, TEST, deferred-process check,
external contact, numeric test run or implementation. Goal remains incomplete.

## Cycle15 — measure duplicate-wrist agreement on all recovered TRAIN poses

Date / commit:2026-09-22 / unchanged base HEAD.
Previous turn: bounded screening progress, no admitted method. Current question
changes causal layer to detector-internal side consistency, not another anchor,
pooling, frame-omission or frozen-readout test. Original extraction/config and
actual checkpoint landmark names establish the coordinate contract.

[Protocol](evidence/WRIST_CONSISTENCY_PROTOCOL.md) registered before counts;
[result](evidence/WRIST_CONSISTENCY_RESULT.md). All7096 TRAIN poses and metadata
inventory verify against recovery hashes. CPU census finds0 flagged frames out
of653774 eligible retained frames, coverage6944/7096 clips. Locked NO_GO for
frequent sustained root/body-wrist disagreement; no threshold relaxation.
Both independently invoked executions exit0 with byte-identical reports.
Eight pre-census fixtures passed; two post-census boundary fixtures add coverage
without changing the census. Source hashes and checkpoint mapping verified.

PROGRESS: new bounded empirical evidence closes a specific proposed input-error
lead. It neither supplies a primary method nor demonstrates detector correctness
or retrieval efficacy. Correlated errors and low-confidence occlusions remain
outside inference. No human annotation, semantic labels, GPU, DEV/TEST, deferred
process polling, external contact or input/model change. Goal remains active.

## Cycle16 — batch-competition lead checked against omitted C18 evidence

Date / commit:2026-09-23 / unchanged base HEAD.
Previous goal turn: **NO PROGRESS**; it only restated the resolved `xong`
handoff. No verified wait or new user-input blocker was present.

Question: does insufficient in-batch competition support a distinct method?
The [screen](evidence/BATCH_COMPETITION_SCREEN.md) locates C18's already completed
physical-B64 experiment, absent from this track's reconstructed blacklist.
Current JSON/schedules verify identical TRAIN/DEV IDs, inherited resource
digests,5120 ordered sample exposures, but80 versus160 optimizer updates.
Two selected fusion matrices reproduce all recorded metrics and per-query
records exactly. B64 mean78.227360 is below control78.323699. This is a
historical recipe outcome, not a new run or causal estimate of negative count.

Official GradCache and gradient-bias publication abstracts establish adjacent
generic prior collisions within their read scope. Reject generic larger/cached
batch as the new method; do not erase C18's attribution limit, reopen its sweep,
or broaden C17's explicit DCL closure into all native batch experiments.

Classification: bounded screening/artifact **PROGRESS**, no new mechanism
admission or efficacy. The negative registry and Q16 now point to the missing
result and its limits. No new training/code, GPU, TEST, human/AI annotation,
deferred-job check or external contact. No primary method has survived yet;
remaining work requires a distinct failure/intervention, not another generic
negative-pool statistic. Goal remains active and incomplete.

## Cycle17 — versioned published frontier contracts

Date / commit:2026-09-23 / unchanged base HEAD.
Previous turn: bounded screening/artifact PROGRESS, no admitted method.
This turn addresses the explicit incomplete frontier deliverable rather than
repeating a training or feature-extraction campaign.

[Source report](evidence/FRONTIER_CONTRACTS_20260923.md) and
[11-row snapshot](evidence/published_frontier_20260923.csv): primary SEDS,
C2RL, UPRet and SAN tables inspected; resource cards replace generic pending
labels with concrete counts, architectures, supervision and remaining gaps.
SAN-GFSLT coarse results now accompany SAN-CiCo. Latest-task search corroborates
SignMatch exclusion; no new standard sentence-frontier result was verified.
C2RL final publisher access and CMCM full numeric tables remain unresolved.

Interpretation: published-reference comparison is direction- and resource-
dependent. Reported counts are not verified ID sets, TEST results are not DEV
comparators, and an envelope across models is not one achieved configuration.
Alternative: reported count discrepancies may include typos; do not infer
leakage or actual membership differences without manifests.

PROGRESS toward required frontier documentation, **not mechanism admission**.
No new primary proposal, method implementation, model execution, GPU, local
TEST access, annotation, external contact or deferred-job check. No global
blocker: method discovery can use the existing local control. Goal remains
active and incomplete; no SOTA claim is authorized by this table.

## Cycle18 — exact CSL text-input identity census

Date / commit:2026-09-23 / unchanged base HEAD.
Previous turn: frontier-documentation PROGRESS, no admitted method. The initially
considered within-group performance decomposition already existed in PMGR;
it was not rerun. The replacement question concerns exact input identity,
not semantic labels or a population-risk training rescue.

[Protocol](evidence/CSL_TEXT_COLLISION_PROTOCOL.md) registered before counts;
[result](evidence/CSL_TEXT_COLLISION_RESULT.md). Six fixtures passed. Two
independent CPU executions exit0 and produce byte-identical reports. All7395
groups match native text arrays; source hashes and manifest memberships verify.
DEV has0/797 participating groups, below the locked10% gate. TRAIN has52/6598;
cleaning/BPE adds one pair and uniform selection adds none in either split.

Bounded empirical **PROGRESS**: reject a dominant exact-input-collapse lead
on this DEV population. Not evidence of faithful translation, video meaning,
encoder quality or a retrieval gain. No threshold rescue, new method/pilot,
training, GPU, local TEST access, human/AI semantic annotation, deferred-job
check or external contact. ARS supplied preregistration and replay discipline.
The main method/SOTA goal remains active and incomplete.

## Cycle19 — reject an inactive BatchNorm coupling premise

Date / commit:2026-09-23 / unchanged base HEAD.
Previous completed turn: Cycle18 empirical PROGRESS; zero CSL DEV text-input
collisions changes the next action. The automatic continuation received during
this cycle does not create an additional completed-turn or blocker count.

Question: does batch-statistics coupling in the adapting pose GCN establish
a computational bottleneck? [Screen](evidence/GCN_BATCHNORM_SCREEN.md) traces
the real masked clean-control route, not the inactive LoRA alternative. Current
per-step mode restoration leaves GCN BN in eval and fusion trainable. All three
retained run configs select this control; source hash differences explicitly
limit retrospective claims. Recorded-commit lookup cannot recover masked_pose.

Three new native-temporal-block CPU fixtures check companion substitution,
train-mode sensitivity and weight updates with fixed buffers. Initial fixture
positive control failed due to ReLU clipping; companion values were corrected
and the failure is retained in the report. Final combined suite6/6 passes.
No retrieved samples or model checkpoints enter this check.

Bounded source/diagnostic **PROGRESS**, not method efficacy: stop a BN coupling
repair before activation extraction or training. Frozen-statistics bias, temporal
padding effects and contrastive gradient dependence are distinct and unmeasured.
No semantic labels, TEST, GPU, deferred-job polling or external contact. ARS
counterevidence discipline prevents treating a trainable parameter as a train-mode
module. No primary method is admitted; the goal remains active and incomplete.

## Cycle20 — active temporal boundary dependency in body GCN

Date / commit:2026-09-23 / unchanged base HEAD.
Previous turn: bounded source/diagnostic PROGRESS. This cycle tests a different
active mechanism: batch-max zero-pose extension before temporal convolution,
not BN statistics, score masking or floating-point batch-row variation.

[Protocol](evidence/GCN_TEMPORAL_PADDING_PROTOCOL.md) precedes four fixed TRAIN
model-output comparisons; [result](evidence/GCN_TEMPORAL_PADDING_RESULT.md).
Three fixtures pass. Selected checkpoint and pose hashes verify; full body state
strict-loads; two CPUfloat64 runs exit0 and produce byte-identical JSONs.
All four samples change at exactly their final four valid frames, relative L2
6.57–18.26%; interior differences≤1.34e-15; extension4/8 results identical.

Empirical **PROGRESS**: observed computational dependency warrants downstream
attribution, not a retrieval method pilot. Three-part pose/sign_conv and final
scores/ranks remain unmeasured. Next action is that fixed propagation check,
not a padding-policy search or another generic normalization hypothesis.

Also found all three historical run source snapshots: nine relevant snapshot
hashes match run reports. Two unique trainer versions preserve selective eval;
Cycle19 receives a source-provenance addendum, not a fresh runtime claim.

No human/AI semantic labels, new training, GPU, TEST, external contact or
deferred-job check. ARS provided pre-execution scope, exact replay and11/11
fallacy scan. No primary method/SOTA claim; main objective remains active.

## Cycle21 — padding effect survives native pose window processing

Date / commit:2026-09-23 / de2f07ecac9ce65563316fde0860f22e4d585acf.
Previous turn: Cycle20 empirical PROGRESS. Current HEAD changed externally;
this cycle makes no commit and preserves existing working edits.

[Protocol](evidence/POSE_PADDING_PROPAGATION_PROTOCOL.md) fixes the same four
TRAIN inputs/checkpoint. Native loader, complete Sign_Bert and get_sign_output
execute on CPUfloat64. All13 selected boundary windows change after sign_conv
and pooling; interior/invalid controls and extension4/8 comparisons stay exact.
This rejects erasure at that stage, not downstream score/rank stability.

[Result](evidence/POSE_PADDING_PROPAGATION_RESULT.md): three fixtures pass;
initial capture omitted concatenation summaries, so v1 and its source are
preserved and instrumentation amended. v2/v3 preserve all primary outcomes;
exact cross-version equality fails for auxiliary norms at≤5.56e-17 despite
contiguous materialization. No further numeric rescue. Two final v3 runs are
byte-identical; all23 current hashes verify. Original body cross-check tolerance
passes. These different verification scopes are not conflated.

Empirical **PROGRESS** toward attribution. Next step is fixed-input final-score
sensitivity before any gallery/ranking claim. No method/pilot, ordinary repair
sweep, semantic labels, training, GPU, TEST, external contact or deferred-job
check. ARS supplied frozen scope, replay and11/11 fallacy scan. Goal active;
no primary method or SOTA demonstrated.
