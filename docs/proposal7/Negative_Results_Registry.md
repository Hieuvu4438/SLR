# Negative Results Registry — Proposal 7

Status: **mandatory closure registry established before fresh literature/candidate ideation.** All proposal reports, implementation specifications and historical checkpoints have been read; targeted implementation call paths and actual negative gates have been inspected. Implementation detail checks continue as needed; no claim of exhaustive verification of every repository line.

## 2026-09-16 user-authorized selective reopening amendment

The user now explicitly allows previously closed but inconclusively tested ideas
to be reconsidered with explicit justification and small controlled pilots.
This supersedes the blanket prohibition ONLY for individually documented exceptions;
it does not erase historical failures or grant automatic eligibility to every row.
Unreopened rows remain closed. TEST/SEDS-asset/relevance constraints and method GO
standards remain unchanged. A pilot lead is not novelty or a final GO.

First exception: [CICO-REOPEN-01](evidence/autonomous_search/CICO-REOPEN-01_protocol.md),
learned sentence-conditioned outer clip weighting on frozen CiCo. Prior generic
query-gate closure is selectively lifted for this screen. C24/C25's failed
inference-only inner pooling/entropy operators are not repeated. Explicit controls:
unchanged initialization/B0, query-independent head and input-hashed random TRAIN
conditioning head, equal parameters/steps/batches. No ELSC support/lexical loss,
pose assets, OT, new miner or broad horizon/recipe rescue is included.

Outcome: [CICO-REOPEN-01](evidence/autonomous_search/CICO-REOPEN-01_result.md)
completed260updates/head, all three selectors retain initialization. Sentence
head final74.759152meanR1 versus75.240848CiCo. This exact reopened specification
is now unsupported; do not scale or tune it as a rescue. Other exceptions still
require individual justification; selective authority is not revoked by one failure.

The original guide closed every method and conceptual variant in proposals1–6.
Historical “GO”, “promising”, “pending” or “blocked” statements cannot reopen them;
the explicit amendment above is the new authority for specific exceptions.
Absence of surviving experiments means the failure mechanism is [U], not a
demonstrated success. No experiment here proves a whole family can never work.

Evidence labels: [V] inspected project text/code; [M] locally recomputed evidence; [I] inference; [H] proposed hypothesis in the historical document; [U] not established. These labels do not certify citations in older reports; primary-source verification is a separate remaining phase.

## Named method closures

| Family / source | Central hypothesis, modules and supervision | Intended benefit | Observed evidence and failure mode | Confounders / limits | Lesson and concepts that must not return |
|---|---|---|---|---|---|
| **ELSC**, proposal1 implementation §§8–18 | [H] Local sign-aware lexical margins should improve sentence ranking. Frozen teacher selects stable support; pointwise residual adapter and lexical head; isolated-word text bank, visual-neighbor negatives. Full variant removes evidence/control RF windows; optional ranking KL. CLCL remains deployment objective. | Ground lexical discrimination; protect coarse retrieval. | [V] Saved three-seed Gate G and M both `no_go`. Mean bidirectional dev R1 delta +0.096339 pp; true support beats matched-caption and random-support controls only 1/3 seeds. [M] Min seeds 1337/2026 selected initialization (epoch −1); trainer explicitly includes it before training. Lower-LR +0.256904 pp and generic local word–video +0.224791 pp also below their 0.5 pp gate. | PH H2S-transfer-aware features, release initialization, limited dev queries; baseline continuation and auxiliary budgets differ in some comparisons. No causal proof that localization is universally useless. Full not required to be reopened because user closes the family. | Auxiliary loss reduction is not evidence of deployment gain. Do not recycle support mining, lexical margins, evidence deletion/invariance, reliability gates, teacher changes, preservation losses or the same adapter at another layer. |
| **DIVE-SLR**, proposal2 reviewed proposal §§5–7 and full spec | [H] Two real contradictory pairs should isolate rival-specific local evidence. Local RGB+pose teacher warm-up; four directional margins; differential support with absolute gates and two-view stability; centered residual `S0 + gamma(E_student−E_ref)/2`; full-score bridge loss; human-accepted numeric/lexical bank. | Transfer local discrimination to the actual full-caption score. | [V] Status reports fixture and integration-contract tests, but real native pipeline stops at missing SEDS resources/parents. [U] No successful benchmark causal test established by those fixtures; precise researcher-reported unsatisfactory-run mechanism not recovered. Family CLOSED by current user authority. | Main prior design depended on SEDS pretrained assets, now forbidden; bank acceptance required genuine sign/semantic review. Resource failure must not be presented as measured model failure. | No bounded evidence residual, teacher/reference subtraction, same-pool quartet bridge, rival-specific support, numeric-pair or audited-lexical remix. Do not pursue missing SEDS downloads. |
| **PLEL**, proposal3 full report | [H] Real near-paired captions can train an inference-local discriminative scorer. Frozen CiCo, pointwise local MLP, frozen isolated phrase text, log-mean-exp bag score and residual blend; reference warm-up, shared support and four directional margins. | Avoid synthetic-caption shortcuts and align training discrimination with inference. | [U] Detailed negative run trace not yet located; proposal is design, not evidence of gain. User closes family. | Pair coverage, linguistic cross-negative validity, head capacity and residual scale could confound outcome; these are hypotheses, not retrospective measurements. | Moving from synthetic to real pairs or isolated words to phrases does not create a new causal mechanism. No local paired-evidence reranker. |
| **OCEM**, proposal4 checkpoint/report and actual Stage-A gate | [H] False matches overuse shared temporal evidence. Pre-contextual local projection; physical RF atoms, overlap matrix, entropy-regularized real/null assignment under shared-time capacities; zero-score centering; inference blended with CiCo. | Prevent repeated counting of overlapping temporal evidence. | [V] `methods/ocem/runs/wp09_stage_a_phoenix_rerun1/failure_gate_A.json`: `NO_GO_SCIENTIFIC`. Targeted stratum 21/259 errors = 8.108% (<10% gate); adjusted concentration effect +0.0006947, 95% CI [−0.0431487, 0.0448006], 245 matched pairs. Old proposal4 “GO WITH CONDITIONS” is superseded. | Contextual concentration is a proxy; matching adjusts measured length/lexical variables, not all confounders. Gate rejects investment rationale, not every possible transport model. | No overlap/capacity/null/coverage rebrand, alternate solver, shifted constraints or physical-support weighting as the next central contribution. |
| **SSSC / Shared-Support Sign Contrast**, proposal5 Method1 | [H] Compare changed text spans on exactly the same reference-selected visual support. UPRet-family baseline, train-only sign-aware miner, shared positive support, bounded discrimination margin; no extra inference branch. | Prevent each alternative from finding its own favorable evidence. | [V] Experiment notes describe an invalid RNG-order run stopped at 870/2600 updates, then a corrected parity path. [U] A completed attributable efficacy result is not established by that log. User closure is definitive. | RNG/exposure differences invalidate causal comparisons; smoke recall is not benchmark performance. Current process inspection found no continuing training. | No shared-support-versus-independent-support loss, reselected teacher, focal/margin/uncertainty dressing, or generic local-hard-negative variant. |
| **Sampling-consistent partial alignment**, proposal5 Method2 | [H] Retrieval should depend on evidence, not overlapping observation count. Video coverage masses, word/BPE masses, fixed real matched mass, two dustbins, centered partial OT, EMA correspondence projection across samplings. | Reduce sampling-sensitive matching. | [U] Detailed empirical failure mechanism not recovered; user closes conceptual variants too. | Teacher consistency can preserve wrong alignment; physical overlap invariance does not establish linguistic relevance; no retained run establishes causal effect. | No sampling-consistent OT, learned dustbin, EMA alignment stability, inverse-overlap capacity, alternative mass prior or consistency-only rescue. |
| **PMGR**, proposal6 report/spec and CSL Phase-B report | [H] Train against the actual grouped-gallery risk instead of one sampled performance. The implemented/specification contract uses all current performances, max of the mixed score, population-weighted V2T and exact two-level replay; no stale cross-step representation memory. Rank extension is conditional. | Improve bidirectional full-gallery ranks by matching deployment population. | [V] `artifacts/pmgr/csl_phase_b_seed0_report.json`: no-go. C4 mean R1 65.71725 vs strongest equal-input C3 65.60809, delta +0.10916 pp (<0.5 pp); R5/R10 decline versus C3. 797 sentence groups/1077 dev videos, 240 updates. | One seed short pilot; common masking and resource regime differ from earlier CSL ELSC runs. Do not compare raw scores across those regimes as a method effect. | Population mismatch can exist without enough useful causal headroom. No gallery-risk, smooth-rank, group-marginal, all-performance-memory or negative-pool reformulation of this mechanism. |
| **RPCA**, proposal6 report | [H] Use sentence-context adaptation without damaging deployment retrieval. Training-only sentence decoder and shared raw visual adaptation; project proposed updates using both retrieval directions in actual Adam step coordinates. | Import context while controlling retrieval-loss conflict. | [U] Surviving causal run evidence not established; user closure overrides prior conditional recommendation. | Local first-order protection does not guarantee ranking improvement; decoder and optimizer changes can confound comparisons. | No generation/context + retrieval preservation, gradient surgery, bilevel protection, teacher anchoring or distillation remix as the primary idea. |

## Conceptual closure surface (also binding)

The registry records semantics rather than inventing an unsupported one-to-one numbering of R1–R5. Proposal6 explicitly identifies sampling-consistent teacher OT as R5; all historical R1–R5 and their descriptions remain closed.

| Prior location | Closed variants / mechanisms | Missing empirical explanation |
|---|---|---|
| Proposal1 strategy and implementation alternatives | Reliability-aware RGB/pose fusion; clip/body/query gates from confidence/missingness/motion; generic objective-conflict control; signer/graph invariance; hard monotonic alignment; unlimited LLM paraphrases; scale-only backbone changes. | [U] Variant-specific logs not recovered. User closure is sufficient. |
| Proposal4 candidates C2–C5 | Rival-Conditional Evidence Scoring (RCES): remove common real-caption content and train a query/video student. Boundary-Marginalized Retrieval (BMR): fixed within-clip span ensemble/logsumexp. Relation-Consistent Late Interaction (RCLI): unary alignment plus fused-GW relation penalties. Reliability-Bounded Positive Sets (RBPS): cross-fitted soft positives. | [U] Separate run mechanisms not recovered. These were alternatives in old docs but are now closed, not fallback candidates. |
| Proposal6 candidates C3–C8 | Equivalence/group positives; query-bank score calibration; nuisance removal; generic temporal self-supervision; distributional soft-positive matching; global composition/order. | [U] Variant-specific negative evidence incomplete. No generic relaunch on the same hypothesized bottleneck. |
| Across proposals | CiCo+SEDS, UPRet+hard negatives, C²RL+pose, generic causal/multigrained stacks, extra streams, local/global alignment by itself, uncertainty/teacher/reliability decoration. | Adding components cannot establish semantic distance. |

## Rules for the later candidate screen

1. Remove names and draw the causal path: what observed error is changed, by which intervention and supervision, and why should full-gallery ranks improve?
2. Compare that path against every row, including conceptual alternatives. A different loss, teacher, layer, sampler, stream or optimizer is insufficient.
3. Reject when the primary explanatory mechanism is the same. A diagnostic revealing an old bottleneck does not override the user's closure.
4. Independent novelty requires both an internal collision check and fresh primary-literature/code verification. Search absence alone is not novelty proof.
5. Each proposed causal claim needs a control that can refute it. No guarantee of SOTA; a well-supported no-go is preferable to a renamed closed method after the required search cycles.

## Evidence links and audit limits

- [Saved gate extraction](evidence/historical_gates.json), [dev run ledger](evidence/dev_run_ledger.json), [independent PH dev recomputation](evidence/ph_dev_residuals.json).
- [Workspace/source hashes](evidence/workspace_snapshot.json), [reading progress and pending files](AUDIT_PROGRESS.md).
- The registry is sufficient to prevent accidental reopening, but it does **not** declare the remaining complete source/spec/config/test audit finished. Detailed entries will be amended only with directly inspected evidence, without reopening closures.
