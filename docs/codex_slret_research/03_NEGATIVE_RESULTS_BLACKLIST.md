# Mechanism blacklist — current-goal reconstruction

2026-09-22, repository `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.
Created before any new candidate design in this track. The authority is the
current [proposal9 guide](../proposal9/CODEX_SLRET_AUTONOMOUS_GOAL.md), not
historical permission to selectively reopen ideas. Closures apply to the named
mechanisms and close variants; they do not establish universal impossibility.

Evidence labels throughout this directory: [V] directly inspected source or
artifact; [A] author-reported external result; [R] locally reproduced under a
recorded run; [M] newly measured; [I] interpretation; [H] hypothesis; [U] missing
or insufficient evidence. A retained historical metric is identified explicitly
as such, even when its file is verified today. Hash verification is not rerunning
the experiment.

## Revalidated original records

[M] The [state audit](evidence/state_audit_20260922.json) reopened all **12**
original paths in proposal7's `historical_gates.json`. All 12 files still exist
and match the archived SHA256. This establishes that these particular negative
records survived subsequent storage changes. It does not certify all later
`artifacts/slret_goal/` files, which the dataset-first state reports as removed.

| Closed mechanism | Causal pathway | Evidence and limits | Excluded rebranding |
|---|---|---|---|
| ELSC | Teacher-selected lexical support → auxiliary margin → pointwise visual residual | [V] Original three-seed G/M gates survive; recorded mean R1 increment +0.096339 pp, support specificity only 1/3 seeds. Initialization can be selected. | Teacher/miner replacement, different margin, reliability, evidence deletion, same supervision at another layer |
| DIVE-SLR | Rival-specific local RGB/pose support → reference-subtracted score residual and quartet margins | [V] Historical registry records an integration/resource stop; [U] attributable full-gallery efficacy. Closed by scope, not proved empirically harmful. | Bounded local residual, rival evidence, numeric-pair remix |
| PLEL | Real paired captions → local phrase score → blended retrieval | [V] Design exists; [U] complete negative-run attribution. Closed by scope. | Real instead of synthetic pairs, phrase instead of word, local reranker |
| OCEM | Physical support overlap/capacity/null assignment → corrected score | [V] Original Stage-A `NO_GO_SCIENTIFIC` survives: 21/259 errors affected, 8.108% <10%; adjusted effect CI crosses zero. Contextual index proxy limits interpretation. | New transport solver, dustbin, mass prior, overlap/occupancy weighting |
| SSSC | Changed text spans compared on teacher-selected common visual support | [V] Historical RNG-order failure; [U] completed attributable efficacy. Closed by scope. | Shared-support contrast with uncertainty, another teacher or margin |
| Sampling-consistent partial alignment | Coverage/BPE masses + partial OT + teacher consistency | [U] complete causal result; user closes family. | Attention/OT substitution, EMA/dustbin/mass change |
| PMGR | Complete groups/current performances → population-weighted gallery risk | [V] Phase-B report retained separately under `artifacts/pmgr/`; historical +0.10916 pp versus matched C3 <0.5 pp, R5/R10 declines. One short seed. | Group-risk, all-performance memory, smooth-rank or negative-pool reformulation |
| RPCA | Sentence decoder → adaptation with protected retrieval updates | [U] attributable completed efficacy; closed by scope. | Decoder/distillation/gradient surgery/optimizer-coordinate protection |
| R1–R5 conceptual alternatives | Prior proposal6 and cross-proposal alternatives | [V] Semantic descriptions in proposal7 registry; do not invent a one-to-one numbering. | Generic local/global composition, equivalence positives, nuisance removal, soft positives, context preservation |

Sources: [proposal7 registry](../proposal7/Negative_Results_Registry.md),
[retained gate ledger](../proposal7/evidence/historical_gates.json),
[later mechanism registry](../../research/slret_goal/NO_GO_REGISTRY.md).

## Later exclusions and qualifications

The later registry is binding evidence against repeating the exact tried recipe.
The following are inherited recorded results, not newly rerun efficacy tests:

| Family | Recorded outcome | Scope of exclusion |
|---|---|---|
| Scalar pooling / learned query clip weights | AS-C24/C25 and CICO-REOPEN-01 fail; reopened sentence head selects initialization | No weight/temperature/conditioning rescue |
| Frozen residual readouts | Multiple probes fail; AS-C42 completes 8,800 updates without adequate held control | No longer-budget/optimizer/seed rescue; **not** proof information is absent |
| Spatial-grid readout | Crossed controls fail attribution | No generic extra-grid/position stream disguised as new evidence |
| Geometry/offset calibration | Fixed mixtures/centering/assignment controls fail or depend on evaluation cohort | No cohort-query labels or fixed-bank calibration claim |
| Masking-only | Inference exclusions lose | A necessary engineering correction could be justified; it is not novelty |
| Temporal reversal/sampling | Prior reversal/averaging/consistency interventions fail | Do not treat shuffled videos as linguistically verified negatives |
| Nuisance/source/signer removal | Attribution unresolved and generic routes closed | No signer/source invariant module from correlation alone |
| Text/prefix repair | Low measured burden; held contrasts inadequate | No multilingual/character/prefix rescue without a distinct observed defect |
| Augmentation-off continuation | Six matched runs, no seed gate passes | No swap-strength/LR/horizon rescue |
| Ensembles/merging | A stronger ensemble control exists; tested merging fails | Keep as comparator, not new contribution |
| Gallery graph | Real/shuffled graph controls fail | No learned graph/strength/k rescue |
| Exact gloss equivalence | Strict persistent-confuser burden below prior 10% gate | No altered relevance or fuzzy-gloss mining; equality does not establish semantic equivalence |
| UPRet max→sum transport reduction | Small amplitude-controlled gradient differences at a partial checkpoint | No training campaign from this mismatch alone |
| CMCM numerical/covariance repair | Source defects verified historically, retrieval impact unresolved | No ordinary repair-as-method claim |

Current dataset-first IDs **are not** proposal7 AS-C IDs. In particular,
dataset-first C27 means gloss CTC; historical AS-C27 concerns another mechanism.

| Recent recipe | Current disposition and reason |
|---|---|
| V2 C17 DCL | User-closed; incomplete blend endpoint remains unresolved, not a valid completed negative |
| V2 C18 native physical B64 | Selected78.227360 versus exposure-matched B32 control78.323699; exact recipe deferred, no larger-batch sweep. 80 versus160 updates prevents negative-cardinality attribution. [Cycle16 artifact check](evidence/BATCH_COMPETITION_SCREEN.md) verifies schedules and replays selected scores; not a ban on all batch-composition research. |
| V2 C20 dominant-stream dropout | Timed out before terminal schedule; do not convert intermediate loss to completed failure |
| V2 C21 canonical 3D motion | Exact recipe selects initialization; defer without refinements |
| V2 C22 phase scale/shift | −0.170817 pp against matched clean GCN, despite improvement over release initialization |
| V2 C23 hierarchical graph adapters | −0.192678 pp against matched GCN; directional tradeoff |
| V2 C24 UniFormerV2 complement | User-deferred; no efficacy conclusion; associated extraction has a no-poll handoff |
| Dataset-first C25 CoSign-LI | User-deferred before efficacy; not a measured failure |
| Dataset-first C26-A frozen Uni-Sign global head | Recorded CSL DEV mean 43.700670; weak standalone recipe. Original job artifacts reported absent; do not relabel as currently replayable. |
| Dataset-first C27 gloss CTC | **OPEN**; no efficacy result or established order bottleneck; see collision and falsification audit |

## Admission rule

For each proposed mechanism record observed failure → actual information input
→ changed representation/parameters → deployed rank effect. Compare this path to
every applicable row. Different supervision may create a testable distinction,
but is not sufficient evidence of novelty or efficacy. If a separating control
does not exist, reject conceptual equivalence. Never erase a negative result
because its logs are inconvenient or another method sounds promising.
