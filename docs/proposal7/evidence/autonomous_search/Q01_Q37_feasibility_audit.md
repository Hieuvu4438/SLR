# Q01–Q37: current-evidence and feasibility audit

## Material Passport

- Date: 2026-09-16; academic-research-suite, inline evidence synthesis and adversarial review.
- Status: ANALYZED decision audit, **not** an exhaustion certificate or a new experiment.
- AI-assisted; no independent reviewer, human linguistic validation or new external-result replication.
- Authority: [research loop](../../Astra_SLRet_Autonomous_Research_Loop.md), especially §§10, 31, 33, 41 and 43; [closure registry](../../Negative_Results_Registry.md).

## Outcome

All 37 existing questions are retained below. **The evidence does not establish
either a GO method or a global research barrier.** In particular, failed readouts
do not establish information absence; unavailable annotation joins do not make
relations scientifically impossible; a failed scalar mixture does not exhaust
directional geometry. Those are unresolved questions, not successful methods.

The audit also does not identify a new experiment that is already both
high-information and clear of the binding mechanism closures. That is a limitation
of the current formulation work, **not evidence that such an experiment cannot
exist**. It cannot be used as the guide's terminal stopping condition.

The old numeric priority scores are historical judgments, not empirical evidence
of remaining opportunity. Keep them for provenance, but do not launch a run merely
because its parent question once scored highly. Require an explicit decision that
could change under either outcome, a feasible control, and an open causal path.

## Evidence scope and notation

This audit rereads current map/closure text, selected result and limitation
sections, and relevant machine-summary fields. It does not rerun all experiments,
inspect every tensor/step log, or claim a fresh full-paper review of every cited
prior. Result links below are **reports**, not substitutes for numerical run
records. The linked reports identify their protocols, runs, validations and
reading limits; selected machine records are linked directly where needed.
The companion inventory hashes these explicitly linked local files. Existence,
JSON readability and hashes establish traceability, not scientific correctness.

Verdicts apply to the **next action**, not to an entire scientific field:

- **U — unresolved formulation/attribution gap:** no defensible closure and no
  experiment ready to launch. Missing formulation is not impossibility.
- **N — bounded specification unsupported:** retain the negative result; do not
  rescue it by the explicitly prohibited parameter/recipe sweep.
- **C — specified method collision:** an internal closure or documented prior
  blocks that particular proposal. This is not a proof against every mechanism.
- **R — resource/authority prerequisite:** name the missing resource or authority;
  do not infer that it blocks other questions.
- **D — conditional downstream check:** important after a mechanism survives,
  not an independent source of a method today.

Multiple verdicts distinguish, for example, a failed intervention from the broader
unresolved question. No row is classified as globally impossible.

## Full map: evidence, scope and permissible next action

| Question / layer | What was actually tested; current evidence | What remains unestablished | Next permissible action, prerequisites and collision boundary | Verdict |
|---|---|---|---|---|
| Q01 Recoverability / C,E | [C01 summary](AS-C01-R1_summary.json): nine selectors retain initialization. [C05 summary](AS-C05-SUMMARY.json): removing the frozen-score term improves limited TRAIN discrimination, not useful DEV gain; coefficient-one selectors need not all be initialization. [C42 result](AS-C42_result.md): clean held mean 22.472727%, both directions below original 50% adequacy. | Neither representation insufficiency nor a well-powered nonlinear recovery ceiling. The clean residual substrate is inadequate. | Specify a different, positively calibrated information test before training. Same residual-head, optimizer, freeze, duration or seed rescue is not permitted. [SignRep check](SIGNREP_scope_and_feasibility.md) rules out attributing a plain pretrained-backbone swap to information loss. A suitable test has not yet been supplied; do not relabel that omission as a resource impossibility. | U; N for tested probes |
| Q02 Direction-specific geometry / F | [C03 run](AS-C03-DIRECTION_run.json): TRAIN-fitted global shared/directional mixtures reach 74.662813/74.566474 mean R1 versus 75.240848 baseline. | General direction-specific geometry, an unavoidable trade-off, or an independently deployable missing distinction. The strongest-R0-confuser fitting objective is limited. | First state a geometric property not reducible to channel weights, candidate-only offsets, grouped risk or the failed pooling/entropy operators. Derive a refutable consequence using the existing score/token resources. That property is not yet specified; another mixture sweep is not the next experiment. | U; N for scalar mixture |
| Q03 Candidate-relative discrimination / F,J | [C07 candidate screen](AS-C07_candidate_screen.md), [collision screen](AS-C07_collision_screen.md), [C15 result](AS-C15_result.md): cohort headroom survives jitter but uses other evaluation queries. Fixed-bank assignment reduces to column correction; present teacher specification collides with transport distillation; standalone assignment loss is underidentified. | A gap available to one independent query, not one-to-one cohort capacity. | Require a concrete query-and-gallery interaction not covered by those reductions or Q32's failed graph operators. Existing candidate features/scores are available, but no such new operator is established here. Do not train the already-screened teacher. | U; C for three specifications |
| Q04 Pre-pooling spatial information / B | [C02 summary](AS-C02-CROSSED-SUMMARY.json): 30 crossed runs fail attribution/gates. [C44 result](AS-C44_result.md): the added raw readout is permutation invariant; four mixed bins do not isolate articulators. | Recovery of arrangement, actual linguistic relation loss, or an anatomical intervention. | A valid cue-isolating intervention and an open mechanism are prerequisites. Position heads, RF reweighting, generic streams and RCLI cannot be used as rescue. The old readout's negative result does not close arrangement; no alternative isolated-cue resource is certified. | U; N; R for cue attribution |
| Q05 Padding / E | [C14 result](AS-C14_result.md): all seven inference exclusions lose mean R1; text-pad-only loses 1.445087 pp. Some persistent ranks improve under video-pad exclusion. | Benefit of a policy trained consistently from initialization. Inference mismatch is not a training-policy test. | Training-consistent masking is technically implementable with existing code, but currently only an ordinary correction with no surviving new mechanism. Do not spend a new campaign on it as novelty; a distinct supported causal hypothesis would be required first. | N; C for correction-as-method |
| Q06 Temporal order / C,E | [C06 run](AS-C06-TEMPORAL_run.json): pre-context reversal loses 20.905588 pp; post-context reversal is scorer-invariant as recorded in the [state ledger](../../AUTONOMOUS_RESEARCH_STATE.md). | Whether particular valid linguistic order contrasts are lost. Reordered windows are not human-validated minimal pairs. | An existing-label, unchanged-task contrast would need validity/support verification. No new annotations or synthetic linguistic ground truth; no generic order, monotonicity or alignment revival. Valid contrast resource not established. | U; R; N for global blindness |
| Q07 Contextual retention / C | [C13 summary](AS-C13-SUMMARY.json): aligned linear prediction transfers; every tested substitution loses retrieval. | Whether rank-critical semantic information is discarded; reconstruction error is not that measurement. | Formulate a semantic/retrieval-specific consequence with a positive adequacy control, without a map/teacher sweep or a closed lexical residual. Existing pre/post tokens make numerical work possible, but that does not make the missing causal test adequate. | U; N |
| Q08 Anisotropy / C,D | [C11/C12 summary](AS-C11-C12-SUMMARY.json): fixed centering, PC removal and whitening controls hurt despite low effective rank. | A general metric or semantic-information ceiling. | No new centering/metric sweep. A distinct mechanism would need independent support before revisiting geometry; low rank by itself supplies none. | N |
| Q09 Relational text / D | [C16](AS-C16_result.md): 61 TRAIN exact-lexicon pairs, no eligible DEV group. [C22](AS-C22_result.md)/[C23](AS-C23_result.md): omission restoration +0.096339 pp, no affected persistent R1 repair. [C41](AS-C41_result.md): 396 bilingual one-edit TRAIN pairs all strict, zero eligible DEV pairs. | Natural wording differences need not be meaning differences; TRAIN separation is not held-out compositional competence. | Require an existing valid contrast set before a linguistic attribution. No edit-distance relaxation, new labels, native-caption or lexical-support rescue. Existing narrow inventories are exhausted, not all relational language. | N; R; U |
| Q10 Translation collapse / D | [state ledger](../../AUTONOMOUS_RESEARCH_STATE.md): two additional exact-collapse classes; [C22](AS-C22_result.md) finds no new deployed-input collision from truncation. | Semantic collapse beyond equality or whether translation causes persistent errors. | Native/English similarity disagreement is computable, but without validated meaning distinctions it remains a model-dependent descriptor and currently cannot select an open intervention. No native-caption preservation or multilingual-encoder swap as a renamed candidate. | U; C for specified repair |
| Q11 Relevance equivalence / A,I | [state ledger](../../AUTONOMOUS_RESEARCH_STATE.md), [C45](AS-C45_result.md): repeated text/gloss exists, but exact equality is not human semantic equivalence. | Actual alternative relevance and ambiguity among confusers. | Expert relevance labels would resolve a different evidential question, but new annotation/positives are outside current scope. Existing labels can be inventoried, not upgraded into new relevance truth. No PMGR/equivalence-positive reopening. | R; C |
| Q12 Source nuisance / H | [C10 record](AS-C10-NUISANCE.json) is descriptive; [C26/C27](AS-C26_C27_result.md) holds model and gallery size fixed and finds no uniform seen-prefix advantage. | Verified recording identity and causal separation of source from content. | Seek a documented existing provenance field before any source-conditioned causal test. Filename prefixes are not that field; repeating C27 at another weak checkpoint is not a new mechanism. Generic nuisance correction remains closed. | U; R; C |
| Q13 Signer nuisance / H,B | [C10 record](AS-C10-NUISANCE.json): signer slices and text-cosine-matched confusers are inconclusive. | Whether signer information causes errors independently of linguistic content. | An admissible matched intervention and open decision consequence are missing. Generic adversarial removal, projection, selective normalization or confidence fusion collide with closures. Descriptive association alone cannot authorize them. | U; C |
| Q14 Directional gradient conflict / G | [C11/C12 summary](AS-C11-C12-SUMMARY.json), [C18](AS-C18_result.md): all 24 valid-interface and all six measured visual-parameter directional cosines positive. | Text parameters, individual layers, optimizer-coordinate updates and other training stages. | Those gradients are technically measurable, but another slice currently leads only to the closed RPCA/conflict-control path. Do not confuse unmeasured scope with falsification; require a distinct decision consequence before expanding the audit. | N; U; C |
| Q15 Selection/drift / G | [C09 trajectory](AS-C09-TRAJECTORY.json), [cross run](AS-C09-CROSS_run.json): selected epochs 0/−1/146, endpoint hybrids do not beat the best. Per-query ever-correct envelope uses labels. | A transferable source of drift, as opposed to selection noise; intermediate representation states were not saved. | Additional historical intermediate-weight attribution requires unavailable saved states; recreating them is a new training campaign, not read-only recovery. No oracle router, early-stopping, soup or RPCA rescue. Another distinct dynamic mechanism is not yet formulated. | U; R; C |
| Q16 Easy-negative dominance / G | [C18](AS-C18_result.md): duplicate-video pathway projections vary; restricting a video covector does not isolate duplicate query losses. | Actual harmful updates and their generalization consequences. | Loss-level attribution is feasible in code, but no permissible new intervention follows yet; duplicate-aware risk/mining is closed. Do not report pathway concentration as harm or remove/relabel examples. | U; C |
| Q17 Simultaneity/relations / B,C,E | [C44](AS-C44_result.md) corrects readout scope; [annotation result](POST_C44_annotation_result.md) documents exclusion of the published ECCV mouthing sequences from all PH2014T sets, and no populated offsets in 14,711 permitted CSV rows. | The joint-cue failure burden; actual footage overlap was not independently measured. | Direct mouthing-archive join is not a valid next step. Another already-existing compatible annotation resource could matter, but none is established here. No new annotation, pseudo-label substitute, generic stream/synchronization or RCLI revival. This resource route fails; relations are not proven impossible. | R; U; C |
| Q18 Candidate multiplicity / A,F | [state ledger](../../AUTONOMOUS_RESEARCH_STATE.md): repeated captions and paired-gallery asymmetry; [C15](AS-C15_result.md) exposes cohort assumptions. | Semantic multiplicity rather than exact input duplicates; benefit under immutable official positives. | More duplicate slices are computable but would currently feed the closed population/group-risk route. Preserve evaluator/positives; no group-aware method without a materially different mechanism. | U; C |
| Q19 Temporal aliasing / B | [C17](AS-C17_result.md): nearby-view mean loses 0.481696 pp; 475 videos have one effective view. | Broader temporal sampling loss; this test has limited actual view diversity. | No sampler/seed/crop sweep after the fixed screen; sampling consistency and partial OT are closed. A different measured acquisition mechanism would be needed, not a stronger claim from low-diversity negatives. | N; U; C |
| Q20 Score geometry / E,F | [C24/C25](AS-C24_C25_result.md): six hard/mean and three entropy interventions all lose mean R1 and worsen persistent mean ranks. | All score functionals or a harmful-encoder-gradient mechanism; negative coordinate derivatives are not harmful updates. | No temperature, entropy, mask or pooling rescue. The original tokens remain available, but a distinct scorer explanation must precede another run. | N; U |
| Q21 Word augmentation / D,G | [C39](AS-C39_result.md) supplies exact training replay; [C40](AS-C40_result.md): six matched 260-update runs, 0/3 seeds pass all gates; all OFF endpoints below initialization. | Other horizons, not a universal augmentation verdict. | No horizon/strength/schedule/precision/seed rescue. Ordinary removal is not novel. Keep the controlled local negative, rather than treating one favorable seed as survival. | N; C |
| Q22 Scalar-score expressivity / E,J | [C15](AS-C15_result.md): additive dual offsets support every assigned edge non-strictly; ties and evaluation-cohort dependence remain. | Necessity of a nonadditive or cohort-dependent scorer; a general scalar expressivity ceiling is not shown. | First define the permitted inference information and a falsifiable function-class restriction. Pairwise scalar scoring in the abstract is too broad for the current ceiling claim. No duplicate assignment experiment or fixed-bank correction. | U |
| Q23 Unavailable information/ambiguity / I | [C01 summary](AS-C01-R1_summary.json), [C42](AS-C42_result.md), [C45](AS-C45_result.md): persistent errors, inadequate readouts and sparse exact-gloss confusers. | Irreducibility or absence of signed evidence. | Needs a positively adequate probe or already-existing expert evidence. Neither is supplied by repeated errors; new annotation is out of scope. Revisit only after an upstream evidential prerequisite changes. | U; R; D |
| Q24 Cross-dataset transfer / H,J | [research loop](../../Astra_SLRet_Autonomous_Research_Loop.md) §39 requires conditional confirmation; no new surviving signature in this audit. | Generalization of a mechanism that has not yet survived locally. | Apply a surviving frozen signature to locked CSL/How2Sign DEV with proper grouping/provenance. Do not open another dataset to search for a favorable score or silently change benchmark/resources. | D |
| Q25 Recipe fidelity / G | [C28/C29](AS-C28_C29_result.md) identifies upstream-default differences; [C30](AS-C30_result.md) exactly replays control, freeze-only improves 17.490909→18.254545 held mean but still fails adequacy. | Other recipe differences and an adequate clean residual substrate. | No optimizer/table rescue sweep, no ordinary correction-as-novelty. This was not historical upstream reproduction; keep that distinction. Broader adequacy belongs to Q01 rather than declaring recipe equivalence. | N; U; C |
| Q26 Right context / D | [C31](AS-C31_result.md): broad prefix burden 1/92 T2V, 0/87 V2T; 32 fixed-shape suffix interventions leave prefix states exact while EOT changes. | All other right-context limits; EOT/contextual later tokens already carry sentence information. | No bidirectional encoder or prefix correction justified by this failed signature. Do not reinterpret batch-shape numerical differences as semantic context leakage. | N; U |
| Q27 Checkpoint complementarity / F,J | [C32/C33](AS-C32_C33_result.md): uniform score ensemble 77.263969%, a stronger three-model control. [C34](AS-C34_result.md)/[C35](AS-C35_result.md): mismatched 76.107900%, shared aligned 76.011561%; alignment does not repair pairing penalty. | A novel causal source of complementarity or efficient equal-resource improvement. Shared strict confusers already bound convex score mixing for many persistent errors. | Keep ensemble as control. No mixture, anchor, layer, CCA, alignment or parameter-soup rescue. A new operator needs a mechanism beyond known ensembling/merging and fresh collision screening. | U; N; C |
| Q28 Exact input collapse / A,B | [C36](AS-C36_result.md): all 7,615 deployed inputs unique. [C37](AS-C37_result.md): zero within-clip valid-slot repeats among 447,139 slots, FP32/FP16. | Semantic preservation or raw-footage leakage absence; exact equality is a narrow predicate. | Exact-collapse route unsupported. No near-duplicate threshold, precision or sampler rescue; do not turn uniqueness into a semantic-sufficiency claim. | N |
| Q29 Numerical batching / J | [C38](AS-C38_result.md): five single-model V2T tie-rank changes, one R1 loss; no strict preference-set change, all ensemble ranks unchanged. | Numerical robustness under every execution setting. | Preserve official evaluator and three-model control. No batching/precision/sorter method or sweep; measured tie sensitivity does not supply a semantic defect. | N |
| Q30 Larger-budget adequacy / G | [C42](AS-C42_result.md), [successful run](AS-C42-BUDGET-attempt2_run.json), [validation](AS-C42-VALIDATION-attempt2_run.json): 8,800 updates complete, exact final held-score replay, adequacy false. Original interrupted attempt is not a completed replicate. | Adequate clean learning; longer horizon also stretches the schedule, so effect is not duration alone. | No further budget/schedule/optimizer/freeze/seed rescue. Return to Q01's unresolved diagnostic design; do not train on these residuals as though the prerequisite passed. | N; U |
| Q31 Discourse / A,I | [boundary audit](POST_C42_discourse_boundary_audit.md): 7,615 temporal files are sentence-local; numeric predecessor counts do not certify continuity. | Local linguistic/retrieval failure caused by unavailable discourse. | A documented existing original-recording link is required for this route. No inferred-neighbor gold captions, context decoder/protection or source calibration. Missing linkage in inspected files is not global absence. | R; U; C |
| Q32 Gallery-neighbor support / J,E | [C43](AS-C43_result.md): fixed k5 smooth/sharp real/shuffle controls fail 0/3 seeds each; ensemble deltas −6.551/−0.578 pp. | All gallery structure or independent-score limits. | No graph-size, strength, descriptor, pooling, temperature or learned-graph rescue. A different gallery mechanism must first survive Q03/Q22's deployment/algebra boundaries. | N; U |
| Q33 Exact gloss confusers / A,I | [C45](AS-C45_result.md): same-gloss/different-text strict persistent confusers 1/92 T2V, 2/87 V2T, below fixed 10% gate; independent census/scalar checks agree. | Semantic equivalence, nonmanual distinction burden or visual information ceiling. | Stop this exact-gloss lead. No fuzzy-gloss, group-positive, hard-mining or lexical-support expansion; equality of gloss is not permission to alter relevance. | N; U |
| Q34 UPRet reduction / G,E | [real-gradient result](UPRET_real_gradient_result.md): amplitude-controlled max→sum gradient changes 0.174891% visual / 0.220684% text at partial step767, below 1% screen; replay exact. | B512, competitive completed baseline, long-horizon retrieval effect. The frozen transport plan is intentional, not itself a bug. | No long reduction/weight/temperature/optimizer campaign from this lead. A new causal effect would require more than paper-code mismatch; ordinary repair cannot establish novelty. B32 partial-state limitation is explicit, not a universal rejection. | N; U; C |
| Q35 CMCM covariance / B,G | [source result](CMCM_source_result.md), [run](CMCM-COVARIANCE_run.json), [validation](CMCM-COVARIANCE_validation.json): trace/epsilon backward mismatch verified on valid PSD inputs; isolated consistency control restores forward derivative. | Exposure in actual trained activations, retrieval harm and matched native-pipeline effect. | First establish an available compatible trained pipeline/activation provenance, if pursuing engineering fidelity. Do not substitute CiCo activation statistics or build a replacement pipeline as evidence. No epsilon sweep or generic causal-gating proposal; repair alone is not a method. | R; U; C |
| Q36 CiCo pseudo-clips / A,B | [result](CICO_pseudoclip_result.md), [run](CICO-PSEUDOCLIP_run.json): known grouping behavior; new local TRAIN census has any-gap crop exposure 0.615417% versus locked 10% gate; exact replay/frame/interval/sampler checks. | Semantic label noise, retrieval harm, or exposure in the stronger H2S-transfer-aware resource. | No NMS/crop/teacher/fusion repair campaign; local P14T index cannot stand in for strong H2S resources. OCEM remains closed. | N; U; C |
| Q37 Subword/fingerspelling / D,A | [resource audit](SUBWORD_resource_and_collision_audit.md): FS convention exists, compatible human-label join unverified; CiCo inverse byte/symbol maps invalidate a source-only spelling-erasure argument. Prior detection/character models recorded with limited read scopes. | Actual PH/How2Sign failure prevalence and whether embeddings expose spelling distinctions. | Establish a permitted existing-label join before attribution. No character head, tokenizer replacement, teacher/crop/extra stream from tokenization alone. No new annotation or claim that a closed issue means labels released. | R; U; C |

## What the audit changes

1. **No global exhaustion declaration.** The ledger is now explicit, but U entries
   are genuine missing evidence/formulation. A table containing them is not a
   certificate that every high-value intervention is unavailable.
2. **No low-information automatic continuation.** Measurable is not synonymous
   with useful: more gradient slices, similarity disagreements or duplicate
   counts need an admissible decision consequence before more code/compute.
3. **No resource conflation.** Strong PH-fitted CiCo resources, clean generic
   initialization, partial local UPRet and local P14T pseudo-labels are distinct
   regimes. Neither a negative nor a positive transfers between them by name.
4. **No stale task launch.** Historical result endings describe later experiments
   as pending; subsequent completed records take precedence. In particular,
   C27, C30, C38, C40, C42, C43 and C45 are not to be restarted from old prose.
5. **No repeat of the inadequate-residual route.** C42 closes its fixed budget
   lead, not Q01. Residual abundance and excellent fit recall do not satisfy the
   missing held adequacy requirement.

## Selected bounded follow-up: formulation before another run

Completed after this audit: [Q02/Q22 function-class boundary](Q02_Q22_function_class_boundary.md).
It rejects shared-score and dimension-count arguments as sufficient explanations
using explicit existence counterexamples, and checks the actual scorer scope.
It does not find a new launch-ready operator or close learnability. The selection
rationale below is retained as history, not an instruction to repeat the work.

Prioritize **Q02/Q22's function-class boundary**, not a Q38 counter increment.
This is analytical work with existing source and evidence, not authorization for
another mixture/graph/pooling run. Specify exactly:

- Which input information a deployed query may consume (one query, candidate
  gallery, no other evaluation queries and no evaluation labels).
- Which restriction of the current CiCo score is alleged to fail, separating a
  restriction of its implemented function from unrestricted scalar pair scoring.
- An observable consequence not already tested by C03, C07/C15, C24/C25 or C43.
- Why that consequence could distinguish an open mechanism from closed column
  correction, group risk, local evidence reranking, relation alignment or generic
  graph filtering.

Use algebra to reject underidentified or closed formulations before code.
If a concrete new restriction survives, register its smallest diagnostic and
targeted primary-literature screen. If it does not, record the failed formulation
and move to another existing U row. **Do not call that single failure a global
barrier.** Q01 and Q17 stay open with their explicit adequacy/resource gaps.

This next action is justified by an unresolved high-priority claim, not evidence
that a new expressivity method already exists. No candidate, training run or
new DEV query selection is authorized merely by this handoff.

## Inline adversarial checkpoints

These are sequential perspectives of the same assistant, not independent reviews.

1. **Scoping — revised:** Counting failed probes would overstate coverage.
   Correction: preserve tested specifications separately from broader questions,
   retain all 37 IDs, and disallow a global-barrier inference from their count.
2. **Synthesis — revised:** Calling unformulated interventions infeasible would
   turn lack of an idea into evidence. Correction: explicit U verdicts; identify
   concrete technically feasible but currently decision-poor measurements, and
   separate them from genuine resource/authority prerequisites.
3. **Conclusion — bounded pass:** Strongest counterargument: this table still
   does not exhaust all permissible operators, establish independent linguistic
   validity, or audit every raw artifact. Accepted. Therefore it supports the
   next analytical work, not GO, a terminal barrier, a universal negative claim,
   or fresh confirmatory significance from heavily reused DEV.

No new model execution, dataset/TEST access, checkpoint selection, asset download,
upstream edit or changed positive occurred in this audit. Earlier incidental
TEST-related issue-comment exposure remains disclosed and quarantined in the
subword report; this audit does not erase that history. Research goal stays active.
