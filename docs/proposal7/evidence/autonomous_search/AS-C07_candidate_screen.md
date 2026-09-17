# AS-C07: three materially different follow-up candidates, not selected methods

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: empirical candidate screening
- Origin Date: 2026-09-15
- Verification Status: ANALYZED; novelty coverage OPEN
- Version Label: cohort-candidates-v1

Observed finding: label-free maximum-score one-to-one assignment over the complete
PH query cohort raises mean bidirectional R1 by 7.803468pp across three historical
baselines. With half-query cohorts, gain falls to 2.087347pp. This is a transductive,
capacity-assumption-dependent diagnostic, not a deployable method or linguistic
disambiguation claim. No dev labels enter the assignment solver.

The following are distinct mechanisms derived from that finding. None is named
as a new research method, none passes the method gate, and no proposal8 is created.

## Candidate A: fixed train-reference competition at independent-query inference

- Observed failure: independent maxima can differ from high-scoring coherent
  cohort assignments; full-cohort resources are not always available.
- Evidence: AS-C07 full/half/single-query comparison.
- Hypothesis: a fixed train query bank can supply missing competitor structure.
- Mechanism: score each candidate by the best assignment with the current query
  forced to that candidate and the remaining slots assigned to the fixed bank.
- Training: optional train-only reference selection; no dev fitting.
- Inference: forced-edge assignment over the bank and current candidate gallery.
- Baseline expressivity: alleged nonlocal interaction must be demonstrated, not assumed.
- Closest SLRet prior: OPEN targeted search; Proposal7 candidate-prior correction is a critical local collision.
- Closest adjacent prior: query-bank normalization / hubness correction, exact primary-source check pending.
- Collision: **fatal algebraic reduction to candidate-only additive correction**
  when bank/gallery are fixed. For candidate j, F(q,j)=s(q,j)+M(bank,V\{j}).
  The second term does not depend on q. This reopens the closed prior-column family.
- Cheap falsification: exhaustive 4x4 forced assignments versus the additive formula.
- Failure criterion: exact additive reduction; no need to spend a training run.
- Compute: multiple assignment solves unless candidate potentials are cached;
  caching makes the closed additive structure even more explicit.
- Status: **REJECT as a novel independent-query mechanism**. Do not rename it.

## Candidate B: out-of-fold cohort teacher distilled into an independent scorer

- Observed failure: useful joint assignments cannot directly serve isolated queries.
- Evidence: cohort-size dependence; AS-C04 also shows almost no distinguishable
  in-sample baseline train failures to teach a correction.
- Hypothesis: genuinely out-of-sample train cohorts provide transferable correction targets.
- Mechanism: a cohort teacher on source-held-out TRAIN folds supplies soft ranking
  targets; a query-independent student learns them under fixed retrieval evaluation.
- Training: teacher folds and student fitting use train only; dev selects the
  prespecified pilot. Existing PH-fitted release initialization is NOT clean
  out-of-fold provenance merely because later fine-tuning excludes a fold.
- Inference: one query, fixed candidate gallery, no evaluation-query cohort.
- Baseline expressivity: the student may use the same score family; the claim
  would concern supervision, not inability of a scalar score to express a ranking.
- Closest SLRet prior: OPEN targeted distillation/ranking search.
- Closest adjacent prior: transductive-to-inductive distillation / listwise ranking
  distillation; no absence-of-prior-art claim.
- Collision: ordinary teacher replacement is not novelty. Must distinguish the
  causal mechanism from ELSC/DIVE/RPCA and other closed teacher/supervision families.
- Cheap falsification: first verify that a provenance-clean held-out train teacher
  generates corrected different-input pairs, then compare an identical-capacity
  student against ordinary and shuffled teacher targets.
- Failure criterion: no transferable corrections, no independent-query dev gain,
  matched controls explain gain, or prior-art equivalence.
- Compute: multiple baseline fits may be required. CLIP/I3D assets exist; clean
  initialization and fold provenance must be audited before authorizing the pilot.
- Status: **OPEN collision/provenance screening, not GO**. Do not train before screening.

## Candidate C: structured permutation training with ordinary pairwise inference

- Observed failure: independent ranking does not impose cohort capacity.
- Evidence: AS-C07 assignment advantage.
- Hypothesis: a permutation-structured loss could teach global compatibility
  without requiring joint inference.
- Mechanism: normalize scores over assignments or use an assignment margin.
- Training: paired train labels under batch assignment constraints; unchanged
  features, update budget and parameter controls.
- Inference: ordinary independent candidate ranking.
- Baseline expressivity: different objective, not greater pair-score capacity.
- Closest SLRet prior: OPEN; ordinary OT variants and population-risk remixes are constrained.
- Closest adjacent prior: structured prediction / assignment contrastive objectives,
  exact collision check pending if this survives algebra.
- Collision: could be another OT/contrastive-loss variant; no novelty claim.
- Cheap falsification: add arbitrary row/column potentials to a score matrix.
  Every complete assignment receives the same additive total, so its structured
  loss is unchanged while independent T2V/V2T rankings may change completely.
- Failure criterion: **standalone permutation loss cannot identify the score
  potentials required for independent ranking**. A calibrated extra objective
  would be a new candidate requiring separate evidence, not an automatic rescue.
- Compute: exact partition is factorial; approximations add separate attribution
  and collision questions, but the invariance holds before choosing a solver.
- Status: **REJECT the standalone formulation as underidentified**.

## Five inline adversarial perspectives (single-model, not independent agents)

- Novelty: A reduces to a closed correction family; C has classical structured-loss
  invariance; B needs targeted primary-source collision checks before any novelty claim.
- Retrieval: full-cohort assignment gain does not prove independent-query gain;
  B must demonstrate transfer under the ordinary full-gallery protocol.
- Sign-language validity: one-to-one capacity can force arbitrary distinctions
  among repeated/underspecified captions. No signed-language equivalence has been verified.
- Reproducibility: published PH-fitted initialization contaminates naive held-out
  teacher folds; exact clean asset lineage is a prerequisite for B.
- Causal attribution: compare B with same student capacity, computation, ordinary
  teacher targets, shuffled targets and a matched closest-prior mechanism.

No candidate currently survives. The original next step was targeted collision
and clean initialization audit for B. Update2026-09-15: `AS-C07_collision_screen.md`
closes the present B specification after the OTTER mechanism collision. No fold
training was launched. This does not prove all structured retrieval is exhausted.
