# Adversarial review of inherited C27

2026-09-22. Four sequential analytical perspectives from one assistant using
the academic-research-suite skepticism prompts. No independent agents, external
model, human reviewer or sign-language expert review is claimed.

## A — novelty

Major: sign-level supervision jointly with sentence retrieval already exists
in CSLR²; CTC with contrastive sign learning exists in CVT-SLR. Applying an
ordinary auxiliary to SEDS may be a useful baseline, not a paper contribution.
Response: no novelty claim; require a measured failure and an experiment that
distinguishes ordered supervision from generic gloss content. Task transfer
alone does not answer the criticism.

## B — experimental validity

Major: CTC versus no-gloss continuation confounds sequence modeling and extra
expert annotation. Head attachment changes initialization RNG and parameter
groups; unequal DEV evaluations create selector advantage. Previously deleted
feature artifacts cannot be treated as currently reproducible.
Response: parameter/resource-matched order-free auxiliary, matching RNG and
batch/video IDs, equal update/selection exposure, fresh hashes after the user's
recovery handoff. Preserve original versus adapted feature ancestry.

Major: TRAIN-internal head holdouts still inherit a release checkpoint trained
on those examples. Response: label them exploratory representation probes;
independent confirmation requires an initialization/training history that
excludes the held groups or an appropriate untouched evaluation split.

## C — sign-language assumptions

Major: a gloss sequence records only part of signing. CTC's ordered emissions
do not model simultaneous nonmanual information, and spoken translation order
cannot be imposed as sign order. Distinct exact translations do not prove two
gloss permutations are linguistically different meanings.
Response: use only released ordered gloss labels, keep score relevance fixed,
do not manufacture negative pairs or claim expert linguistic validation. The
140-row exact-multiset slice is a descriptive diagnostic, not a validated
order benchmark and not evidence for the cause of general retrieval errors.

## D — retrieval

Major: lower gloss loss/WER can coexist with unchanged or worse retrieval;
CSLR²'s CE auxiliary illustrates this possibility in another resource setting.
The PH best observed GCN result is already above release initialization.
Response: deploy the unchanged scoring path, report both directions and full
R@1/5/10, compare at fixed steps and identical selection opportunities, retain
the strongest practical incumbent. No TEST tuning or cherry-picked direction.

## Stage verdict

Cycle 2 evidence narrows the question: zero persistent errors have a strictly
outranking same-gloss-multiset/different-order confuser. The predeclared exact
order-confusion gate fails. This does not refute broader lexical/sequence
supervision. Both pose and RGB miss most persistent errors, but scores alone
cannot distinguish missing visual information, shared training bias, scoring
limitations, or incorrect singleton relevance. No semantic labels were added.
The 38-question map is one-assistant analysis, not independent expert review.

**REVISE diagnostic design; candidate remains OPEN.** There is no demonstrated
C27 retrieval effect, no completed matched pilot, and no order-specific
bottleneck. These gaps block promotion, not read-only investigation or the
whole research program. The academic skill's evidence-first review led to
the additional same-gloss control and explicit source-reading limits.

Strongest counterargument: “Any gain would be ordinary extra annotation and
GCN adaptation; no evidence identifies order as the missing signal.” That
argument remains live until a separating experiment succeeds. Reject a broad
novelty story now; do not reject all possible value of gloss supervision.

## Cycle4 — four-perspective review of the six-route screen

Scope: [candidate definitions and controls](09_CANDIDATE_SCREEN.md). These are
sequential perspectives from the same assistant, not independent agents, external
models, human reviewers or qualified sign-language judgments. The academic skill's
skeptical review challenges admission; it does not generate endorsements.

| Route | A — novelty | B — experimental validity | C — sign-language assumptions | D — retrieval |
|---|---|---|---|---|
| P1 ordered gloss | CVT-SLR/CSLR² defeat a generic sign-supervision story | Ordered versus no-label comparison confounds extra annotation; same-label content control required | Gloss order omits simultaneous meaning; permutations are not validated semantic negatives | Lower gloss loss may not repair ranks; measured exact-order burden is zero |
| P2 pair cross-attention | Thinking Fast and Slow precedes cross-attention reranking; local verification closures remain | More pair compute/capacity and shortlist selection can explain gains | Interaction cannot restore an absent visual cue; translation tokens are not gloss alignments | Near-misses do not prove scorer incapacity; full-gallery and both-direction effects required |
| P3 relation binding | Generic relation composition collides with RCLI; SEDS already encodes pose relations | No valid relation-label join; positional bins or invented anatomy invalidate controls | Simultaneity and nonmanual grammar need real evidence; swapped components are not grammatical contrasts | Relation prediction can improve without correcting retrieval; connect effect to actual confusers |
| P4 decoder score | Normalized conditional caption likelihood is established | Match donor supervision and decoder compute; language-only/shuffled-visual controls required | Fluent spoken-language likelihood may ignore signed distinctions | Text-prior subtraction cannot alter T2V by itself; perplexity/BLEU do not establish rank gains |
| P5 raw acquisition | Already proposal7 C; mmSampler occupies generic acquisition policy | Charge every candidate's decoding/encoding; compare equal-budget fixed sampling | Query-led crops may discard simultaneous cues; missing cue remains hypothetical | More observations need not resolve relevance ambiguity; deferred work does not authorize this pipeline |
| P6 candidate-set comparison | TokenBinder supplies one-to-many comparison; local graph/reranker closures remain | No other evaluation queries/labels; shortlist and set-size controls required | Rival sets are not independent evidence of sign semantics | Scalar pair scores are not inherently incapable; set-order/distractor stability required |

### Responses and admission decisions

- **P1:** retain OPEN, not SUPPORTED-FOR-PILOT. The burden result rejects its narrow
  justification; broader same-label controls are specified but missing bottleneck
  evidence remains. A recovered cache is not that evidence.
- **P2:** reject the generic transformer route. An unmeasured distinction and a
  hypothetical control do not satisfy local-separation gates.
- **P3:** reject the generic relation head. Q17 stays unresolved; an unavailable
  annotation join is not impossibility of relational reasoning.
- **P4:** reject generic likelihood scoring because of the direct prior collision
  and missing measured failure. Do not train a decoder merely to test a known
  mechanism under a different task name.
- **P5:** reject the inherited closed proposal without rerunning the policy or
  relabeling a larger visual encoder as a new information principle.
- **P6:** reject generic candidate-aware comparison; no new statistic or processing
  restriction survives. Fixed-score-mixture limits do not justify a set model.

Strongest shared criticism: “These are known interventions placed after an error
census, not mechanisms derived from an identified failure.” The screen accepts
that criticism for the current formulations. This bounded rejection result is
not a global NO-GO theorem or a completed methods proposal. Files10–12 cannot be
substantively completed without a supported survivor. Further research must add
new discriminating evidence or a genuinely different causal mechanism, not another
synonym for these six routes.
