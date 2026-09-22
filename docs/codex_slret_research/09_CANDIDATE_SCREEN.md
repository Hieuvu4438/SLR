# Six-route mechanism admission screen

2026-09-22; Cycle4; base HEAD `53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`.
Decision: **no candidate admitted for a pilot**. P1 is OPEN; P2–P6 are REJECTED
in the forms specified below. These are six distinct intervention routes, not
six claimed discoveries. Inherited or equivalent routes are included explicitly
to prevent their reintroduction under new names. None is being implemented.

Prerequisites: [history](01_CODEBASE_AUDIT.md),
[baseline mechanisms](02_THIRD_PARTY_AUDIT.md),
[binding closures](03_NEGATIVE_RESULTS_BLACKLIST.md),
[fair comparison](04_BASELINE_AND_SOTA_MATRIX.md),
[failure map](05_FAILURE_LAYER_MAP.md),
[38 questions](06_OPEN_RESEARCH_SPACE.md),
[falsifiable hypotheses](07_HYPOTHESES_AND_FALSIFICATION.md), and
[prior collisions](08_PRIOR_ART_COLLISIONS.md). The screen follows these
audits; it does not turn a missing explanation into evidence for an architecture.

## Shared evidence and admission rules

[M] Three selected GCN DEV runs share113 T2V and98 V2T errors;88/74 fail in
both pose and RGB in every seed. These are error counts, not measured causal
bottlenecks. Most persistent errors are already in ranks2–10. Zero persistent
errors have a strictly outranking exact-gloss-multiset/different-order confuser.
The narrow exact-order burden gate therefore failed. Source:
[retained-score experiment](evidence/GCN_RESIDUAL_RESULT.md).

All proposed tests below are **PROPOSED — NOT MEASURED**, not launch orders.
The current guide prohibits implementing a candidate until SUPPORTED-FOR-PILOT.
A separating control is necessary but not sufficient: a mechanism must also
address an evidenced bottleneck and be materially distinct from local closures.
The broad questions Q01/Q17/Q22/Q23 remain open when a particular route fails.

Cycle12 [scope correction](evidence/ADMISSION_SCOPE_AUDIT.md): validated natural
contrasts below are requirements of the particular semantic explanations, not
a universal seventh pilot gate. Computational/statistical mechanisms may be
tested against existing official pairs, with conclusions restricted to that
protocol. Rechecking the six routes under the original gates changes none of
their statuses. No human or AI semantic annotation is requested.

For any future admitted efficacy pilot, predeclare a matched same-resource,
same-exposure control and equal DEV selection opportunities. A proposed
screening threshold is mean bidirectional R1 improvement ≥0.5 percentage points,
neither direction declining >0.5 points, beyond each mechanism-specific control.
This is a decision threshold, not a significance claim; three selected-checkpoint
results do not supply a valid null distribution for a new comparison. TEST stays
locked. Synthetic linguistic contrasts cannot establish sign-video relevance.

## Six-baseline distinction matrix

Each cell answers why the **specified intervention** is not already that
baseline, or admits an overlap. Absence in six implementations is not novelty.
CiCo/SEDS already use contextual token interaction: neither is treated as a
simple pooled dot-product straw baseline. CMCM scope is bounded by its component
release and primary metadata, not a certified integrated implementation.

| Route | CiCo | SEDS | UPRet | C²RL | SAN | CMCM |
|---|---|---|---|---|---|---|
| P1 ordered gloss auxiliary | No gloss CTC in audited retrieval path | Same backbone/scorer; adds supervised CTC, absent from native forward | Gaussian/transport auxiliary is not ordered gloss CTC | Translation/context pretraining already overlaps broad auxiliary story; exact gloss CTC differs | Caption-negative loss, not gloss transcription | Published components do not establish ordered gloss CTC |
| P2 pair cross-encoder | Adds joint pair-conditioned token updates before scoring, not merely existing token similarities | Adds text-conditioned visual interaction; pose/RGB fusion alone is not this | Learned token weights/transport are not the proposed pair transformer | Independently encoded downstream CLCL differs from pair transformer | Hard-caption CE does not supply this joint pair module | Covariance/Gaussian modules do not establish this pair transformer |
| P3 articulator relation binding | No explicit relation-labeled articulator input | GCN/fusion already model relations implicitly; proposed extra binding is not automatically distinct | Distributional alignment does not validate articulator relations | Contextual encoders may retain them; no proven missing relation interface | Visual confusion mining is not independently labeled simultaneous binding | Covariance could retain related interactions; cannot claim relation-free baseline |
| P4 decoder likelihood at inference | Token similarity, not candidate sentence likelihood | Same distinction; auxiliary decoder alone would not suffice | Gaussian embedding score is not autoregressive sentence likelihood | Decoder likelihood is already a training objective; direct deployed ranking is the proposed difference | Negative-caption discriminative score, not generative likelihood | Published components do not establish this sentence decoder scorer |
| P5 adaptive raw acquisition | Cached input processing does not acquire new pixels | Existing pose/RGB path is not an adaptive raw-reading policy | Token weights reweight existing observations, not acquire new ones | A stronger visual encoder is not itself adaptive acquisition | Hard-caption selection is not raw visual acquisition | Augmentation adjustment is not evidence of this acquisition policy |
| P6 candidate-set verification | Pair token scores do not jointly condition on rival video features | Streams fuse within videos, not across candidate videos | Training-batch transport is not deployed candidate-set comparison | Independent downstream encoders do not perform it | Training hard negatives do not make deployed scores set-conditioned | Published pair alignment/covariance does not establish this operation |

## P1 — Ordered gloss supervision of the existing pose branch

- **Status / provenance:** OPEN; inherited dataset-first C27, not a new acronym.
- **Measured bottleneck:** annotation availability and persistent errors measured;
  missing ordered evidence not established. Exact-order-confuser rationale failed.
- **Hypothesis:** ordered human gloss labels teach rank-relevant structure beyond
  the same labels used without order.
- **Core mechanism / information in and out:** TRAIN gloss sequences and valid
  chronological pose tokens enter CTC; auxiliary gradients leave. The deployed
  pose/RGB/fusion score remains unchanged; no gloss labels enter evaluation.
- **Changed parameters / rank prediction:** classifier and explicitly declared
  upstream pose scope; expected improvement must appear in actual full-gallery
  rankings, not merely lower gloss loss. Trainer scope is still unspecified.
- **Why not a closed local proposal:** human ordered labels differ from ELSC/SSSC
  teacher-selected support and RPCA sentence-decoder protected updates. Generic
  auxiliary regularization and temporal-order stories still collide. A different
  label source alone is not a sufficient distinction.
- **Closest adjacent prior / novelty risk:** CVT-SLR and CSLR²; high risk from
  existing CTC/contrastive learning and sign-plus-sentence supervision. See08.
- **Implementation complexity:** moderate; existing head/tests are not a trainer.
  Needs exact IDs, valid masks, repeat-aware CTC feasibility and matched controls.
- **Inference complexity:** unchanged if head removed; training adds gloss logits
  O(BFG) and sequence alignment. Existing classifier has557,118 parameters.
- **Cheap falsification / separating experiment:** the completed strict-confuser
  test already rejects the narrow rationale. Broader admission requires a new
  demonstrated label-relevant distinction; any subsequent pilot must compare
  native continuation, CTC and same-label order-free supervision at matched cost.
- **Expected diagnostic signature:** ordered objective uniquely improves the
  demonstrated distinction and deployed ranks beyond the content-only control.
- **Kill criterion:** equal content-only gain kills the order-specific contribution;
  neither beating continuation closes the recipe. Recovery success alone cannot
  promote it. No new diagnostic establishing broader admission is specified yet.

## P2 — Full pair-conditioned cross-attention verification

- **Status / provenance:** REJECTED as proposed; not a reopening of residual heads.
- **Measured bottleneck:** stable near-misses are measured; failure of a particular
  pair-independent representation is not. CiCo already has nonlinear token scoring.
- **Hypothesis:** joint visual/text contextualization recovers a natural distinction
  that separately contextualized tokens and the incumbent reduction cannot use.
- **Core mechanism / information in and out:** all valid frozen visual/text tokens
  enter a small joint transformer; one pair compatibility score leaves. No local
  teacher support or new raw observations are introduced.
- **Changed parameters / rank prediction:** joint transformer and score head only
  in the minimal version; an incorrect high-ranked candidate should fall below
  the paired candidate under the official relevance rule.
- **Why not a closed local proposal:** using full pair interaction differs in
  syntax from pointwise residuals, but the causal path remains learned same-feature
  verification/reranking close to PLEL, DIVE and failed frozen readouts. No measured
  distinguishing cue or sufficient scientific separation has been demonstrated.
- **Closest adjacent prior / novelty risk:** Thinking Fast and Slow already uses
  cross-attention retrieval and fast/slow reranking. High; transplantation is not
  a new mechanism. Source and reading limits in08.
- **Implementation complexity:** moderate to high; pair expansion, masking, negatives,
  gradient parity and fair compute control are substantive changes.
- **Inference complexity:** for full joint attention approximately
  O(Nv Nt (F+W)^2 D) per layer, excluding encoders/projections. A shortlist reduces
  pair count but changes deployment and imposes a shortlist recall ceiling.
- **Cheap falsification / separating experiment:** before fitting, exhibit a
  naturally occurring, validated distinction and an actual incumbent restriction;
  compare a same-input joint readout with equal-capacity separate readouts on
  untouched groups. Generic artificial IDs/teacher-copy controls are already closed.
- **Expected diagnostic signature:** improvement on that distinction transfers to
  official ranking beyond capacity and shortlist controls, in both directions.
- **Kill criterion:** no new cue/restriction or no distinction from local closures
  rejects admission now. A future negative probe would reject that readout, not
  prove that all frozen representations lack information.

## P3 — Simultaneous articulator-relation binding

- **Status / provenance:** REJECTED in this generic relation-head form; Q17 stays open.
- **Measured bottleneck:** no validated simultaneous-relation contrast is joined to
  the persistent error set. Old spatial bins are not isolated articulators.
- **Hypothesis:** a specific hand/body/nonmanual relation distinguishes candidates
  with similar component content, and a known processing interface loses it.
- **Core mechanism / information in and out:** synchronized anatomically identified
  component trajectories plus timestamps enter a relation-sensitive operator;
  bound relation features enter retrieval. Component identity must be real, not
  inferred from pooled array positions.
- **Changed parameters / rank prediction:** relation encoder and retrieval interface;
  ranks change only when the independently validated relation is relevant.
- **Why not a closed local proposal:** this is the same broad relation-composition
  pathway as RCLI/global-composition proposals without a new measured interface
  defect. SEDS GCN and contextual features may already retain the relation.
  Calling the tokens articulators does not resolve either objection.
- **Closest adjacent prior / novelty risk:** SEDS relational pose processing,
  SignSeek articulator masking and CMCM covariance are overlap checks, not proofs
  of an identical operator. Exact relation-specific prior search remains OPEN;
  high local collision risk already rejects this generic candidate.
- **Implementation complexity:** high until a valid existing cue/label join exists;
  synchronization, coordinate systems and occlusion handling are not free.
- **Inference complexity:** pairwise A-articulator binding costs at least
  O(F A^2 D) for a dense pair operator, plus the unchanged retrieval path;
  missing articulator extraction would add unmeasured cost.
- **Cheap falsification / separating experiment:** independently validated natural
  contrasts, then component-only versus relation-sensitive readout with matched
  resources and correspondence-null control. Such contrasts are currently absent;
  synthetic sign permutations are not ground truth. No new human study is authorized.
- **Expected diagnostic signature:** relation-specific rank repairs survive the
  component-only and nuisance controls, not just a generic overall gain.
- **Kill criterion:** no reliable relation join, relation already accessible to
  the control, or effects explained by component content/extra capacity. Current
  missing join and conceptual overlap prevent implementation.

## P4 — Sentence-decoder likelihood as the deployed retrieval score

- **Status / provenance:** REJECTED as a generic likelihood-retrieval contribution.
- **Measured bottleneck:** no evidence attributes GCN errors to discriminative
  rather than conditional-generative scoring. C26-A's pooled-head failure is not
  evidence for or against this different inference score.
- **Hypothesis:** a visually conditioned language model can distinguish candidate
  sentences from full conditional evidence that the current score does not use.
- **Core mechanism / information in and out:** visual sequence and teacher-forced
  candidate text enter a decoder; score is log p(text|video) − log p(text).
  The prior must use permitted TRAIN data, not evaluation-cohort labels.
- **Changed parameters / rank prediction:** decoder/visual conditioning if trained;
  deployed score replaces similarity rather than being a training-only auxiliary.
  A text-only prior is constant across videos for one text query, so its subtraction
  cannot change T2V ranking; it can change V2T ranking.
- **Why not a closed local proposal:** inference likelihood differs from RPCA's
  protected auxiliary training and proposal7's visual-density candidate. That
  distinction does not create a new generative retrieval mechanism; nuisance/prior
  correction also touches closed calibration routes.
- **Closest adjacent prior / novelty risk:** m-RNN §6 explicitly uses conditional
  caption likelihood normalized by a caption prior for retrieval. C²RL covers joint
  generation/retrieval representation training. Very high; direct prior collision.
- **Implementation complexity:** high without a verified compatible decoder and
  resource contract; no SLT proxy-training or checkpoint search campaign is admitted.
- **Inference complexity:** candidate-conditioned decoding for Nv×Nt pairs, roughly
  O(Nv Nt (W^2+WF) D) per transformer layer before vocabulary projection. Candidate
  vocabulary logits add O(Nv Nt W D V); encoder caching does not remove decoding.
- **Cheap falsification / separating experiment:** existing prior formula and the
  directional invariance already defeat the generic novelty/normalization claims.
  A genuinely different future claim would need fixed visual-conditioned versus
  shuffled-visual/language-only/equal-compute controls; low perplexity is insufficient.
- **Expected diagnostic signature:** correct visual evidence changes pair ordering
  beyond the language prior and transfers to both official retrieval directions.
- **Kill criterion:** prior-equivalent scoring without a new demonstrated failure
  mechanism rejects now. Language-only or capacity controls explaining a putative
  gain would reject a later visual-evidence claim, not all decoder usefulness.

## P5 — Budgeted acquisition of additional raw visual observations

- **Status / provenance:** REJECTED; explicitly inherited proposal7 candidate C.
- **Measured bottleneck:** shared branch errors do not demonstrate which raw cue
  is missing or whether it can be recovered at a permitted cost.
- **Hypothesis:** a bounded observation policy captures an identified omitted cue
  more efficiently than equal-budget fixed raw sampling.
- **Core mechanism / information in and out:** a query/coarse state selects new
  raw frames or resolution; newly encoded observations update the score. Merely
  weighting existing tokens would instead be the already-closed support mechanism.
- **Changed parameters / rank prediction:** acquisition policy and observation
  encoder; ranks change when the acquired cue resolves a natural ambiguous pair.
- **Why not a closed local proposal:** it does not differ from proposal7 C in a
  demonstrated causal way. Replacing frame choice with crop choice is no exemption.
  Deferred UniFormerV2 work is not authorization to start a new extraction campaign.
- **Closest adjacent prior / novelty risk:** mmSampler's learned acquisition policy,
  documented in the earlier primary-source screen; high external and local risk.
  No new full-paper reading or claim of equation identity is made here.
- **Implementation complexity:** high; raw decoding, policy/encoder integration,
  cost accounting and full-gallery execution would all need controls.
- **Inference complexity:** query-dependent acquisition may repeat decoding/encoding
  per candidate pair. Report total raw reads and encoder calls, not policy FLOPs alone.
- **Cheap falsification / separating experiment:** first demonstrate an existing,
  permitted fixed high-information observation recovers the specified cue against
  a matched low-information view. This is an information diagnostic, not a policy
  contribution; the generic crossed diagnostic has already been proposed historically.
- **Expected diagnostic signature:** acquisition adds a validated distinction and
  improves ranks beyond equal-budget fixed acquisition, not just more raw compute.
- **Kill criterion:** no new observation/cue, no advantage over equal-budget fixed
  acquisition, or the unchanged historical pathway. The latter rejects this form now.

## P6 — Within-query candidate-set competitive verification

- **Status / provenance:** REJECTED as generic candidate-aware reranking.
- **Measured bottleneck:** common strict confusers prevent simple nonnegative seed
  mixtures from repairing many errors. This is not evidence that all independent
  pair scores fail; that broader mathematical argument was already refuted.
- **Hypothesis:** comparing rival candidate features for one query exposes a
  validated distinction that the incumbent pair score cannot express/use.
- **Core mechanism / information in and out:** one query and a label-free candidate
  set enter permutation-equivariant attention; set-conditioned compatibility scores
  leave. Other evaluation queries, their labels and one-to-one assignments are excluded.
- **Changed parameters / rank prediction:** set comparator; expected pair ordering
  improves under natural rivals while remaining stable to irrelevant candidates.
- **Why not a closed local proposal:** true cross-candidate features differ from
  a fixed-bank offset or cohort assignment, but generic gallery/reranker/residual
  pathways remain closed. No demonstrated new rank-relevant statistic separates it.
- **Closest adjacent prior / novelty risk:** TokenBinder directly compares candidate
  videos through a one-to-many coarse/fine architecture. High; changing the task to
  signing alone does not supply a mechanism contribution.
- **Implementation complexity:** moderate to high; set-size batching, permutation
  checks, distractor controls and direction-specific galleries must be explicit.
- **Inference complexity:** a dense comparator over K compact candidate tokens is
  O(K^2 D) per query/layer beyond encoding; full video-token interaction is higher.
  K shortlisting changes recall ceilings and must preserve the official gallery.
- **Cheap falsification / separating experiment:** derive a real restriction and
  validated contrast before fitting; then test candidate-independent equal-capacity
  control, set permutation and irrelevant-distractor addition on held examples.
  Existing synthetic assignment/oracle and graph controls are not repeated.
- **Expected diagnostic signature:** corrected relevant ordering survives set order
  and distractor changes and improves official bidirectional retrieval.
- **Kill criterion:** reduction to offsets, dependence on other queries/labels,
  no new measurable cue, or generic TokenBinder-style transplantation. Current
  formulation fails novelty and local-separation gates before training.

## Decision and what would change it

Four sequential skeptical perspectives are recorded in
[13_REVIEWER_ATTACK.md](13_REVIEWER_ATTACK.md); they are one-assistant analysis,
not independent reviewers or linguistic validation. None of these candidates
answers all material criticisms with current evidence.

This completes one bounded six-route admission screen, **not** a successful
method-selection stage or a proof that six genuinely novel ideas were found.
Do not write a fictional primary proposal to populate files10–12. Do not launch
C27 or a rejected route, refresh baseline parity, or repeat generic positive-control
and scalar-score arguments as the next experiment.

The next admissible research task must supply something missing here: a
distinct falsifiable mechanism, supported by evidence appropriate to its claim,
and a separating test not already covered by the blacklist. This may concern a
computational/statistical bottleneck measured using official pairs; a validated
natural semantic distinction is not universally required. Q01/Q17/Q22 name
unanswered questions; they are not implementations. Existing relevance
judgments must be checked for task compatibility, not silently converted from
gloss/text disagreement into video truth. A successful recovery handoff after
`xong` can verify masks/IDs but cannot supply this scientific evidence by itself.

No cheap empirical test is presently admitted. This is an honest stage result,
not a global exhaustion claim, goal completion, or permission to loop through the
same rejected families. Primary method, implementation plan and confirmation
remain outstanding.
