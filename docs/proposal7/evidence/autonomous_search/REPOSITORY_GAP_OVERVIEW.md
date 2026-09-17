# Repository-gap overview and method-discovery handoff

2026-09-16 · Version 1 · AI-assisted research synthesis using
academic-research-suite, inline. [Claim-intent passport](REPOSITORY_GAP_OVERVIEW_passport.json).
Status: **audited evidence consolidated; no method selected**.

## Bottom line

The audits establish several real implementation defects, but **none currently
establishes an attributable, novel remedy for the persistent retrieval errors**.
The most defensible route is to use these findings to make comparisons valid,
then target an unresolved representational or scoring limitation. Repairing a
loss, changing a mask, or borrowing a module is not by itself evidence of a
research contribution. This distinction follows the
[research loop, §§21–24 and 35](../../Astra_SLRet_Autonomous_Research_Loop.md)
<!--ref:Astra_SLRet_Autonomous_Research_Loop--><!--anchor:section:21-24,35-->.

This overview covers all requested names and the substantive findings recorded
in the current audit ledger. It is **not** a claim to have inspected every file,
reproduced every paper, or exhausted the scientific space. The earlier
[repository audit ledger](REPO_LED_WEAKNESS_RESEARCH.md)
<!--ref:REPO_LED_WEAKNESS_RESEARCH--><!--anchor:section:Pinned source comparison-->
and [37-question map](Q01_Q37_feasibility_audit.md)
<!--ref:Q01_Q37_feasibility_audit--><!--anchor:section:Full map: evidence, scope and permissible next action-->
retain detailed scope, provenance and negative results.

Evidence labels: [V] verified source behavior; [M] local measurement;
[A] author claim; [I] inference; [H] hypothesis; [U] unresolved.

## 1. Repository identity and coverage

| Requested name | Pinned source | What this overview covers |
|---|---|---|
| CMCM | `5d458719d1da2f082e188cc44705003d919e7e97` | CSA, CCG, temporal/covariance modules, numerical contracts and available integration |
| SAN | `82aba9cbc1beb403abef6e9a3875ca52479805c8` | Negative generator, collator, scoring, training selection, supplied TRAIN captions and resource scope |
| SEDS | `434e3f714fcb6a7d1f4001fb9a246bbd93ec0246` | Source only: enabled losses, loaders and temporal correspondence contract; no prohibited assets |
| UPRet | `046366227417e1d8ec14145965403462df345984` | Distribution sampling, transport reduction, combined objective, partial local integration and saved TRAIN diagnostics |
| CiCo | SLRT `38a4f7b00da7a858d59b7fabe5093876a84db8e0` | Retrieval encoder/scorer, masks, losses, gather gradients, sampling and strongest local control |
| SLRT | Same umbrella repository | CiCo is its retrieval implementation. Recognition, translation and avatar-generation projects are not additional comparable sentence-retrieval baselines. |

Pins and bounded scopes are recorded in the
[source ledger](REPO_LED_WEAKNESS_RESEARCH.md)
<!--ref:REPO_LED_WEAKNESS_RESEARCH--><!--anchor:section:Pinned source comparison-->.
The identity distinction is explicit in the
[official SLRT README](https://github.com/FangyunWei/SLRT#readme)
<!--ref:SLRT--><!--anchor:section:Sign Language Processing-->.
“UPRER” is interpreted as UPRet, as in the existing research state. Pins identify
inspected snapshots, not a certification of each repository's latest release.

## 2. Confirmed weaknesses versus unresolved consequences

### CMCM — strongest numerical evidence, missing trained attribution

- [V/M] The covariance square-root backward omits epsilon used in forward
  normalization. Valid PSD-input fixtures show substantial relative derivative
  errors at low trace; the isolated denominator correction restores agreement.
  [U] Actual trained activation exposure and retrieval harm remain unknown.
  [Covariance audit](CMCM_source_result.md)
  <!--ref:CMCM_source_result--><!--anchor:section:Covariance experiment-->.
- [V/M] The Gaussian loss is negative even for identical distributions; a
  realizable common-variance path drives the auxiliary expression without a
  finite lower bound in real arithmetic. Six unchanged-source checks validate
  the formula and derivative. [U] This does not show trained collapse or an
  unbounded complete training objective. [Gaussian audit](CMCM_gaussian_result.md)
  <!--ref:CMCM_gaussian_result--><!--anchor:section:Source and exact derivation-->.
- [V/I] An unconstrained augmentation gate can retain, cancel or amplify the
  augmentation displacement. A temporal causal mask is not a front-door
  identification certificate. [U] Full-paper identification assumptions and a
  compatible trained integration are not established by the inspected package.
  [Causal-scope audit](CMCM_source_result.md)
  <!--ref:CMCM_source_result--><!--anchor:section:Source and causal scope-->.

**Decision:** numerical corrections belong in a faithful B0 if that pipeline is
later available. Do not present stable KL, variance flooring or bounded gating
as a new method. Do not infer poor published performance from incomplete code.

### SAN — negative validity and input/protocol contracts

- [V/M] The generator lacks explicit duplicate/self-output guards and can
  produce conditional empty-pool or augmentation collisions. Duplicate negatives
  change CE weighting. These are conditional source findings, not a measured
  semantic false-negative rate. [Generator audit](SAN_negative_contract_result.md)
  <!--ref:SAN_negative_contract_result--><!--anchor:section:Verified source behavior-->.
- [M] The supplied 7,096 German TRAIN captions have zero distinct-caption
  zero/one-swap collisions under the inspected operations. Largest identical
  caption multiplicity is63, precluding an all-identical B64 batch in the
  supplied no-replacement configuration. These counterexamples therefore do
  not justify the previously considered repair campaigns.
  [TRAIN exposure](SAN_train_collision_result.md)
  <!--ref:SAN_train_collision_result--><!--anchor:section:Outcome-->.
- [V/U] Collation truncates to70 tokens in training and44 in evaluation.
  This is a verified width mismatch, not verified information loss or collision
  prevalence: a matched tokenizer/corpus exposure check has not been completed.
  [Pinned collator](https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/datasets.py#L192)
  <!--ref:SAN--><!--anchor:section:collate_fn-->.
- [V] The inspected launcher/training path uses the configured test loader for
  epoch selection despite naming outputs “dev.” This must be repaired to
  DEV-only selection in any local reproduction; it does not establish what
  unpublished author commands did. Source was read, not executed.
  [Pinned training loop](https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/train_vlp_v2.py#L277)
  <!--ref:SAN--><!--anchor:section:main-->.

**Decision:** distinguish missing actual negative-table/runtime provenance from
evidence that negative mining is ineffective. No new table, semantic labels,
deduplication training or generic hard-negative variant is justified here.

### SEDS — inspect the active loss, not every available function

- [V] Top3/top5 KL is optional, default false, and absent from all three supplied
  recipes. The enabled auxiliary branch is Pose–RGB clip-index matching with
  coefficient .4. Startup/configuration findings are reproduction issues, not
  proof of failed published training. [Activation gate](SEDS_loss_activation_result.md)
  <!--ref:SEDS_loss_activation_result--><!--anchor:section:Decision-changing finding-->.
- [V/U] Loaders check clip counts but do not certify cross-stream source-frame
  equality. [A] The paper describes shared preprocessing, a concrete alternative
  explanation under which correspondence is valid. No actual desynchronization
  has been measured. [Temporal contract](SEDS_temporal_contract_result.md)
  <!--ref:SEDS_temporal_contract_result--><!--anchor:section:Counterevidence and scope-->.

**Decision:** keep SEDS source-only. Neither an optional-loss critique nor missing
consumer-side provenance checks motivates a pose/OT/teacher repair method.

### UPRet — source discrepancy survives, efficacy rationale does not

- [V] The source averages row/column maxima of transport-weighted sample
  similarities, whereas the final paper specifies a full weighted sum. Sample
  pooling and training-only uncertainty are intentional. [Source/paper comparison](REPO_LED_WEAKNESS_RESEARCH.md)
  <!--ref:REPO_LED_WEAKNESS_RESEARCH--><!--anchor:section:Final-paper and baseline-readiness update, 2026-09-15-->.
- [M] The amplitude-controlled reduction change produces median encoder-gradient
  differences of .174891% visual/.220684% text at the partial checkpoint, below
  the locked1% screen. Synthetic ordering changes did not establish meaningful
  deployed gain. [Real-gradient screen](UPRET_real_gradient_result.md)
  <!--ref:UPRET_real_gradient_result--><!--anchor:section:Outcome-->.
- [I, new] Default auxiliary scores can perturb a logit margin by at most .75
  under the stated finite/unit-mass conditions. Independently, all three saved
  B32 no-OT losses imply strict deterministic-channel batch separation at the
  partial checkpoint. This rejects auxiliary-head-only success on those states,
  not poor generalization. [Bound and saved-loss inference](UPRET_loss_rank_bound_result.md)
  <!--ref:UPRET_loss_rank_bound_result--><!--anchor:section:Saved no-OT evidence-->.

**Decision:** no long max/sum, weight, temperature or uncertainty-head rescue.
The incomplete local UPRet run is not a competitive, resource-matched baseline.

### CiCo — deployed effects matter more than suspicious-looking code

- [V/M] Masked queries can read valid content and enter one scoring channel;
  the tested masked-input-to-valid-state contamination route is blocked.
  All seven earlier inference mask exclusions lose overall mean R1, although
  some persistent ranks improve. Training-consistent masking remains untested,
  not disproved. [Mask result](CICO_visual_mask_result.md)
  <!--ref:CICO_visual_mask_result--><!--anchor:section:What the source establishes-->;
  [Q05 evidence](Q01_Q37_feasibility_audit.md)
  <!--ref:Q01_Q37_feasibility_audit--><!--anchor:section:Full map: evidence, scope and permissible next action-->.
- [V/M] Separate channel CEs add a known agreement penalty relative to CE of
  averaged logits. No harmful realizable parameter update follows from arbitrary
  channel-logit perturbations. [Channel-loss result](CICO_channel_objective_result.md)
  <!--ref:CICO_channel_objective_result--><!--anchor:section:Finding-->.
- [V/M] Local-slice gather backward plus replicated loss/DDP averaging scales
  encoder gradients by1/world_size but not temperature in the tested algebra.
  [V] The historical local single-device control is not exposed to that path.
  [Distributed-gradient result](CICO_distributed_gradient_result.md)
  <!--ref:CICO_distributed_gradient_result--><!--anchor:section:Exposure check and decision-->.
- [M] Local pseudo-clip gap exposure is .615417%, below the locked10% gate.
  All7,096 inspected raw TRAIN lengths are at least16, so the short-input
  repeat-last padding case is not required there. Neither census establishes
  pretraining-resource exposure or semantic adequacy.
  [Pseudo-clip result](CICO_pseudoclip_result.md)
  <!--ref:CICO_pseudoclip_result--><!--anchor:section:Outcome-->;
  [Input-padding result](CICO_input_padding_result.md)
  <!--ref:CICO_input_padding_result--><!--anchor:section:Source behavior and measurement-->.

**Decision:** retain CiCo as the best-established local control foundation.
Padding, gradient scaling, cosine normalization and sampling observations are
not a license to repeat failed fixes as novel mechanisms.

## 3. The unresolved scientific gaps

The shared pattern is an **attribution gap**: a property is missing from a proof
or implementation contract, but that does not establish a prevalent, recoverable
retrieval failure. Conversely, a failed probe is not proof that the information
is absent. The question map documents these boundaries across the tested layers.
[Evidence/feasibility audit](Q01_Q37_feasibility_audit.md)
<!--ref:Q01_Q37_feasibility_audit--><!--anchor:section:What the audit changes-->.

| Priority for formulation, not automatic execution | Unresolved question | Required decision-changing evidence | Explicitly excluded shortcut |
|---|---|---|---|
| 1 — representation/scorer interface | Are rank-critical distinctions present but inaccessible to the actual scorer? Q01/Q22 | A positively calibrated test or actual-function restriction, with a realizable intervention and relevance to full-gallery ranking | Another inadequate residual head; counting failed probes as an information ceiling |
| 2 — relational grounding | Are temporal or simultaneous articulator relations lost, rather than merely local token content? Q04/Q06/Q17 | Valid existing cue support and a relation-isolating control; show what the representation/scorer cannot distinguish | Generic order loss, shuffled-caption semantic labels, pose/crop/position-head rescue |
| 3 — independent-query retrieval | Can gallery structure help one query without consuming other evaluation queries? Q03/Q02 | A concrete operator with a property not reducible to column offsets, channel weights, cohort assignment or the failed graph filters | Cohort headroom treated as independent retrieval; another reranker/graph mixture |
| 4 — ambiguity attribution | Which persistent errors reflect actual signed distinctions rather than annotation/text underspecification? Q09/Q23/Q31 | A permitted, existing validated source of those distinctions with trustworthy joins | New positives, new annotations, fuzzy-gloss equivalence, automatic alignments as independent truth |
| 5 — auxiliary-to-deployment transfer | When does an auxiliary signal improve deployed ranks rather than just its own fit? Q34 and CMCM | A distinct, open causal effect in a comparable trained pipeline; same-state batch evidence is insufficient | Generic distillation, gradient surgery, uncertainty-weight tuning or additional negative mining |

These are judgment priorities, not measured opportunity scores or GO candidates.
No row currently has all prerequisites for a new training launch. That limitation
is not an impossibility theorem. The complete37-row map remains authoritative;
this short table does not erase lower-priority questions or reopen closures.

## 4. How to adapt a paper idea without repeating a closed family

Select a paper **after identifying the specific missing property**, then extract
its mechanism rather than its module name. The mechanism must change what the
actual system can learn/express and survive both local-family and terminology-
independent prior-art checks. This is the guide's required sequence, not a claim
that a paper-derived candidate has already passed.
[Research loop, §§19–24](../../Astra_SLRet_Autonomous_Research_Loop.md)
<!--ref:Astra_SLRet_Autonomous_Research_Loop--><!--anchor:section:19-24-->.

The next candidate card must answer:

1. Which measured or analytically supported failure is targeted, and on which
   inputs/resources does it occur?
2. What exact idea is transferred from the primary paper? What prerequisite
   made it work there, and is that prerequisite valid for signed language here?
3. What changes in training/inference? Why cannot B0 already express the remedy?
4. Why is the mechanism distinct from the closed families—not just differently
   named, differently weighted, or placed at another layer?
5. What outcome would kill it cheaply? Which initialization, shuffle, capacity,
   compute and closest-prior controls can explain away an apparent gain?

Only after support exists should three materially different candidates be
generated and reviewed from novelty, retrieval, sign-language validity,
reproducibility and causal-attribution perspectives. **No three candidate names
are fabricated here to fill an empty template.**

## 5. Comparison target and advancement gate

The strongest existing local PH DEV control is the uniform three-model CiCo
score ensemble: **77.263969% mean bidirectional R1**, versus75.240848% for the
best single model. It carries three-model inference cost and is one ensemble,
not three independent method trials. [C32/C33 report](AS-C32_C33_result.md)
<!--ref:AS-C32_C33_result--><!--anchor:section:Fixed full-gallery metrics-->.

Retain B0 reproduced/corrected baseline, B1 capacity control, B2 compute control,
and B3 closest prior under matching resources. The unchanged advancement gate
requires ≥.5pp mean-R1 improvement in ≥2/3 seeds, positive paired-cluster CI
lower bound, no direction loss>.25pp, no R5/R10 loss>.5pp, persistent-slice
improvement, and failure of initialization/shuffle controls to explain the gain.
The conditional second-dataset requirement also remains mandatory before
Proposal8. [Proposal7 pilot contract](../../SLRet_SOTA_Method_Proposal.md)
<!--ref:SLRet_SOTA_Method_Proposal--><!--anchor:section:Gate P1-->;
[research loop, §§23–24 and39](../../Astra_SLRet_Autonomous_Research_Loop.md)
<!--ref:Astra_SLRet_Autonomous_Research_Loop--><!--anchor:section:23-24,39-->.

This number is a local control, **not a statement that current published SOTA is
77.263969%**. Resource regimes, protocol, task and published-result provenance
must be reconciled before any future SOTA claim. No published TEST score was
used to rank the hypotheses in this overview.

## 6. Coverage limits and continuation

This is a cross-repository evidence synthesis, not a new systematic literature
review or exhaustive pairwise paper comparison. Zero new cross-paper tension
pairs were assessed in this synthesis; the source/paper discrepancies and
counterevidence above come from the linked bounded audits. Human confirmation
is pending. No independent reviewer, author-runtime replication or full-runtime
skill integrity certificate is claimed.

Existing measurements were not rerun for this overview. The new UPRet saved-loss
calculation is explicitly posthoc. Public README/method sources and SAN source
paths were checked; no benchmark TEST contents, SEDS assets, model training,
external messages or upstream edits occurred. Missing compatible pipelines,
negative-table provenance and independent cue validation remain visible.

The requested overview is now available as a research checkpoint. Continue with
decision-changing formulation/source work in the priority table, not another
summary-only cycle. Ordinary source-corrected CiCo training has not been launched;
this overview does not treat the earlier unanswered scope question as approval.
No GO, Proposal8 or global research-barrier declaration is justified yet.
