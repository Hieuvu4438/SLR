# Claim–evidence ledger

## Material Passport

- Origin: academic-research-suite, inline experiment validation and argument audit.
- Updated: 2026-09-18, after locked PH TEST; no independent reviewer/cross-model review.
- Verification: full numerical replication and checkpoint DEV replay verified;
  PH TEST statistics analyzed and score/rank identities rechecked. No manuscript
  submission-readiness or publication certification is implied.

## Current thesis and claim ceiling

CSL TEST update2026-09-19: alltenfixedmodels evaluated and audited. Endpoint
effect+3.557256±1.533727pp, conditional caption-clusterCI[2.299635,4.810784];
DEV-selected contrast−.070862±.446384pp. Every corrected endpoint and each
trained corrected selected checkpoint is below initialmeanR1; seed42 selected
model is exactlyinitialization. Thus this extension strengthens the local
numerical-mitigation claim, not the accuracy/contribution-novelty claim. Its
795/798 DEV-caption overlap and post-PH registration remain explicit limits.
Earlier statements that CSL TEST is missing/pending are superseded here.

In these fixed local CiCo continuation regimes, changing only optimizer-moment
arithmetic to FP32 mitigates retrieval degradation across PH/CSL and three seeds.
PH campaign-held-out TEST confirms the direction and material endpoint effect.
This does **not** yield an improved DEV-selected PH model, a novel optimizer,
or a SOTA result. It is evidence for a scoped empirical numerical-mechanism
study, not yet a validated publishable alternative contribution.

The distinction is actionable: an endpoint comparison against a degraded
continuation can show a large treatment effect while checkpoint selection still
prefers the original model. Numerical controls and initialization must therefore
be visible alongside proposed-method contrasts in this study. Do not claim all
SLRet papers have this problem or that any published ranking has been overturned.

| Claim | Evidence | Counterevidence / current claim ceiling |
|---|---|---|
| CiCo reference is replayable | Complete519-query fresh replay; score delta0, unchanged ranks | Historical dev selected model and transferred features; not unseen confirmation or published test comparison |
| SEDS local release architecture/checkpoint can execute | Exact tensor loading, native evaluator, B16 backward and all696 nonzero gradient-bearing tensors | TRAIN subset, no optimizer updates; release DEV preprocessing unavailable in archive |
| Baseline protocol needs explicit split handling | Combined7615-row dev.pkl includes7096 train rows; source test selection; canonical dev census | Does not prove author experiment leakage or change published ranking |
| New method improves SLRet | No evidence yet | No candidate or pilot admitted |
| Native adapted SEDS continuation helps | Step111/222 measured,222updates complete | Both below initialization;selector keepsstep0. No training gain. |
| Native FP16 optimizer moment rounding is materially exposed | Source moments inherit dtype;97.5132% FP16 entries nonzero m/zero v;FP32 correction removes the phenotype | One seed and adapted PH recipe; do not generalize to all released training or other models. |
| FP32 moments preserve retrieval during this continuation | Matched222updates;finalmeanR1 +5.298651pp vsnative;matched RNG/assets/initialscores verified | Strongest initial baseline still better than final;selected111 gains only0.096339pp and T2V R5 drops0.963391pp. Belowpilotgate;no method or independent confirmation. |
| UPRet partial checkpoint inference is replayable | Full519 DEV ranks match history exactly,1038feature hashes verified | Training incomplete;low recall cannot characterize fully trained UPRet or support SOTA claims. |
| Alternative publishable contribution | Not validated | Asset gaps and small engineering fixes alone are insufficient |
| FP32 moments reduce CiCo continuation degradation | PH DEV +7.675016±1.253643 pp; CSL DEV +4.252678±.834409 pp, all six paired seed effects positive; twelve checkpoint replays bit-exact | Fixed continuation seeds from shared selected initialization, not full-training seeds. PH selected delta0; CSL selected mean+.020912pp, mixed signs; no accuracy gate passes. |
| PH campaign-held-out queries confirm the endpoint effect | Locked seven-model TEST, +7.113188±2.473074 pp; conditional query-cluster 95% CI [5.120159,9.202454], all three seeds positive | Seed SD and query CI describe different uncertainty. Historical TEST/pretraining provenance incomplete; fixed-gallery conditional CI, inferred clusters. Corrected endpoints do not exceed initial meanR1; all DEV-selected models remain identical. |
| CSL held-out videos replicate the endpoint effect | Locked ten-model TEST, all three positive; +3.557256±1.533727pp, conditional caption-clusterCI[2.299635,4.810784] | Caption overlap795/798 withDEV, not novel-query confirmation. Selected contrast−.070862pp; no corrected model improves initialmeanR1. Selected1337 V2T−.510204pp violates earlier.5pp guardrail. |
| Stored numerical phenotype appears beyond SEDS | Provenance-locked CiCo PH/CSL FP16 states:92.5047%/85.3035% nonzero-first/zero-second elements; FP32 UPRet/SEDS stateszero; matched CiCo interventions now complete | Different optimizers/stages; stored state alone cannot establish cause. UPRet is only an exposure negative control, not a fully trained retrieval comparison. Familiar numerical issue is not algorithmic novelty. |
| CiCo arithmetic loses nonzero second moments on real gradients | SamegradientB512 FP16vsFP32 arithmetic,98.0297% PH/97.3666% CSL elements losepositivev; FP32negativecontrol exact; sourceweights unchanged | Onlyisolatedfresh-momentstep, not historicalstate repair or recallcausality. CSLB32gatefailed; batchfidelity matters. Knownmixedprecisionissue, not algorithmicnovelty. |
| Same-I3D tail continuation improves the strongest baseline | Matched222updates, exact initialscores/RNG pairing;36tailtensors changed; freshDEVfeatures evaluated | Selector retainsinitialization; -0.096339pp vs strongestcontrol; bothtrainedendpoints faildirectionalguardrail. Negative fixedconfiguration, not a method gain. |

## Argument audit

1. **Exposure is real, not inferred from source alone.** Stored state census and
   same-real-gradient arithmetic agree. Counterargument: different magnitudes
   and batches can change exposure. Concede: CSL B32 fails the activation gate;
   B512 is a separately registered recipe-fidelity check, not a silently replaced
   negative. Neither proves historical author-training damage.
2. **Precision intervention changes the tested trajectories.** Matching initial
   gradients, batch IDs, processed inputs, LR and RNG rules out those alternative
   explanations in the six CiCo pairs. FP32-parameter arithmetic controls are
   exact; native foreach/single-kernel FP16 weight outputs agree in the probes.
   Counterargument: both first and second moments change together. Concede:
   the training causal claim concerns moment arithmetic jointly; denominator-only
   evidence is one-step arithmetic, not proof of a unique long-run mediator.
3. **Endpoint mitigation replicates, selected accuracy gain does not.** All seeds,
   both datasets and PH TEST support the scoped endpoint claim. Counterargument:
   this is a restart from a good selected checkpoint with fresh optimizer state,
   not training a paper baseline from scratch. Concede exactly that scope; report
   both arms versus initialization and the DEV-selected comparison. No claim
   that a new best retriever has been produced.

The first two claims have strong direct evidence within the tested pipeline.
The third has strong local endpoint evidence but limited external generalization.
Algorithmic novelty and publishability remain unestablished; neither a passed
gate nor a bootstrap interval settles them. CSL TEST input feasibility is being
checked; no CSL TEST result is part of this audit yet.

## Statistical validation (11/11 checked)

Verification status: ANALYZED for PH TEST inference; VERIFIED applies only to
the actual checkpoint/score replay tasks. Confidence label CAUTION because
cluster independence and historical exposure cannot be fully verified. No p-value
was computed; none is inferred from the bootstrap interval. Effects are absolute
percentage points, not relative percentages or standardized effect sizes.

| Check | Finding / limitation |
|---|---|
| 1. Simpson | Report datasets, directions and every seed separately; no pooled benchmark score. |
| 2. Ecological | No claim about individual signers, linguistic meaning or all SLRet models. |
| 3. Selection/Berkson | Shared historical selected initializations restrict the population; no random-model sample. |
| 4. Collider | All fixed pairs are included, not only successful endpoint/selected outcomes. |
| 5. Base rate | Full official ID-based galleries retained; no diagnostic-sensitivity claim. |
| 6. Regression to mean | Matched controls and initialization reported together; no uncontrolled pre/post gain. |
| 7. Survivorship | Seven locked PH models and all six DEV pairs required; all attempts/failures remain in ledger. |
| 8. Look elsewhere | One primary PH TEST endpoint contrast fixed before access; directions and R5/R10 descriptive. Earlier DEV exploration is disclosed, not fresh confirmation. |
| 9. Forking paths | Fixed endpoints, selectors, bootstrap seed/draws and cluster rule; B32-to-B512 amendment retained. CSL confirmation would be a later registered extension, not part of the original PH lock. |
| 10. Causality | Controlled moment-arithmetic intervention supports a local trajectory effect; no isolated long-run second-moment mediation or author-training claim. |
| 11. Reverse causality | Precision is assigned before updates; hashes bind the outputs to that assignment. |

## Evidence-led outline (not a manuscript or approved submission plan)

Working title: **Optimizer-moment precision and checkpoint selection in local
sign-language retrieval continuations**. RQ: under fixed data, initialization,
optimizer schedule and exposure, does moment precision alter continuation
degradation, and does that effect survive comparison to initialization and
DEV-selected models? Scope: the local SEDS/CiCo regimes only.

| Section | Purpose and evidence | Approximate words |
|---|---|---:|
| 1. Problem and scope | Distinguish local endpoint recovery from best-model improvement; cite the baseline papers and numerical prior with their documented reading scope. No first-ever claim. | 500 |
| 2. Data/protocol and numerical intervention | Protocol table, transfer-aware resources, translated queries, grouped relevance, fresh optimizer resets, paired exposure and selectors; BASELINE_AUDIT and locked protocols. | 900 |
| 3. Exposure and arithmetic controls | Stored-state census, B32 negative, B512 same-gradient contrasts, FP32 and kernel controls. Establish what is measured before interpreting retrieval. | 700 |
| 4. Trajectory and held-out results | All-seed DEV curves/table, fixed and selected contrasts, PH TEST complete metrics and conditional interval; retain SEDS/RGB negative results. | 1000 |
| 5. Limits and reproducibility | Shared initializations, one SEDS seed, incomplete UPRet, conditional clusters, known correction, incomplete historical exposure and CSL DEV-caption overlap. Availability/retention, not public data release. | 600 |
| 6. Conclusion | Scoped numerical mitigation; no improved selected PH retriever or SOTA. | 300 |

Total proposed main text 4000 words, no draft generated. Argument progression:
define the comparison → verify exposure → intervene under controls → test both
endpoint and selected accuracy → limit generalization. Publication-specific
formatting, human authorship/CRediT, funding/conflict/ethics declarations and
venue-specific AI disclosure need human confirmation before a real submission.

## Missing evidence / next decision

- Fixed CSL TEST preparation and confirmation are now complete. Its positive
  endpoint result does not establish novel-caption generalization or selected
  model improvement; do not rerun it as an unopened confirmation set.
- Fresh initialization/full-training replication and multi-seed SEDS would be
  needed for broader training claims; not implied by continuation seeds.
- A first-vs-second-moment training factorial would be needed for unique-mediator
  attribution; no such claim is made here and no extra factorial is admitted.
- Broader nearest-prior review and independent domain review remain
  necessary before claiming a publishable novelty gap. Prior mixed precision is
  already acknowledged; do not rebrand this correction as a new optimizer.

## Post-CSL claim assessment

The local causal question has positive evidence: matched moment-arithmetic
intervention changes the tested continuation trajectories and the fixed-endpoint
effect transfers to both datasets' held-out videos. The best-model question has
negative evidence: initialization remains at least as good in meanR1 as all
corrected TEST endpoints, and selected comparisons do not show a validated gain.
The study has not established that this known correction supplies a sufficiently
new or generalizable contribution for publication. Keep the alternative claim
unvalidated rather than promote the mechanism gate into editorial acceptance.

External-theory check NUMERICAL_PRIOR_VERIFICATION.md further limits explanation:
the cited quantization convergence theorem's assumptions exclude underflow;
our evidence is empirical, not an application or proof of that theorem. Both
moments changed jointly in training, so unique second-moment mediation is not
established by the one-step denominator-swap diagnostic.

Statistical fallacy scan11/11 remains applicable to CSL with these updates:
alltenmodels included (survivorship), official full groupedgallery preserved
(base rate), one preregistered primary endpoint contrast and descriptive selected
contrasts (look-elsewhere), post-PH extension disclosed (forking paths), no
inference to novel captions/signers (ecological/selection). Other rows above
retain their paired-control/initialization and local-causality restrictions.
Confidence CAUTION; queryclusterCI is not a training-seed confidence interval.

No further training based on either now-opened TEST is appropriate for this
claim. A subsequent broader study would need a new prospectively defined scope,
stronger initializations/full-training controls and independent confirmation;
this document does not authorize or claim those experiments were performed.

Final adjudication: INCONCLUSIVE_OR_BLOCKED under goal §0(3), specifically
insufficient evidence for a novel/generalizable alternative contribution or
improved retriever. This is not a claim that numerical mitigation is absent;
it separates a supported empirical observation from the unmet research ambition.
