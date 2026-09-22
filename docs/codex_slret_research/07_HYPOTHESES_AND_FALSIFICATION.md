# Hypotheses, falsification and admission limits

2026-09-22. C27 is inherited work, not a newly selected primary method. Status:
**OPEN**. No new training code or pilot is admitted by this document.

Cycle12 [scope correction](evidence/ADMISSION_SCOPE_AUDIT.md): requirements for
independent linguistic contrasts below apply to the corresponding semantic
hypotheses, not all research. Official-pair computational/statistical diagnostics
do not require new semantic labels. Every candidate pilot still needs all six
guide section41 gates; available labels or generic errors alone do not suffice.

Cycle2 update: [saved-score residual analysis](evidence/GCN_RESIDUAL_RESULT.md)
finds zero strict persistent exact-order confusers in either direction, failing
the predeclared material-burden gate. The narrow order-confusion justification
is **REJECTED**. H1 below remains an untested broader efficacy claim, with low
current priority and unresolved novelty; it cannot be promoted from annotation
availability or feature recovery alone.

## H1: ordered gloss evidence improves deployed retrieval

Observation: [M] PH TRAIN has 7,096 nonempty gloss sequences and 1,085 gloss
types. Only 23 exact gloss-multiset classes contain distinct orders (140 rows);
22 classes/138 rows also contain at least a pair with different exact translation.
These are annotation counts, not retrieval-error burden or valid-negative labels.
The exact-order slice covers 1.973% of TRAIN rows; it is not a ceiling on all
possible CTC effects. Source: [audit](evidence/state_audit_20260922.json).

Hypothesis: [H] ordered sign supervision supplies rank-relevant information
beyond unordered gloss content in the current SEDS representation.

Prediction: CTC beats both native continuation and a matched order-free gloss
auxiliary on held retrieval and actual deployed full-gallery DEV scoring. The
difference should exceed ordinary seed variability, preserve both directions,
and appear on independently defined order-sensitive examples if those can be
validated without inventing semantic labels.

Alternative explanations: any gain comes from extra expert labels, generic
auxiliary regularization, extra trainable parameters, different batch/RNG order,
checkpoint selection opportunities, or ordinary GCN adaptation. A CTC loss
decrease alone distinguishes none of these explanations.

Falsification: if a matched unordered gloss objective reproduces the gain,
reject the **order-specific contribution**, even if CTC is a useful recipe.
If neither auxiliary beats native continuation, close the tested recipe without
a weight/horizon sweep. If the putative bottleneck cannot be shown, do not
describe generic retrieval errors as proof of missing order.

Minimal staged experiment — **PROPOSED — NOT MEASURED**:

1. Stage 0: exact TRAIN annotation census (completed), actual selected-video
   target join and real-mask CTC eligibility (Cycle7:13 failures confirmed), plus a positively adequate
   train-internal diagnostic that separates order from label content (not yet
   specified well enough to run). Publicly released weights saw TRAIN; a frozen
   probe holdout is not independent of that pretraining. Disclose this limit.
2. Stage 1: only after a usable diagnostic, compare matched heads for ordered
   CTC versus order-free gloss counts/content. Same vocabulary, input tokens,
   trainable encoder scope and head parameter count. Frozen-head-only success
   cannot itself demonstrate a retrieval change.
3. Stage 2: if admission gates pass, native continuation / CTC / order-free
   auxiliary with common initialization, exposure, batches and selector. Fix
   loss normalization, weight, horizon and seed before scores are observed.
   Mean bidirectional R1 is primary; proposed pilot gate is ≥0.5 pp over both
   controls with neither direction worse by >0.5 pp. Passing one seed supports
   replication, not statistical significance or SOTA.

A randomly permuted full gloss sequence is a stress control, not a valid sign
transcription. An order-free count/presence target provides the useful resource
control without pretending the shuffled sequence is linguistically correct.

## H2: CTC is technically feasible at the existing temporal resolution

Observation: [M] 56 TRAIN sequences contain adjacent repeated glosses; maximum
required CTC steps, length + adjacent repeats, is 30. [V] The implementation
checks this bound, but the actual number of valid windows can be smaller.

Prediction: all intended loaded TRAIN rows have valid count ≥ required count,
chronological starts, exact sample-ID joins and finite loss. Alternative:
compression/filtering removes needed evidence despite a formally valid count.

Falsification: any ineligible sample or ID mismatch blocks that loader recipe.
Do not hide invalid examples with `zero_infinity=True` or silent dropping.
Any policy change must apply equally to controls and preserve the official
evaluation gallery. This is an engineering gate, not a novel contribution.

**Cycle7–8 outcome:** H2's all-TRAIN prediction is falsified:13 examples fail
the bound in native loading. [Frame decomposition](evidence/FRAME_RETENTION_RESULT.md)
shows they have16/17 decoded frames, none removed by sampling/filtering, and
only1/2 native windows. This is not an omitted-frame or64-window-cap defect.
The unchanged recipe is blocked technically; no engineering repair is promoted
to a method or applied automatically.

## Top-question hypothesis contracts after Cycle3

These make the uncertainty explicit; they are not five admitted experiments.
Priority numbers in file06 rank questions, not methods. All proposed future
tests below are **PROPOSED — NOT MEASURED**, and resource/definition gaps cannot
be filled with artificial semantic labels or another generic calibration run.

### Q01 — Recoverability versus failed readout

- Observation: three selected GCN seeds share113/98 errors; both pose and RGB
  fail on88/74 of these in every seed. The older clean residual model failed
  held adequacy despite strong fit recall.
- Hypothesis: existing representations contain a specific rank-relevant
  distinction that the deployed score does not use.
- Prediction: an independently specified readout using the same representation
  recovers that distinction on held data and improves the corresponding ranks
  beyond equally trained capacity/compute controls.
- Alternative: absent cues, ambiguous relevance, inadequate training support,
  or a probe unable to learn the relevant function.
- Falsification: a valid, adequately powered negative result rejects only the
  specified readout/distinction—not all recoverability. Existing negative probes
  cannot upper-bound available information.
- Minimal experiment: first identify the distinction and a non-colliding
  function class with transferable supervision; then a same-input held probe.
  Neither is currently specified. Injected IDs, teacher copying and generic
  positive controls were already assessed and do not meet this requirement.

### Q22 — A specific scoring restriction, not scalar scores generally

- Observation: fixed nonnegative mixtures cannot repair common strict confusers;
  this concerns the three stored functions, not every scalar score.
- Hypothesis: an as-yet-unidentified restriction of the implemented scoring
  function prevents use of a demonstrated available distinction.
- Prediction: an admissible intervention removing only that restriction changes
  the predicted ordering on natural contrasts and transfers to held retrieval.
- Alternative: deficient encodings or optimization; unconstrained scalar scores
  can satisfy both directions, and rank-two embeddings can have arbitrarily many
  paired maxima. Gallery size and direction asymmetry alone are insufficient.
- Falsification: reject a claimed restriction when it already permits the target
  ranking, or when its proposed intervention reduces to a closed operator.
- Minimal experiment: source-derived necessity/separation test for a concrete
  restriction before any model fitting. No new restriction has survived yet;
  do not rerun the completed scalar/rank-dimension existence arguments.

### Q23 — Relevance ambiguity

User update, 2026-09-22: continue without human annotation. New review/label
collection is excluded. The draft review protocol is retained as declined history,
not an executable next step. The following evidentiary argument remains valid,
but cannot be used to keep asking for reviewers or to block automated research.

- Observation: persistent official-pair errors remain; repeated strings and
  gloss/text disagreements exist in historical annotation audits.
- Hypothesis: some strict official errors correspond to genuinely acceptable
  video–text matches or information absent from the sentence clip.
- Prediction: independent sign-language judgments corroborate that interpretation
  for specified confusers while distinguishing unambiguous wrong controls.
- Alternative: ordinary model error; gloss/German disagreement does not determine
  video truth, and distinct strings need not encode distinct signed meanings.
- Falsification: independent, appropriate judgments find the target errors
  unambiguous and visually distinguishable. This rejects the tested subset claim,
  not every source of ambiguity.
- Minimal experiment: use existing task-compatible independent judgments if a
  verified join becomes available. Current307 TRAIN gloss/German judgments are
  not video relevance labels; their scope audit is already complete. New human
  annotation needs separate authority and is not silently introduced here.

### Q17 — Simultaneous relational information

- Observation: historical raw pooled bins are not isolated articulators;
  permutation-invariant readouts did not test explicit spatial arrangement.
- Hypothesis: a particular simultaneous articulator relation distinguishes
  otherwise similar retrieval candidates and is lost at a specific interface.
- Prediction: independently validated contrasts preserve component content but
  differ in the relation; a same-resource relation-sensitive intervention
  recovers the distinction beyond component-only and correspondence-null controls.
- Alternative: ordinary lexical differences, timing/annotation error, nuisance,
  or extra capacity. Array-bin identity is not anatomical identity.
- Falsification: the relation is already retained/used, absent from the target
  failures, or its apparent gain is reproduced by the matched controls.
- Minimal experiment: first establish a valid existing cue/contrast join.
  PH2014T excludes the cited historical mouthing sequences; filename-based joins,
  teacher labels and another positional/spatial head cannot substitute for it.
  That resource route is closed; no experiment is currently admitted.
- Cycle6 external lead: [ASL-MTP scope check](evidence/ASL_MTP_DIAGNOSTIC_SCOPE.md)
  provides no PHOENIX cue/contrast join. A published ASL diagnostic is not evidence
  for this DGS hypothesis, and its release was not located. No priority or
  admission change; no new human or model-generated annotation.
- Cycle9: [Public DGS Corpus source check](evidence/PUBLIC_DGS_RESOURCE_SCOPE.md)
  identifies an existing same-language type/subtype and mouthing annotation
  resource. A small unchanged-export feasibility check is conditional on
  project-use permission; no corpus download or model experiment yet. These
  are not PHOENIX labels or automatically valid mouth-only contrasts.
- Cycle11: [PHOENIX14T-HS provenance check](evidence/HANDSHAPE_RESOURCE_SCOPE.md)
  finds no adequate independent visual-contrast evidence for this hypothesis.
  Do not convert weak handshape supervision into confuser judgments or silently
  substitute it for simultaneous-relation labels. No admission change.

### Q24 — Cross-dataset transfer of a supported mechanism

- Scope: this is one PH-first confirmation contract, not a universal order for
  discovery. A separately specified exploratory mechanism may originate on
  CSL/H2 under the same rules; post-hoc dataset switching is not confirmation.
- Observation: the new residual measurement concerns PH DEV only; CSL/H2
  galleries and resource contracts differ.
- Hypothesis: a specific mechanism supported on PH has the same predicted
  failure signature and intervention effect on another compatible dataset.
- Prediction: fixed, untuned signature definitions and intervention controls
  reproduce their direction under the second dataset's official relevance rule.
- Alternative: PH weather templates, source/signer structure, feature resources,
  or metric implementation account for the first result.
- Falsification: the predicted signature is absent, the effect disappears under
  matched controls, or resource/protocol differences explain it.
- Minimal experiment: after a mechanism survives PH, run its fixed zero-training
  signature check on CSL/H2 DEV before full transfer. No supported mechanism
  currently exists to transfer; copying generic error counts is not confirmation.

Decision: proceed to the bounded admission/collision screen, not another generic
probe. The missing definitions/resources are explicit scientific gaps, not a
claim of global impossibility or permission to invent a primary method.
