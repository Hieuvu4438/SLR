# PHASE 2 — EMPIRICAL BOTTLENECK DISCOVERY FOR SIGN LANGUAGE RETRIEVAL

You have completed `docs/proposal7/SLRet_SOTA_Method_Proposal.md` and reached a scientifically valid NO-GO.

That NO-GO is a **checkpoint, not the termination of the research program**.

Do NOT restart proposal1–6.
Do NOT simply generate more architecture ideas.
Do NOT rename rejected methods.
Do NOT perform another broad literature review before obtaining new empirical evidence.

Your next task is to determine **which layer of the current SLRet pipeline causes the persistent retrieval errors**, and only then search for a new method targeted at that demonstrated bottleneck.

The research sequence is now:

```text
persistent retrieval errors
→ information recoverability experiments
→ bottleneck localization
→ targeted literature collision search
→ new research questions
→ candidate mechanisms
→ falsification
→ method
```

The method must be a consequence of new empirical evidence.

---

# 1. INPUT STATE

Treat the following as authoritative:

- `docs/proposal1` through `docs/proposal6`: CLOSED research families.
- `docs/proposal7/SLRet_SOTA_Method_Proposal.md`: current scientific checkpoint.
- All negative-results constraints remain active.
- No pretrained SEDS asset may be used.
- No new dataset or benchmark.
- Test data must not be used for method selection.
- PHOENIX dev is the first experimental environment.
- How2Sign and CSL-Daily are confirmation datasets only after a mechanism survives PH dev.

Read Proposal 7 carefully before doing anything else.

Do not repeat its completed literature review unless a newly discovered mechanism requires a targeted novelty search.

---

# 2. CENTRAL UNRESOLVED QUESTION

Proposal 7 established persistent retrieval errors but did not determine their source.

The central question is:

> Are the remaining retrieval failures caused primarily by information that is missing from the frozen visual representation, information that exists but is not extracted by the current scorer/representation, or errors that are not recoverable from the available visual/text evidence?

You must experimentally distinguish these possibilities.

Do NOT propose a method until this question has empirical support.

---

# 3. CREATE A RECOVERABILITY MATRIX

Build an experimental matrix over the persistent PH dev errors identified in Proposal 7.

At minimum compare four information regimes.

## R0 — Existing baseline

Use the corrected reproduced baseline exactly as defined by Proposal 7.

No new information.
No new parameters beyond the baseline.

Record full-gallery metrics and per-query ranks.

## R1 — Existing frozen features + additional capacity

Purpose:

> Determine whether the information already exists in the frozen CiCo/UPRet-style visual representation but the deployed scorer fails to exploit it.

Use the exact same frozen visual and text features as R0.

Permit a small diagnostic scorer with greater expressive capacity.

This is NOT a proposed method. It is an information-recoverability probe.

Requirements:

- train only on train;
- select only on dev;
- same positive relation;
- same candidate population;
- no test access;
- no additional visual input;
- parameter-count-matched controls;
- compute-matched controls.

Test whether a stronger scorer can correctly reorder baseline top-K confusions.

The diagnostic scorer should be powerful enough to detect recoverable information but small enough that improvement cannot simply be attributed to massive scale.

Possible diagnostic forms may include a small pairwise interaction network or another justified readout, but do not turn the probe into a publication claim.

## R2 — Additional raw visual information + controlled scorer

Purpose:

> Determine whether useful retrieval information was discarded before the frozen baseline features were produced.

Implement the crossed raw-information experiment proposed in Proposal 7.

Use permitted raw RGB only.

Do NOT use:

- SEDS pretrained assets;
- SEDS features;
- SEDS teacher outputs;
- external test information.

Keep the retrieval objective identical to the baseline.

The raw-information arm must be compared against:

- equal parameter capacity;
- equal compute where practical;
- uninformative/additional-processing controls;
- train-only shuffled or otherwise invalidated information controls.

Do not interpret a gain caused purely by a stronger pretrained raw encoder as evidence of missing raw information.

Pretraining exposure must be accounted for explicitly.

## R3 — Raw information + expressive diagnostic scorer

Purpose:

> Estimate an empirical upper bound on recoverability under the permitted information regime.

R3 is a diagnostic upper-bound experiment, not the final method.

Its purpose is to distinguish:

```text
R1 succeeds, R2 not needed
→ existing frozen representation contains useful information.

R2 succeeds, R1 fails
→ frozen representation likely discards useful information.

R1 and R2 both succeed
→ prefer exploiting existing representation unless R2 gives a distinct, reproducible gain.

R3 succeeds while R1/R2 fail
→ interaction between representation and scoring matters.

R0–R3 all fail
→ investigate semantic ambiguity, annotation limits, or unavailable information before creating another architecture.
```

---

# 4. DO NOT ONLY MEASURE GLOBAL R@1

For every arm inspect:

- full-gallery T2V R@1/R@5/R@10;
- full-gallery V2T R@1/R@5/R@10;
- MedR/MnR;
- baseline persistent-error subset;
- top-10 persistent errors;
- score margin between correct candidate and top incorrect candidates;
- rank movement;
- T2V/V2T agreement;
- initialization performance;
- learned improvement over initialization.

A diagnostic that only improves the selected error slice but damages full-gallery retrieval does not establish a viable method.

---

# 5. LOCALIZE THE INFORMATION IF RAW VISUAL INPUT HELPS

If and only if R2/R3 demonstrate attributable improvement, determine **what information is responsible**.

Do not immediately construct a multi-stream architecture.

Use controlled diagnostic perturbations of existing raw data.

Potential information factors to test include:

- temporal resolution;
- fine spatial hand detail;
- face/non-manual cues;
- upper-body motion;
- long-range temporal information;
- background/context;
- signer identity leakage;
- crop resolution;
- frame sampling.

These are diagnostics, NOT assumed bottlenecks.

Design interventions so that each factor can be falsified.

Example principle:

```text
raw RGB improves
→ determine which visual information causes improvement
→ only then design the smallest mechanism preserving/exploiting it
```

Do not propose “RGB + pose + face + flow + transformer”.

---

# 6. IF FROZEN FEATURES ALREADY CONTAIN THE INFORMATION

If R1 significantly improves persistent errors and full-gallery retrieval while using exactly the existing frozen features:

Do NOT change visual backbone.

Instead answer:

> What mathematical property of the current retrieval scorer prevents it from using information that is demonstrably present?

Analyze:

- interaction rank/capacity;
- temporal ordering sensitivity;
- compositional interactions;
- token-token versus pooled information;
- separability assumptions;
- representation collapse/anisotropy;
- similarity geometry;
- score calibration;
- asymmetric T2V/V2T ranking;
- global versus conditional candidate interaction.

These are hypotheses, not conclusions.

Measure them.

Then conduct a **targeted** literature search for the demonstrated mechanism.

Do not redo generic SLRet literature review.

---

# 7. HARD-PAIR RECOVERABILITY TEST

Proposal 7 found that many persistent errors are already in the baseline top-10.

Exploit this fact.

Construct train-derived hard-pair examples using only train information.

Then test on the fixed PH dev persistent-error population:

> Given the correct candidate and the baseline's strongest confusing candidate, can the representation distinguish them?

Measure at least:

```text
baseline scorer accuracy
frozen-feature diagnostic accuracy
raw-information diagnostic accuracy
combined diagnostic upper bound
```

Do NOT train directly on dev error labels.

Dev is evaluation only.

This experiment is critical because it distinguishes:

```text
ranking failure
vs
representation failure.
```

---

# 8. ERROR EXPLANATION AUDIT

For a representative stratified sample of persistent errors, automatically inspect:

- captions;
- candidate captions;
- video durations;
- baseline similarities;
- text similarity;
- feature similarity;
- source/signers if available;
- repeated-caption structure;
- model-input translations;
- nearest competitors.

Do not fabricate linguistic interpretations.

Label explanations:

`OBSERVED`
`INFERRED`
`REQUIRES SIGN-LANGUAGE EXPERT`

If an error cannot be interpreted from metadata/features, say so.

---

# 9. DECISION GATE

After the experiments, classify the bottleneck into one of these states:

### STATE A — FROZEN-INFORMATION BOTTLENECK

Existing features contain discriminative information but current retrieval formulation fails to exploit it.

Only now generate methods addressing the measured scoring/representation deficiency.

### STATE B — VISUAL-INFORMATION BOTTLENECK

Raw visual information provides significant attributable gain beyond capacity/compute controls.

Only now generate methods preserving or exposing the identified missing visual factor.

### STATE C — JOINT BOTTLENECK

Raw information and stronger interaction are both required.

Generate mechanisms specifically explaining their interaction.

Avoid generic multimodal fusion.

### STATE D — DATA/SEMANTIC LIMIT

No permitted information regime reliably resolves errors.

Do not invent another neural block.

Investigate annotation ambiguity, semantic underspecification, or irreducible candidate equivalence using existing data.

### STATE E — INCONCLUSIVE

Experimental implementation/provenance prevents attribution.

Fix the experiment rather than creating a method.

---

# 10. SUCCESS CRITERIA

Use the Proposal 7 pilot criteria unless new evidence justifies a stricter threshold.

At minimum, an information channel is considered supported only if:

- mean bidirectional PH dev R@1 improves ≥ 0.5 pp over the strongest matched control;
- improvement occurs in at least 2/3 seeds;
- paired source-cluster bootstrap lower confidence bound is > 0;
- neither direction loses more than 0.25 pp R@1;
- R@5/R@10 do not materially regress;
- persistent-error ranks improve in the expected direction;
- improvement occurs after initialization, not merely at epoch -1;
- shuffled/uninformative controls do not obtain equivalent gain.

Do not tune thresholds after observing results.

---

# 11. ONLY AFTER A STATE IS ESTABLISHED: METHOD DISCOVERY

Once STATE A/B/C is supported:

Generate at least 3 mechanistically different candidate methods derived from that state.

Each candidate must answer:

1. What measured failure does it address?
2. Why does the intervention follow from the experiment?
3. Why are simpler controls insufficient?
4. Why is it not Proposal 1–6 again?
5. What is the closest published mechanism?
6. What observation would falsify it?
7. What is the minimum implementation required?
8. Does it change inference cost?
9. Can its contribution be separated from backbone/pretraining scale?

Do not generate candidates unrelated to the measured bottleneck.

---

# 12. NOVELTY SEARCH MUST NOW BE TARGETED

Do NOT repeat the previous 39-query broad search.

Search literature using the experimentally discovered mechanism.

For example:

```text
"<measured mechanism>" video text retrieval
"<measured mechanism>" multimodal retrieval
"<measured mechanism>" sign language retrieval
"<measured mechanism>" contrastive retrieval
```

Search adjacent fields aggressively because a method may already exist under another name.

Reject candidates with material prior-art collision.

---

# 13. INTERNAL REVIEW BEFORE SELECTION

Review each surviving candidate as:

- novelty reviewer;
- retrieval reviewer;
- sign-language reviewer;
- reproducibility reviewer;
- causal-attribution reviewer.

The causal-attribution reviewer must ask:

> Does this method specifically exploit the evidence produced by R0–R3, or could the same gain plausibly come from generic additional capacity?

Reject candidates without an answer.

---

# 14. ITERATIVE RESEARCH RULE

Do not terminate immediately after one candidate set fails.

You are allowed up to **three empirical discovery cycles**.

A cycle must contain NEW information:

```text
new controlled experiment
OR
new dataset measurement
OR
new source-code finding
OR
new targeted prior-art collision
```

A new architecture name does NOT count as a new cycle.

After a failed cycle:

```text
update hypothesis
→ run the next discriminating experiment
→ reassess bottleneck
```

Do not repeat broad literature review.

Do not resurrect CLOSED proposal1–6 families.

If all three cycles fail, produce an evidence-backed research barrier report rather than fabricated novelty.

---

# 15. IMPLEMENT, DO NOT ONLY SPECULATE

Proposal 7 already specified the preliminary location:

`methods/information_probe/`

Use this as a diagnostic namespace.

Implement and run the lowest-cost valid experiments.

At minimum produce:

```text
methods/information_probe/
docs/proposal7/evidence/phase2/
docs/proposal7/PHASE2_RESEARCH_STATE.md
```

The state file must contain:

- experiment IDs;
- commit SHA;
- manifest hashes;
- configs;
- seeds;
- initialization metrics;
- final metrics;
- per-query rank deltas;
- control comparisons;
- bootstrap results;
- failed runs;
- bottleneck classification;
- next experiment.

Do not create `proposal8` yet.

---

# 16. WHEN A METHOD FINALLY SURVIVES

Only after:

```text
experimental bottleneck established
+
candidate mechanism derived from bottleneck
+
prior-art collision passed
+
cheap falsification passed
```

create:

`docs/proposal8/SLRet_Method_From_Evidence.md`

Proposal 8 must begin with:

```text
Observed bottleneck:
Experimental evidence:
Why current methods fail on this bottleneck:
Method hypothesis:
Minimum causal intervention:
```

A proposed method that cannot fill these five fields with measured evidence is rejected.

---

# 17. FINAL INSTRUCTION

Your objective is no longer:

> "invent a SOTA method."

Your objective is:

> "discover experimentally what prevents the current system from ranking correctly, then derive the smallest novel mechanism justified by that evidence."

Continue autonomously through implementation, measurement, diagnosis, targeted prior-art search, and method generation.

Do not stop at another conceptual NO-GO before performing the unresolved empirical probes that Proposal 7 itself identified.
