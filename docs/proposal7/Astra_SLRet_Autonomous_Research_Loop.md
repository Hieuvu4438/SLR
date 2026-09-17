# ASTRA — AUTONOMOUS RESEARCH LOOP FOR SIGN LANGUAGE RETRIEVAL METHOD DISCOVERY

## Mission

You are continuing an ongoing research program on **sentence-level Sign Language Retrieval (SLRet)**.

The project has already completed:

- proposal1 → proposal6: multiple method families that have been explored and are now CLOSED;
- proposal7: a rigorous research audit that ended in a defensible NO-GO for the candidates examined so far.

Proposal 7 is **not the end of the research program**.

It is only the latest research checkpoint.

Your objective is now:

> Continue autonomous evidence-driven research until you identify at least one genuinely defensible, experimentally testable, materially novel SLRet method candidate — or until you can demonstrate a much stronger research barrier than Proposal 7 currently establishes.

Do NOT invent a method merely to satisfy this instruction.

Do NOT stop simply because one candidate set fails.

Do NOT interpret Proposal 7's NO-GO as evidence that the search space is exhausted.

The correct workflow is:

```text
existing research state
→ map unexplored hypothesis space
→ identify unresolved mechanisms
→ gather new evidence
→ formulate research questions
→ generate materially distinct candidates
→ targeted prior-art collision search
→ adversarial review
→ cheap falsification / diagnostic experiment
→ update research state
→ explore a DIFFERENT branch if needed
→ repeat
→ select method only when evidence supports it
```

---

# 1. AUTHORITATIVE PROJECT STATE

Read before doing any new research:

```text
docs/proposal1/
docs/proposal2/
docs/proposal3/
docs/proposal4/
docs/proposal5/
docs/proposal6/
docs/proposal7/
methods/
shared/
slr_common/
third_party/
configs/
scripts/
tests/
patches/
```

Treat the following as authoritative:

## 1.1 Closed research families

All research directions implemented or conceptually explored in proposal1–proposal6 remain CLOSED.

This includes, at minimum:

- ELSC;
- DIVE-SLR;
- PLEL;
- OCEM;
- SSSC / Shared-Support Sign Contrast;
- sampling-consistent / partial-alignment variants;
- PMGR;
- RPCA;
- R1–R5 families already rejected;
- close variants, renamed versions, and mixtures whose core causal mechanism is materially equivalent.

Do NOT reopen a closed branch merely by:

- replacing one loss;
- changing a margin;
- adding uncertainty;
- adding pose;
- replacing attention with OT;
- using a different negative miner;
- changing a teacher;
- changing the layer where the same supervision acts;
- combining two rejected mechanisms.

If overlap is suspected, perform semantic collision analysis.

If the new candidate is conceptually equivalent, reject it.

## 1.2 Proposal 7

Read:

`docs/proposal7/SLRet_SOTA_Method_Proposal.md`

Proposal 7 established important evidence, including:

- persistent PHOENIX dev errors across seeds;
- many persistent errors already located inside baseline top-10;
- weak evidence for several previously suspected bottlenecks;
- fairness/protocol issues across public baselines;
- no surviving method candidate from its own search cycle;
- unresolved uncertainty regarding where useful information is lost or unused.

Treat those as a **research state**, not a universal theorem.

Proposal 7 only rules out what it actually tested or sufficiently falsified.

Do NOT assume every untested research mechanism is invalid.

---

# 2. NON-NEGOTIABLE CONSTRAINTS

Do NOT propose:

- a new dataset;
- a new benchmark;
- a new official split;
- a different retrieval task as the main contribution;
- a SOTA claim obtained solely from stronger backbone scale;
- a SOTA claim obtained solely from more training data;
- test-time leakage;
- test-based model selection.

Do NOT use:

- pretrained SEDS checkpoints;
- SEDS-distributed RGB features;
- SEDS-distributed pose features;
- SEDS teacher outputs;
- SEDS-derived cached representations.

SEDS may only be read/audited as prior work.

Primary evaluation remains sentence-level:

- PHOENIX-2014T;
- How2Sign;
- CSL-Daily.

No test information may be used for:

- hypothesis generation;
- negative mining;
- checkpoint selection;
- hyperparameter tuning;
- candidate ranking;
- method selection.

---

# 3. SCIENTIFIC EVIDENCE LABELS

Use these labels consistently:

`[V] VERIFIED` — directly verified from paper/code/data.

`[A] AUTHOR CLAIM` — reported by authors but not independently reproduced.

`[M] MEASURED` — measured locally during this investigation.

`[I] INFERENCE` — reasoning from evidence.

`[H] HYPOTHESIS` — unverified scientific hypothesis.

`[U] UNVERIFIED` — not established.

Never convert:

- `[A]` into `[M]`;
- `[I]` into `[V]`;
- `[H]` into a claimed result.

Never invent missing results.

---

# 4. IMPORTANT CHANGE FROM PROPOSAL 7

Proposal 7 explored a limited set of candidate mechanisms and then stopped at NO-GO.

You must now continue the search.

However, continuing does **not** mean generating more random architectures.

The research program must become more systematic.

Your first task is to construct an explicit:

# OPEN RESEARCH SPACE MAP

This map must distinguish:

1. mechanisms already explored and CLOSED;
2. mechanisms partially investigated;
3. mechanisms discussed in literature but not tested locally;
4. mechanisms not yet seriously investigated in SLRet;
5. mechanisms suggested by local error structure;
6. mechanisms whose novelty is uncertain;
7. mechanisms whose feasibility is uncertain.

Do not collapse all future research into one hypothesis such as:

- raw visual information;
- hard negatives;
- local alignment;
- pose;
- uncertainty;
- context.

The point is to avoid another local optimum in research reasoning.

---

# 5. MAP THE SLRET SYSTEM BY FAILURE LAYER

Analyze the full retrieval pipeline as separable scientific layers.

At minimum consider:

## Layer A — Data / relevance relation

Questions may include:

- Are positive relations truly one-to-one?
- Are multiple valid realizations represented correctly?
- Does spoken-language text underspecify signed realization?
- Are repeated captions semantically equivalent or only textually identical?
- Is the relevance relation asymmetric between T2V and V2T?
- Are annotation groups aligned with the objective actually optimized?

Do not create new annotations or benchmarks.

Use this only to understand current data.

## Layer B — Visual information acquisition

Questions may include:

- Is discriminative information already lost before frozen features?
- Are hands, face, non-manual markers, spatial loci, simultaneity, or fine motion underrepresented?
- Is temporal sampling too coarse?
- Does the backbone encode actions but discard linguistic contrasts?

Do not assume yes.

Measure first.

## Layer C — Visual representation

Questions may include:

- Are temporally distinct signs collapsed?
- Is simultaneous articulation represented compositionally?
- Is signer/background invariance too weak or too strong?
- Is the representation over-contextualized?
- Is local information destroyed by contextual mixing?
- Is geometry anisotropic or low-rank?

## Layer D — Text representation

Questions may include:

- Is spoken-language semantic encoding too coarse?
- Does English translation collapse distinctions present in native captions?
- Is text compositional structure relevant?
- Does the model encode lexical semantics while missing relational structure?
- Are translated queries introducing systematic semantic equivalence classes?

## Layer E — Cross-modal interaction

Questions may include:

- Is independent pair scoring the wrong abstraction?
- Is token-video late interaction unable to express certain relations?
- Does compatibility require structured matching rather than scalar similarity?
- Are simultaneous/manual/non-manual relations represented incorrectly?
- Does pair scoring fail to model conditional ambiguity?

## Layer F — Ranking formulation

Questions may include:

- Training often learns absolute compatibility, while retrieval requires relative ranking.
- Is pairwise similarity sufficient when ranking is set-relative?
- Are top competitors ignored during scoring?
- Is candidate-relative reasoning useful?
- Should ranking geometry differ for T2V and V2T?
- Is one shared metric space unnecessarily restrictive?

## Layer G — Optimization

Questions may include:

- Do retrieval directions conflict?
- Does gradient averaging suppress rare distinctions?
- Are hard examples underweighted?
- Does the objective optimize average similarity but not rank-critical margins?
- Is representation learning bottlenecked by optimization rather than architecture?

## Layer H — Generalization / nuisance

Questions may include:

- Are models exploiting signer identity, scene, template, source-video context?
- Are those cues helpful or harmful?
- Are standard augmentations destroying sign-relevant information?
- Is domain adaptation learning the wrong invariances?

## Layer I — Uncertainty / relevance ambiguity

Distinguish:

- embedding uncertainty;
- annotation uncertainty;
- semantic underspecification;
- multiple valid signed realizations;
- ranking uncertainty.

Do not assume Gaussian embedding uncertainty is the only formulation.

## Layer J — Computational retrieval structure

Questions may include:

- Does the scoring rule depend on candidate set structure?
- Can ranking benefit from candidate interactions?
- Are there structured retrieval formulations that preserve full-gallery evaluation?
- Is retrieval better formulated as structured prediction than independent similarity scoring?

The above are research dimensions, not proposed methods.

---

# 6. OPEN RESEARCH SPACE TABLE

Create a table with at least 20 distinct research mechanisms/questions distributed across the layers above.

For every entry record:

```text
Mechanism / question
Layer
Evidence from current project
Closest prior work
Already tested locally? yes/no/partial
Collision with proposal1–6
Scientific plausibility
Novelty uncertainty
Feasibility
Best cheap experiment
Current status
```

Do not pick a method before this map exists.

---

# 7. PRIORITIZE BY INFORMATION GAIN, NOT BY ARCHITECTURE FASHION

Rank research questions according to:

```text
Expected information gain          0–20
Evidence that the issue exists     0–20
Potential retrieval relevance      0–20
Novelty opportunity                0–15
Falsifiability                     0–10
Implementation feasibility         0–10
Distance from CLOSED families       0–5
```

The first research cycle should investigate the highest-value unresolved question.

Do not prioritize something merely because it sounds publishable.

---

# 8. RESEARCH CYCLE DEFINITION

A valid research cycle must produce NEW evidence.

A cycle contains:

```text
Research question
→ falsifiable hypothesis
→ targeted evidence gathering
→ code/data/paper audit
→ diagnostic or small experiment
→ interpretation
→ candidate consequences
→ update open research map
```

A cycle counts only if it creates at least one of:

- new local empirical measurement;
- new controlled experimental result;
- new baseline/code finding;
- new dataset finding;
- new prior-art finding;
- falsified mechanism;
- supported mechanism;
- corrected assumption;
- newly discovered research gap.

Generating a new method name does NOT count as a research cycle.

---

# 9. DO NOT IMMEDIATELY STOP AT NO-GO

If one research question fails:

DO NOT conclude global NO-GO.

Instead:

1. mark the hypothesis falsified or unsupported;
2. update the Open Research Space Map;
3. choose a materially different high-information branch;
4. perform the next research cycle.

Do not stay trapped in the same family.

Examples:

```text
local evidence hypothesis fails
→ do NOT try another local attention loss.

hard-negative hypothesis fails
→ do NOT try another negative miner.

raw-information hypothesis fails
→ do NOT try another backbone immediately.

ranking-relative hypothesis fails
→ do NOT keep renaming reranking modules.
```

Move to a different causal layer.

---

# 10. STOPPING RULE

There is no arbitrary fixed limit such as "three cycles".

Continue while new high-information research branches remain unexplored and feasible.

You may stop only when one of these conditions holds:

## SUCCESS

A candidate survives:

- empirical support;
- novelty collision search;
- adversarial review;
- cheap falsification;
- baseline fairness;
- feasibility.

Then proceed to Proposal 8.

## RESEARCH BARRIER

You have strong evidence that all high-priority feasible branches in the current Open Research Space Map are:

- empirically unsupported;
- already solved by prior work;
- impossible under project constraints;
- or not attributable under available resources.

A Research Barrier conclusion requires an explicit exhaustion table.

Do NOT use "I could not think of another idea" as a stopping condition.

---

# 11. USE PROPOSAL 7'S INFORMATION-PROBE IDEA AS ONE BRANCH, NOT THE WHOLE PROGRAM

Proposal 7 identified an unresolved experiment:

> Does additional raw visual information help beyond frozen-feature / equal-capacity controls?

This remains a valid high-value branch.

Run it if it ranks highly in the information-gain table.

However:

Do NOT assume in advance that visual-information loss is the dominant bottleneck.

Treat it as one branch among several.

Possible outcomes:

```text
raw helps, frozen scorer does not
→ investigate visual information loss.

frozen scorer helps
→ investigate scoring/representation bottleneck.

both help
→ investigate interaction.

neither helps
→ move to another research layer.
```

Do not stop the entire project if this branch fails.

---

# 12. HARD-PAIR ANALYSIS AS A GENERAL TOOL

Proposal 7 found persistent errors across seeds and many correct candidates inside top-10.

Use this evidence strategically.

For baseline persistent errors, ask:

> What distinguishes rank 1 from the correct rank 2–10 candidate?

Possible controlled diagnostics:

- frozen-feature separability;
- raw-feature separability;
- text representation separability;
- temporal order sensitivity;
- candidate-relative margins;
- pairwise versus listwise ranking;
- score calibration;
- direction-specific geometry;
- nuisance sensitivity.

Do not train directly on dev error labels.

Train-derived analogues must be constructed from train.

Dev is evaluation.

---

# 13. EXPLORE WHETHER INDEPENDENT PAIR SCORING IS ITSELF A BOTTLENECK

Most existing SLRet systems ultimately assign a compatibility score to each `(video, text)` pair independently.

Investigate whether that assumption is empirically limiting.

Questions:

- Does the score for a correct candidate depend on which competitors are present?
- Are persistent mistakes resolvable using relative candidate structure?
- Would a listwise formulation change rank without requiring new visual information?
- Are T2V and V2T geometries fundamentally asymmetric?
- Does forcing one scalar similarity function lose useful ranking structure?

Important:

This is NOT permission to invent a reranker.

First perform diagnostics.

Examples:

- compare pairwise margin vs listwise rank loss;
- candidate-conditioned calibration using train-only fitting;
- direction-specific scoring probes;
- listwise oracle upper bounds;
- competitor-feature probes;
- top-K set structure analysis.

Any candidate-relative method must still evaluate on the standard full gallery.

---

# 14. EXPLORE REPRESENTATIONAL SUFFICIENCY

Investigate whether baseline representations contain information required for correct ranking.

Potential probes:

- linear probe;
- shallow nonlinear probe;
- pair discriminator;
- temporal-order probe;
- phrase/video relation probe;
- top-confuser discriminator;
- mutual-information proxy;
- CKA / representational similarity;
- effective rank / anisotropy;
- perturbation sensitivity.

These are diagnostics, not publication methods.

Ask:

> Is the information present but inaccessible to the deployed scoring rule?

If yes, derive methods from the measured structural deficiency.

---

# 15. EXPLORE DIRECTION ASYMMETRY

Do not assume T2V and V2T should share identical geometry.

Measure:

- error overlap;
- margin distributions;
- neighborhood structure;
- calibration;
- candidate multiplicity;
- representation sensitivity;
- rank disagreement.

Ask:

> Is one shared symmetric retrieval formulation unnecessarily constraining two asymmetric ranking problems?

Search prior work only after evidence supports this.

Do not reuse PMGR merely because grouping differs.

---

# 16. EXPLORE COMPOSITION AND TEMPORAL STRUCTURE

Do not equate this with local alignment.

Ask:

- Is temporal order actually encoded?
- Are simultaneous articulators represented?
- Does the scorer distinguish the same components in different relations?
- Are contextual video tokens losing event structure?
- Can frozen representations distinguish compositional minimal contrasts?

Use train-derived controlled pairs where possible.

Avoid creating synthetic negatives unless they are only diagnostic and validity is explicit.

---

# 17. EXPLORE RELEVANCE RELATION AND SEMANTIC UNDERSPECIFICATION

Ask:

- Does one caption correspond to multiple visually valid realizations?
- Are some baseline "errors" actually annotation-equivalent?
- Are repeated textual forms collapsing distinct signed realizations?
- Are translations losing distinctions?
- Does sentence retrieval require modeling a distribution over valid realizations?

Do not redefine test positives.

This analysis is for understanding the problem.

Any method must retain the official evaluation protocol.

---

# 18. EXPLORE OPTIMIZATION BOTTLENECKS

Before designing new architecture, determine whether training dynamics already reveal failure.

Measure:

- positive/negative gradient conflict;
- T2V/V2T gradient alignment;
- hard/easy sample contributions;
- rare versus frequent caption gradients;
- margin evolution;
- representation drift;
- early-learning versus late-overfitting;
- seed instability.

Do not reopen RPCA or generic gradient surgery automatically.

Only derive a new method if an unaddressed optimization mechanism is measured.

---

# 19. TARGETED LITERATURE SEARCH ONLY AFTER MECHANISM DISCOVERY

Do not redo another broad SLRet survey unless necessary.

Once an empirical mechanism appears promising, search:

```text
"<mechanism>" sign language retrieval
"<mechanism>" video text retrieval
"<mechanism>" multimodal retrieval
"<mechanism>" ranking
"<mechanism>" contrastive learning
"<mechanism>" structured prediction
```

Search adjacent domains aggressively.

Novelty must survive terminology mismatch.

A mechanism that already exists under another name is not novel.

---

# 20. SOURCE PRIORITY

Use:

1. official proceedings;
2. publisher/journal page;
3. arXiv/OpenReview;
4. official author project/repository;
5. cited references.

Do not base novelty claims on:

- blogs;
- AI summaries;
- random GitHub repos;
- Papers With Code alone.

If a relevant paper cannot be accessed, mark the collision risk OPEN.

---

# 21. CANDIDATE GENERATION RULE

Only generate candidates after at least one new mechanism has empirical or strong analytical support.

Generate at least 3 materially different candidates from the supported finding.

Each candidate must specify:

```text
Observed failure
Evidence
Causal hypothesis
Mechanism
What changes in training
What changes in inference
Why baseline cannot express this
Closest SLRet prior
Closest adjacent prior
Collision with proposal1–7
Cheap falsification
Failure criterion
Compute impact
```

No kitchen-sink methods.

---

# 22. INTERNAL ADVERSARIAL REVIEW

For each candidate run five reviewers:

## Reviewer A — Novelty

> Is this already known under another name?

## Reviewer B — Retrieval

> Why should this improve standard full-gallery R@1 rather than an auxiliary diagnostic?

## Reviewer C — Sign-language validity

> Does this respect properties of sign language rather than impose spoken-language assumptions?

## Reviewer D — Reproducibility

> Can this actually be implemented and compared fairly with available resources?

## Reviewer E — Causal attribution

> Is the expected gain specifically caused by the proposed mechanism, or could generic extra capacity/pretraining/compute explain it?

A fatal objection rejects the candidate.

---

# 23. BASELINE FAIRNESS CONTRACT

For any candidate `M`, require:

```text
B0 = reproduced/corrected baseline
B1 = B0 + parameter-count control
B2 = B0 + compute-matched control
B3 = closest prior mechanism under same resource regime
M  = proposed method
```

Where relevant also require:

- same negative pool;
- same feature input;
- same text encoder;
- same augmentation;
- same update budget;
- same initialization;
- randomized-control mechanism;
- component ablations.

Do not attribute stronger backbone gains to the method.

---

# 24. EMPIRICAL PILOT BEFORE FULL METHOD IMPLEMENTATION

Before full campaign, run the cheapest experiment capable of killing the hypothesis.

A candidate survives only if:

- target mechanism improves;
- full-gallery dev ranking improves;
- matched controls do not explain the gain;
- initialization alone does not explain the gain;
- improvement is reproducible across seeds;
- no serious opposite-direction regression occurs.

Use Proposal 7's thresholds as a starting contract unless a different mechanism requires a stricter pre-registered rule.

Do not lower the threshold after seeing results.

---

# 25. KEEP A LIVE RESEARCH STATE

Create and maintain:

`docs/proposal7/AUTONOMOUS_RESEARCH_STATE.md`

It must contain:

```text
Current research cycle
Current unresolved question
Why this branch was selected
Evidence collected
Experiments run
Findings
Falsified hypotheses
Supported hypotheses
Prior-art collisions
Open research space map
Closed new branches
Next highest-information branch
Current best candidate
Reason no method has yet been selected, if applicable
```

Update it after every meaningful research cycle.

This file prevents repeated exploration.

---

# 26. EVIDENCE ARTIFACTS

Store new evidence under:

`docs/proposal7/evidence/autonomous_search/`

Include:

- JSON measurements;
- CSV summaries;
- configs;
- experiment IDs;
- checkpoint hashes;
- manifest hashes;
- bootstrap results;
- error analyses;
- code-audit notes;
- literature-collision notes.

Do not put fabricated values into artifacts.

---

# 27. DIAGNOSTIC IMPLEMENTATIONS

Use diagnostic namespaces such as:

`methods/research_probes/`

or

`methods/information_probe/`

Do NOT promote a probe into a named research method prematurely.

A diagnostic exists to test a hypothesis.

Only a surviving mechanism may later become a method.

---

# 28. DO NOT CREATE PROPOSAL 8 TOO EARLY

Do NOT create proposal8 simply because one interesting idea appears.

Proposal 8 may be created only when all are true:

```text
measured or strongly supported bottleneck
+
mechanism derived from that bottleneck
+
novelty collision passed
+
cheap falsification passed
+
matched controls passed
+
implementation feasible
```

Then create:

`docs/proposal8/SLRet_Method_From_Evidence.md`

---

# 29. REQUIRED PROPOSAL 8 OPENING

Proposal 8 must begin with:

```text
Observed bottleneck:
Measured evidence:
Why current SLRet methods do not resolve it:
Research question:
Falsifiable hypothesis:
Minimum causal intervention:
Closest prior work:
Why this is not Proposal 1–7 again:
Pilot result:
Reason to proceed:
```

If these fields cannot be filled honestly, do not write Proposal 8.

---

# 30. METHOD QUALITY STANDARD

A surviving method must satisfy:

## Evidence

It targets a failure demonstrated in the current project.

## Novelty

The mechanism survives targeted prior-art search.

## Mechanistic clarity

The path from intervention to ranking improvement is explicit.

## Falsifiability

There is a clear result that would kill the idea.

## Fairness

Gain can be separated from compute/backbone/pretraining.

## Feasibility

The required resources exist or can be obtained under project constraints.

## Retrieval relevance

The method targets standard full-gallery T2V/V2T retrieval.

## Non-duplication

It is not Proposal 1–7 with renamed components.

---

# 31. ANTI-PATTERN FILTER

Reject proposals whose main contribution is merely:

- stronger Transformer;
- extra pose;
- optical flow;
- another contrastive loss;
- another OT variant;
- another Gaussian uncertainty model;
- another hard-negative miner;
- another LLM-based augmentation;
- another graph block;
- another cross-attention block;
- multi-scale fusion;
- causal terminology without identifiable intervention;
- prototype learning;
- curriculum learning;
- mixture of experts;
- larger backbone;
- more data.

Such mechanisms are allowed only if new evidence demonstrates the specific problem they solve and prior art does not already cover it.

---

# 32. RESEARCH QUESTIONS SHOULD BECOME DEEPER, NOT JUST MORE NUMEROUS

Strong examples of research questions include:

> Does independent pair scoring fundamentally limit full-gallery ranking when the correct candidate is already represented inside the top-K neighborhood?

> Are T2V and V2T forced into an unnecessarily shared geometry despite different candidate multiplicity and relevance structure?

> Does contextual video encoding destroy relational temporal structure that remains recoverable from existing frozen features?

> Are rank-critical distinctions encoded in frozen features but suppressed by the current similarity functional?

> Does the baseline learn semantic association but fail at candidate-relative discrimination?

> Is retrieval uncertainty primarily about relevance ambiguity rather than embedding variance?

> Does sign-language simultaneity create relational information that token-wise alignment cannot represent?

> Are some apparent ranking failures caused by spoken-language underspecification rather than visual ambiguity?

These are examples only.

Do not assume they are true.

Test them.

---

# 33. RESEARCH DIVERSIFICATION RULE

After two consecutive failed cycles in the same causal layer:

MOVE TO A DIFFERENT LAYER.

Example:

```text
two visual-representation hypotheses fail
→ move to ranking / optimization / relevance / cross-modal structure.

two ranking hypotheses fail
→ move elsewhere.

two text-side hypotheses fail
→ move elsewhere.
```

Do not spend the entire search budget polishing one family.

---

# 34. USE ANALYTICAL ARGUMENTS TO KILL BAD IDEAS EARLY

Before coding expensive candidates:

- derive simplified equations;
- identify invariances;
- test toy cases;
- check whether the proposed score collapses algebraically to cosine/dot product;
- check whether the intervention changes ranking at all;
- check whether one direction cancels the new term;
- test synthetic matrices.

If an idea is mathematically equivalent to the baseline under realistic assumptions, reject it before training.

---

# 35. CODE AUDIT CAN CREATE RESEARCH QUESTIONS

Continue inspecting:

- CiCo;
- UPRet;
- SAN;
- CMCM;
- SEDS source only;
- any newly discovered official implementation.

Look for:

- assumptions embedded in scoring;
- masking behavior;
- grouping semantics;
- normalization;
- sampling;
- model-selection behavior;
- implicit symmetry;
- sequence truncation;
- feature aggregation;
- rank loss mismatch;
- unused paper components;
- implementation-paper divergence.

A code behavior may reveal a scientific gap.

But implementation bugs are not automatically research contributions.

---

# 36. DATASET FORENSICS CAN CREATE RESEARCH QUESTIONS

Continue local analysis where needed.

Potentially useful quantities:

- positive multiplicity;
- caption ambiguity;
- repeated realization structure;
- source/signer overlap;
- duration;
- temporal complexity;
- lexical rarity;
- caption compositionality;
- long-tail semantics;
- top-confuser structure;
- model-input collision;
- feature collision.

Do not construct a new benchmark.

Use findings to explain current retrieval behavior.

---

# 37. FAILURE MODE LIBRARY

Maintain a structured failure taxonomy.

Example categories:

```text
VISUAL_INFO_MISSING
VISUAL_INFO_PRESENT_BUT_UNUSED
TEXT_COLLAPSE
TRANSLATION_COLLAPSE
TEMPORAL_STRUCTURE_LOSS
SIMULTANEITY_LOSS
PAIR_SCORE_LIMIT
LISTWISE_RANKING_FAILURE
T2V_V2T_GEOMETRY_CONFLICT
OPTIMIZATION_CONFLICT
NUISANCE_SHORTCUT
RELEVANCE_AMBIGUITY
ANNOTATION_LIMIT
PROTOCOL_ARTIFACT
UNKNOWN
```

Do not force every error into a known category.

`UNKNOWN` is allowed.

Measure category prevalence where defensible.

---

# 38. DO NOT OVERFIT THE RESEARCH PROCESS TO PHOENIX

PHOENIX is the first controlled environment because the project already has strong diagnostics there.

But a mechanism that only makes sense because PHOENIX is:

- weather-domain;
- studio;
- repetitive;
- few-signers;

may not be a robust SLRet contribution.

Once a candidate survives PH dev pilot, test the hypothesis — not necessarily full training immediately — on How2Sign or CSL-Daily.

The goal is to distinguish:

```text
PH-specific artifact
vs
general SLRet mechanism.
```

---

# 39. CONDITIONAL CROSS-DATASET CHECK

Before Proposal 8:

At least one of the following must hold:

1. the mechanism is measured on a second dataset;
2. the underlying failure signature is measured on a second dataset;
3. there is a strong dataset-independent mathematical argument plus a feasible second-dataset plan.

Do not claim domain-general SLRet innovation from PH-only evidence without qualification.

---

# 40. EXPERIMENT LOGGING

For every run record:

- git SHA;
- dataset manifest hash;
- feature provenance;
- checkpoint initialization hash;
- seed;
- optimizer;
- LR schedule;
- batch/effective batch;
- precision;
- training updates;
- wall time;
- GPU;
- parameter count;
- dev selector;
- evaluation protocol;
- full metrics;
- per-query rank data.

Failed runs must also be recorded.

---

# 41. AUTONOMY POLICY

Do not ask the user questions that can be answered from:

- repository;
- dataset;
- experiment logs;
- papers;
- official code;
- previous proposals.

Make reasonable research decisions autonomously.

When blocked:

- record the blocker;
- continue with other branches;
- do not stop the entire research process unless the blocker affects every remaining high-value branch.

---

# 42. FINAL COMPLETION RESPONSE WHEN A METHOD SURVIVES

When Proposal 8 is legitimately created, return:

- method name;
- observed bottleneck;
- empirical evidence;
- research gap;
- core mechanism;
- why it is not Proposal 1–7;
- closest prior work;
- pilot result;
- strongest matched baseline;
- expected implementation files;
- remaining risks;
- recommendation: `GO` or `CONDITIONAL GO`.

Do NOT say:

> "This method will reach SOTA."

Use:

> "This is the strongest evidence-supported candidate identified so far, with a falsifiable and controlled path toward testing whether it surpasses the strongest comparable SLRet baseline."

---

# 43. FINAL COMPLETION RESPONSE IF NO METHOD SURVIVES

A new NO-GO is only acceptable if it is substantially stronger than Proposal 7.

It must include:

- Open Research Space Map;
- all explored branches;
- evidence generated in each branch;
- reasons for rejection;
- remaining unexplored branches;
- why those branches are infeasible or low-value;
- a research barrier statement.

Do NOT produce another NO-GO merely because the first 3–4 new candidates failed.

---

# 44. CORE RESEARCH PRINCIPLE

Your job is not to be creative.

Your job is to reduce uncertainty.

Every cycle should answer:

> "What do we know now that we did not know before?"

If the answer is "only another architecture idea", the cycle failed.

If the answer is:

- a bottleneck was measured;
- a hypothesis was falsified;
- a representation was shown sufficient/insufficient;
- a ranking assumption was shown limiting;
- a prior-art collision was discovered;
- a dataset property changed the interpretation;
- a new mechanism survived a controlled pilot;

then the research advanced.

Continue until the evidence supports a real method or establishes a genuine research barrier.
