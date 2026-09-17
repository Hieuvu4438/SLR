# CICO-REOPEN-01: learned sentence-conditioned clip weighting

## Material Passport

2026-09-16; academic-research-suite, inline experiment design/execution.
Registered before implementation and data execution. Exploratory efficacy screen,
not a novel method, independent review, full baseline reproduction or GO pilot.

## Authorization and selective reopening

User explicitly permits selective reopening with justification and controls.
Reopen ONLY learned query-conditioned outer clip weighting on frozen CiCo tokens.
The old inference-only hard/mean/entropy interventions (C24/C25) remain negative;
their results did not test training a sentence-conditioned outer weighting.
Generic query gates were previously conceptually closed. This exception does not
reopen ELSC lexical/support supervision, OT/coverage, pose resources or old budgets.

Observed source property: CiCo uniformly averages valid outer visual token scores.
Hypothesis: learned sentence conditioning improves their relevance weighting
enough to improve full-gallery retrieval beyond query-independent weighting.
The uniform average is verified; its harmfulness is **unverified** and is the
question tested here. Existing persistent errors motivate investigation, not proof.

## Mechanism and comparison

Keep frozen contextual tokens, cosine affinities, .07 inner softmax, legacy inner
CLS/PAD competitors, text-side channel B, logit scale and .5 channel mixture.
For channel A's per-clip soft expectations a_ijf, replace uniform valid-clip
weighting by w_ijf=softmax_f(tanh(K v_if)·tanh(Q e_j)/sqrt(32)), where e_j is the
normalized EOT vector, normalized inputs multiplied by sqrt(512) before K/Q.
K:512→32 with bias; Q:512→32 without bias. Q starts at zero. Implement the change
as scale*sum_f((w-w_uniform)*a_ijf) added to A. Initialization is exactly zero
correction; no posthoc coefficient, new loss, visual stream or encoder update.

Three trained arms, identical32800 parameters, initialization and minibatches:

1. sentence: actual EOT conditioning.
2. blind: one fixed normalized TRAIN-mean EOT for every query. This controls
   learnable visual importance; equal parameter count does not mean equal
   function-class capacity or effective query-projection rank.
3. shuffled: deterministic random TRAIN-bank EOT, indexed by a hash of the
   actual EOT bytes and seed, avoiding identical descriptors. Same rule on TRAIN
   and DEV, no other DEV query consumed, no IDs/relevance labels used for mapping.

Unchanged CiCo is B0 and every arm's initial checkpoint. Encoders are frozen for
all arms, so B0 has zero trainable parameters; do not call this end-to-end CiCo
continuation. The two trained controls match added head count/compute. Full X-Pool
is prior art, not reproduced as B3 here; a passing screen still needs that closer
prior/novelty work before any method advancement.

## Prior and candidate comparison

[X-Pool, §§4.2–4.4](https://arxiv.org/html/2203.15086v1) establishes learned
text-conditioned aggregation. This pilot retains CiCo's local matching rather
than importing the full sentence-vector/video-vector architecture. That difference
does not automatically establish novelty. UPRet's query-independent learned
visual weighting is another relevant control motivation.

Three considered implementation directions: learned outer weighting (selected:
minimal intervention); full X-Pool replacement (deferred: changes representation
and similarity together); relation-aware matching (deferred: lacks validated
relation-specific attribution and substantially expands this screen). These are
not three claimed novel methods or permission to revive all former families.

Inline pre-build perspectives, same assistant, not independent reviewers:
novelty—fails as a standalone novel contribution, acceptable prior-inspired screen;
retrieval—must improve complete519×519 DEV, not training CE;
sign-language validity—no claim one clip equals one gloss or that low weight is
linguistic irrelevance; reproducibility—existing permitted CiCo caches suffice;
attribution—blind/shuffled controls necessary, frozen-feature scope remains limited.

## Locked small run

Seed42 only, three heads×260 updates, B64. Reuse existing TRAIN-only confuser
indices:32 randomly drawn anchors plus alternating directional confusers, unique
fill to64. No new miner and no DEV confuser training. AdamW lr.001, betas(.9,.999),
eps1e-8, wd.001; cosine260, no warmup; gradient norm clipping1. FP32, one GPU.
Existing clean contextual caches: no random-swap augmentation for any arm.
This common screening deviation prevents a final baseline-fairness claim.

Evaluate steps0,130,260 on official full PH DEV; selector highest official meanR1,
then meanR5, earliest on ties; initialization eligible. Retain all checkpoints,
full score arrays, per-query metrics, loss traces, sampled batch hash, input/source
hashes and elapsed time. Do not tune on outcomes or extend horizon/seed on failure.

Frozen screen lead (not GO): sentence selected meanR1 must exceed BOTH trained
controls and B0 by≥.5pp, each direction lose≤.25pp, R5/R10 lose≤.5pp versus B0,
and persistent mean rank improve in both directions. A lead only warrants the
next pre-registered method/closest-prior test. Report distance to the stronger
77.263969% score ensemble explicitly. Full GO thresholds/3seeds/paired cluster
CI/B0–B3/second-dataset requirement are unchanged, not satisfied by this screen.

## Execution gates and monitoring

CPU tests: zero initialization, gradient reachability, invalid-slot exclusion,
block/independent-query invariance, original channel parity and conditioning
behavior. Before optimizer updates verify manifest/checkpoint identities and
fresh DEV channel parity against stored baseline (maxabs≤5e-5); zero correction
must reproduce stored mixed scores exactly. Fail closed on mismatch/nonfinite.

Command: `timeout --signal=TERM --kill-after=10s 900s /home/haipd/miniconda3/bin/python -m methods.information_probe.sentence_weight_pilot`
Working directory `/home/haipd/SLR`; no automatic retry.15minute hard cap.
Monitor process/output about every30s; report completion/failure, loss and DEV
events. No metric-based early stopping beyond locked checkpoint selection.
Outputs only in `artifacts/proposal7/CICO-REOPEN-01/` (refuse existing directory)
and `docs/proposal7/evidence/autonomous_search/CICO-REOPEN-01_run.json`.
No TEST/SEDS assets, new labels, altered positives, source data edits or uploads.

The user's autonomous build/pilot authorization supplies execution scope; the
skill's generic user-command-only confirmation does not require another approval
for this specifically disclosed small pilot. Failed experiment is retained and
not silently restarted. No optional resource change is authorized by this note.
