# CoSign-LI — compositional sign-window late interaction

Status: USER_DEFERRED on 2026-09-21 before the matched GPU pilot; implementation
core complete, but no efficacy result, novelty certification, or SOTA claim.
Retained as historical design only; do not launch without a new user decision.

## One-sentence lead

CoSign-LI retrieves sentences by keeping a bank of contextual sign-prior windows
through inference and letting each contextual text token select its best-supported
visual window at two representation levels, instead of compressing the video to a
single vector before cross-modal comparison.

## Error → mechanism → rank

Measured error condition: across PH DEV, CSL DEV, and How2Sign DEV, 61%, 90%, and
58% of rows respectively contain only TRAIN-seen lexical units but at least one
unseen bigram and no exact TRAIN caption. A global embedding must preserve every
content distinction in one vector, which can blur locally supported concepts.

Intervention: use the verified public SignRep 16-frame window `features` and
`latent` banks directly. A shared text encoder exposes contextual token vectors.
Each visual level and the text tokens are independently projected and normalized.
For text token `q_j` and video windows `v_i`, the level score is:

`s(Q,V) = (1 / |Q|) * sum_j max_i <q_j, v_i>`.

The final retrieval score is the sum of feature-level and latent-level scores.
Both levels receive their own in-batch sigmoid loss so neither can become an
untrained residual. Padded windows/tokens are masked exactly. The same
video-by-text matrix is ranked in both official directions.

Expected rank effect: locally supported content tokens should keep the correct
video above candidates sharing global topic/source cues, most visibly on the
predeclared compositional slice, without the coarse-R@1 regression seen in SAN.

## What is deliberately absent

- no SEDS module, checkpoint, fusion block, pose scorer, or teacher loss;
- no hard-negative mining, changed positives, duplicate equivalence, or reranker;
- no optimal transport, dustbin/null assignment, monotonicity, or order loss;
- no source/signer/query reliability gate;
- no score ensemble with the incumbent;
- no 3D/UniFormer stream in the first contrast.

## First matched experiment

Inputs: existing cached SignRep PH TRAIN512 and full DEV519 windows; one frozen
contextual text tower for both arms. Arm A averages the normalized projected
tokens at each valid SignRep level and then applies cosine retrieval. Arm B
applies late interaction to both banks. Both arms instantiate the exact same projectors, logit parameters, dual
loss, examples, batch order, batch size, optimizer, updates, DEV evaluation
points, and selection rule; the only changed operator is mean pooling versus
MeanMaxSim. A single-level ablation is permitted only after the matched pair
passes.

Primary metric: mean of official full-gallery T2V and V2T R@1. Secondary:
directional R@1/5/10, median/mean rank, runtime, score storage, exact-seen versus
compositional versus lexical-novel slices. Slices never select a checkpoint.

Progression gate for the TRAIN512 mechanism pilot:

1. late interaction improves mean R@1 over its matched pooled control by at least
   0.5 percentage points;
2. neither direction is more than 0.5 points below control;
3. score/mask/gradient tests and finite full-DEV evaluation pass;
4. gain is not restricted to exact-caption repeats or one filename source group.

If the gate fails, stop CoSign-LI without query expansion, loss-temperature,
pooling, or projection-width sweeps. If it passes, extract SignRep for full
TRAIN7096, repeat matched training with at least three fixed seeds, then validate
on CSL or How2Sign before any locked TEST evaluation.

## Alternative if the lead fails

The one retained alternative is an articulator-bank encoder inspired by SignSeek:
body, face, left hand, and right hand remain separate visual tokens until the same
late-interaction scorer. It requires public/reimplemented pretraining and a much
larger resource audit, so it is not silently combined with the first pilot.

Rejected now: VTaMo/partial-OT retrieval adaptation (prior-art and local no-go
collision), SAN-style mining (user-closed and coarse trade-off), generative
reranking (C²RL proximity and cost), and evaluation-only benchmark contribution
(useful companion, insufficient for the requested SOTA-oriented lead).

## Implementation and verification

Core: `methods/sl_mvr/late_interaction.py`.

Tests cover padding exclusion, a constructive case where mean pooling loses
composition but late interaction separates it, the matched mean-pooling control,
square in-batch sigmoid loss, finite gradients, and dataset tokenization/ngram
analysis. Seven tests pass in the current base environment. These are
mathematical/software checks, not retrieval evidence.
