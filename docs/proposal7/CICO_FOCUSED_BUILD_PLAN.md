# CiCo-focused method development

## Material Passport

- Date: 2026-09-16
- Origin: academic-research-suite, experiment planning inline
- Status: direction updated; first candidate not selected; no training launched
- AI-assisted source reasoning; not an independently reviewed method proposal

## User-directed priority

CiCo is now the implementation baseline. The objective is to build an attributable
improvement and ultimately compare against SEDS and other comparable SLRet methods.
Repository overviews are supporting records, not the primary work product.
Use SEDS source/paper for ideas without its prohibited checkpoints, features,
teacher outputs or derived assets. Existing TEST and closure constraints remain.

## What the current source comparison establishes

CiCo `third_party/SLRT/CiCo/CLCL/modules/modeling.py:468` normalizes tokens and
computes bidirectional softmax-weighted token similarities, then validity averages.
SEDS `third_party/SEDS/modules/modeling.py:414` applies the same form separately
to pose, RGB and fused representations. Its fusion consumes pose/RGB/mask, not
the text query; the later similarity computation does consume text.
`module_fusionencoder.py:414` implements the inspected gloss-attention forward.
Supplied training scripts request gloss_atten and Pose–RGB matching.

Therefore SEDS is not evidence that CiCo simply lacks any text-dependent visual
selection. For normalized video tokens v_i and a text token t_j, CiCo's text-side
inner score can be written exactly as

`sum_i p_ij (v_i · t_j) = (sum_i p_ij v_i) · t_j`,
`p_ij = softmax_i((v_i · t_j)/.07)`.

This is query-token-conditioned pooling already. It is not equivalent to the
complete X-Pool architecture, which includes learned Q/K/V and output transforms,
normalization and a sentence-level query. It blocks the overbroad justification
that adding attention would introduce query conditioning for the first time.
Mask conventions and training text augmentation are not changed by this identity.

Primary reading:

- [SEDS](https://arxiv.org/html/2407.16394v1), architecture/method context;
  earlier activation/temporal audits retain their narrower findings.
- [X-Pool](https://arxiv.org/html/2203.15086v1), §§4.2–4.4: text-conditioned
  pooling and learned cross-modal attention. The CVF page fetch failed403;
  primary arXiv methods were retrieved. No claim of full-paper/code replication.

No published performance table was used to select a candidate. This comparison
is a design/prior-art boundary, not a new measured CiCo error explanation.
Earlier C24/C25 pooling and entropy negatives remain in force.

## Development sequence

1. Reuse the verified CiCo bridge and full-gallery evaluator; preserve original
   baseline code/checkpoints. Do not build another training framework or restart
   the failed generic readout experiments.
2. Select one precise missing capability with new evidence or strong analytical
   support. Inspect only the code and primary papers needed for that decision.
3. Specify the minimum intervention, train/inference change, internal-family
   collision and closest implemented prior. Generate/review the required distinct
   candidates only when that support exists.
4. Implement the surviving candidate in its own method namespace. Require exact
   disabled-candidate baseline parity, gradient/mask/shape checks, and a fixed
   small pilot against baseline continuation, initialization, randomized mechanism,
   capacity/compute controls and closest prior.
5. Advance only after the unchanged full-gallery DEV gates, seed controls and
   conditional second-dataset check. Passing a toy check or lowering an auxiliary
   loss does not justify full training or a paper claim.

Implementation interfaces inspected: `shared/slr_common/upstream/cico_bridge.py`
provides video/text encoding, two score channels, the mixed score and paired-score
parity path. `shared/slr_common/evaluation/cico_eval.py` is the existing evaluator
location, not a newly reviewed or modified evaluator in this turn.

## Comparison targets

- First: outperform matched CiCo continuation on official full-gallery DEV.
- Keep the existing three-model score ensemble (77.263969% PH DEV meanR1) as
  the stronger local control, with its extra inference cost explicit.
- Then: establish cross-dataset evidence and a comparable resource/protocol
  relationship to SEDS and other methods before claiming to beat them.

Use the existing ≥.5pp/≥2-of3-seed, paired-cluster interval, directional regression,
R5/R10, persistent-slice and B0–B3 requirements. No threshold changes after results.
The local ensemble value is not a current published-SOTA threshold and is not
directly comparable to an author-reported TEST number.

## Scope decision resolved: selective reopening authorized

The user explicitly permits selective reopening with justification and controls.
Each exception must be documented; historical negative results stay intact.
First registered exception: [CICO-REOPEN-01](evidence/autonomous_search/CICO-REOPEN-01_protocol.md),
a small frozen-CiCo learned sentence-conditioning screen with blind/shuffled
controls. It is prior-inspired and does not itself establish novelty. No broad
family reset, new SEDS asset permission, TEST access or lower GO threshold follows.

No candidate is named merely to satisfy the build request. The next substantive
deliverable should be an eligible minimal method specification and controlled
implementation/pilot, not another broad repository overview.
