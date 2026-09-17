## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: targeted prior-art screen after a supported diagnostic
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (primary abstracts/proceedings page; not full-paper replication)
- Version Label: AS-C32-screen-v1

## Supported starting point

AS-C32 uniform3-score ensemble improves PH DEV mean R1 by2.023121pp over
the best single checkpoint; conditional95% CI[.465506,3.619048]. It is ordinary
independent-query inference with3models, unlike AS-C07 cohort assignment.
This establishes deployable cross-checkpoint complementarity, not novelty,
missing linguistic information, calibration of uncertainty, or a general
representation bottleneck. AS-C33 parameter average is a measured prior-art
control, not an invented method.

## Three distinct mechanism routes screened, none promoted

1. **Output-space consensus:** average independently computed pair scores.
   This is exactly the ordinary ensemble control measured in AS-C32. It requires
   three encoder/scorer models. It does not address88/92T and72/87V persistent
   queries having a common strictly better confuser across all three members.
   No new operator or standalone novelty claim.

2. **Parameter-space collapse:** merge model weights and execute a single
   encoder/scorer. [Model Soups, ICML2022](https://proceedings.mlr.press/v162/wortsman22a.html)
   directly establishes fine-tuned weight averaging as a way to improve results
   without multiple-model inference. [WiSE-FT, CVPR2022](https://arxiv.org/abs/2109.01903v3)
   also combines original and fine-tuned weights. Applying these operators to
   SLRet is not by itself a new mechanism. AS-C33 fixed uniform soup is therefore
   a closest-prior control, and it fails the registered ensemble-retention gate.

3. **Failure-conditioned sample-wise mixing:** use proximity to training
   failures to choose model mixture weights. [VRF, NeurIPS2024](https://arxiv.org/abs/2411.06966v1)
   explicitly constructs a zero-shot failure set and adjusts per-sample ensemble
   weights using distance to it. That mechanism already exists. This rejects
   this particular failure-bank formulation, not every conceivable adaptive
   mixture. Locally, the inadequate clean residual model and closed
   candidate-prior/correction families are additional unresolved barriers.

These are mechanism-level routes with different locations/operators, NOT three
novel surviving proposal candidates and NOT completion of the user’s GO
requirements. Do not rename an ensemble, soup, failure bank or teacher to create
an apparent new proposal. No new training is licensed by these collisions.

## Search scope and access limits

Queries: Model Soups weight averaging/inference cost; robust fine-tuning
zero-shot models and WiSE-FT. Primary proceedings page and arXiv abstracts opened
on2026-09-15; VRF found in that targeted search and its author abstract checked.
Used these sources only for their explicitly described operators. No claim of
SLRet-specific absence or full related-work coverage; no secondary summaries
used as technical evidence. No local PDF page anchors or human-read status claimed.
No numerical results from these papers treated as replication on PH.

## Next information-gain question

Before a new candidate, localize AS-C32's positive effect: does complementarity
require each video's and text's encoders to come from the SAME checkpoint, or
does it survive crossed encoder sources? A fixed3x3 crossed-encoder diagnostic
can compare all matched/crossed cells, verify the three diagonal score replays,
and separate within-pair co-adaptation from marginal encoder diversity. It is
not a learned router or a selected best cell; preregister all cells/aggregates,
resource costs and decision before measuring. A positive score average alone
does not answer this causal-location question. No AS-C34 launched here.
