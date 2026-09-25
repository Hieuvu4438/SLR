# Novelty Matrix (2026-09-23)

Searches: literature agent (LITERATURE_MATRIX.md, forward citations of CiCo/SEDS/UPRet to 2026-09) + targeted
web searches "sign language retrieval cross-encoder / re-rank / video-text matching" (2026-09-23): no SLRet paper
with a cross-encoder / ITM re-ranker found. VTaMo (arXiv 2607.09126) = SLT only.

| Mechanism | Prior work | Same/different | Supervision | Train/Inference | Repr. level | Alignment | Negatives | Uncertainty | Why non-trivial | Citation |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 cross-encoder reranker over pose+RGB clip tokens | ALBEF/BLIP/VINDLU (general image/video-text ITM + top-k rerank); none in SLRet | different domain; typed articulator memory (pose vs RGB tokens); sign-visual negative ablation | captions only | train (ITM loss) + inference (top-K) | token-level | cross-attention text→clips | in-batch hard (∝ bi-encoder) / visual / textual / random | none | SLRet errors hypothesised to be recoverable fine-grained confusions; needs failure evidence + negative-type & memory ablations | Li et al. 2021 (ALBEF); Li et al. 2022 (BLIP); Cheng et al. 2023 (VINDLU) |
| SEDS bi-encoder (baseline) | SEDS | — | captions | train+inf | token (FILIP-like softmax pooling) | late interaction | in-batch | none | — | Jiang et al. 2024 |
| CiCo CLCL | CiCo | late interaction only | captions (+pseudo-labels) | train+inf | token | softmax-max pooling | in-batch | none | — | Cheng et al. 2023 |
| UPRet distributions | UPRet | probabilistic embeddings, not cross-encoding | captions | train (OT) | global dist. | OT | in-batch | yes | — | Wu et al. 2024 |
| SAN sign-aware negatives | SAN | negative mining for contrastive bi-encoder; no coarse gain | captions + trained retriever | train | global | — | visual-confusable | none | H1 uses SAN's insight only as an ablation axis for ITM negatives | Lee et al. 2026 |
| C2RL | C2RL | pretraining (content+context), dual encoders | captions | pretrain | token/global | — | in-batch | none | — | Chen et al. 2025 |
| H2 stream ensemble | common ensembling | control only | — | inference | global | — | — | — | not a contribution | — |

Reviewer-risk notes for H1:
- "Generic ALBEF module ported to sign" → must show (i) where it helps (failure categories), (ii) typed
  pose/RGB memory > fused/RGB-only memory, (iii) negative type matters (hard vs visual vs random), and
  (iv) gain > simple ensemble control and > bi-encoder of the same run (w=0).
- "More parameters" → head ≈ 2×TransformerDecoderLayer(512) ≈ 8.4 M params (vs SEDS ≈ 200 M); report.
- "More inference compute" → report K·N head passes and wall time; not transductive (per-query only).
