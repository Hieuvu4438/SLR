# Results (promoted / informative experiments). All numbers: PH val (519, carved from train), MEASURED LOCALLY.

## EXP E002 vs E003 — H1 → H1b (ITM auxiliary training), warm-start pilot (Stage C)

### Hypothesis
H1: cross-encoder re-ranking fixes recoverable near-miss confusions. H1b (post-hoc, from this experiment): the ITM
objective over pose/RGB clip tokens with in-batch hard negatives acts as a training signal that improves the
**bi-encoder** itself (no inference cost).

### Change
`--method cmr` (SignMatchHead 2×TransformerDecoderLayer(512), ITM BCE weight 1, hard negatives ∝ softmax of the
bi-encoder score, identical captions excluded). Both arms warm-start from E001 best (epoch 40), 10 epochs,
same schedule (warmup-cosine 10%), seed 42, batch 128 via gradcache.

### Dev results
| Arm | checkpoint | scorer | T2V R@1 | V2T R@1 | PrimaryDev |
|---|---|---|---:|---:|---:|
| E001 baseline (start point) | best ep40 | bi-enc | 71.10 | 76.11 | 73.60 |
| E003 baseline continuation | best (ep5) | bi-enc | 71.68 | 74.18 | 72.93 |
| E003 baseline continuation | last (ep9) | bi-enc | 71.68 | 73.60 | 72.64 |
| E002 cmr | best by reranked (ep4) | rerank K16 w1 | 72.64 | 74.76 | 73.70 |
| E002 cmr | same ckpt | **bi-enc (w=0)** | 74.95 | 75.34 | **75.14** |
| E002 cmr | last (ep9) | **bi-enc** | 73.60 | 76.88 | **75.24** |

Paired bootstrap (sanity evaluator, 10k resamples over sentence groups), E002-last vs E003-last, bi-encoder:
ΔT2V +1.93 [−0.77, 4.62], ΔV2T +2.70 [0.00, 5.59], **ΔPrimary +2.31 [0.29, 4.43]**, p(Δ≤0)=0.013;
improved/worsened queries T2V 32/22, V2T 34/20.

### Seed statistics
Single seed so far (seed-43 pair queued: E005/E006).

### Interpretation
The re-ranker is harmful (N2) but joint ITM training improves the retrieval embedding by ~+2.3–2.6 over the matched
continuation. Candidate mechanism: token-level cross-attention with hard negatives back-propagates fine-grained
word↔clip discrimination into the text/pose/RGB token representations that the bi-encoder scorer uses.

### Alternative explanations (to test)
- extra regularisation / noise, not ITM-specific → E008 random negatives; E007 detach (head cannot shape encoders:
  if the gain persists, it is not representation shaping).
- seed luck → E005/E006 seed 43.
- warm-start artefact (E003 loses −0.7 vs E001 from the LR restart) → E009 from scratch vs E001.

### Decision
REVISE (H1 → H1b) and REPEAT/ABLATE (queue B).
