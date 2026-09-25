# Hypotheses (drafted 2026-09-23 before E001 failure analysis; evidence columns updated after)

All are tested against a matched SEDS baseline trained by our trainer on the PH val protocol.
Stage-C pilot = warm-start from E001 best checkpoint, +10 epochs fine-tuning, method vs. baseline
continuation with identical schedule/seed (cheap causal test); Stage-D = full 50-epoch from scratch.

## H1 — Cross-modal Matching Reranker (cmr)  [LEADING]
- Observed failure: (release ckpt, test, parity context only) T2V R@1 75.4 vs R@5 91.6; to be confirmed on val.
- Mechanism: SEDS scores a pair by softmax-pooled token similarity in a shared space; distinctions that
  depend on *which* words align to *which* clips (entities, numbers, regions, negation, order) are
  averaged away. A cross-encoder that conditions text tokens on pose/RGB clip tokens can resolve them.
- Intervention: 2-layer cross-attention head (text → [pose; RGB] tokens), trained jointly with ITM BCE on
  in-batch hard negatives (sampled ∝ bi-encoder softmax, identical captions excluded); inference re-ranks
  each query's bi-encoder top-K with S + w·logit.
- Where: training (extra loss) + inference (top-K rerank). Bi-encoder unchanged.
- Why not already solved: CiCo/UPRet/SEDS/C2RL/SAN are all dual-encoder scorers; none uses a cross-encoder.
- Expected effect: +≥1.5 PrimaryDev on val; concentrated on misses with the positive in top-K and high
  lexical overlap with the hardest negative. R@10 unchanged (K ≥ 10) by construction.
- Side effects: ITM gradients may perturb bi-encoder (→ ablate `detach`); inference cost K·N_query head passes.
- Falsification: rerank gain ≤ 0.5 PrimaryDev over the same checkpoint's bi-encoder scores, or no gain over
  the matched baseline continuation.
- Controls/ablations: rerank w=0 (bi-encoder of the same run); detach=True; neg ∈ {hard, random, visual, textual};
  memory ∈ {pose_rgb, fusion, rgb}; simple score-ensemble control (H2).
- Compute: pilot ~40 min; full 50-ep ~3 h. Novelty risk: medium (ITM rerank is standard in general VLP —
  ALBEF/BLIP — novelty must come from sign-specific evidence: articulator-typed memory + sign-visual negatives).

## H2 — Stream score ensemble (control, not a contribution)
- SEDS trains pose/RGB/fusion heads but scores with fusion only. Late fusion S_f + a·S_r + b·S_p tuned on val.
- Role: the "simple control" H1 must beat. Falsification: n/a (control).

## H3 — Sign-visual hard negatives for the bi-encoder (SAN-style, in our framework)
- SAN reported no coarse-benchmark gain on PH; kept only as the `neg` ablation axis of H1.

## H4 — Larger negative pool (batch 256/512 via gradcache)
- Hyper-parameter-like; tested only if H1 fails, as a representation-level lever.

## H5 — Pose-reliability-aware fusion
- Needs evidence that low-pose-confidence videos lose accuracy (Probe B). PH hand confidence is narrow
  (5–95%: 0.60–0.81) → low prior on PH; revisit on How2Sign.

## H6 — Temporal-order sensitivity (Probe C)
- If shuffling clip order barely changes val R@1, the bi-encoder is order-insensitive; a cross-encoder
  with positional clip tokens (H1) is one route to exploit order.

## H7 — Native-language captions + multilingual CLIP-aligned text tower (mtext)
- Observed failure: captions are Google MT of German/Chinese with visible noise; C2RL (native MBart) beats SEDS by
  +4.5 T2V R@1 on CSL without pose → text side plausibly a bottleneck.
- Intervention: replace CLIP text transformer with sentence-transformers/clip-ViT-B-32-multilingual-v1 (DistilBERT-
  multilingual distilled into CLIP B/32 space; token-level + per-token 768→512 projection) on official native
  captions (PH German, CSL Chinese; H2S unchanged English). max_words 48.
- 2×2 attribution: (CLIP, EN-MT) = E001; (mtext, EN-MT) = E005; (mtext, native) = E004.
- Expected: E004 − E001 ≥ +1.5 PrimaryDev on PH, larger on CSL; E005 ≈ E001 if the gain is from native text.
- Falsification: E004 ≤ E001 + 0.5.
- Comparison class: text encoder differs (135M multilingual DistilBERT vs 63M CLIP text); disclose; still no
  gloss, no extra sign data. Novelty risk: HIGH as a standalone contribution (encoder swap) → a component only.
