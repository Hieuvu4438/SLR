# Negative Results (this run only)

## N1 — H2 late stream ensemble (control)
- Change: final score = S_fusion + a·S_rgb + b·S_pose, a,b ∈ {0,.25,.5,.75,1}, tuned on PH val (E001 ep20 ckpt).
- Plausible because: SEDS trains three heads but scores with fusion only.
- Result: best +0.29 PrimaryDev (a=0, b=0.25) — optimistic (tuned on the evaluation split).
- Conclusive: yes for this checkpoint (pose/RGB heads are much weaker: 58/63 vs 70 T2V).
- Do NOT try again: plain late fusion of SEDS streams as a contribution.
- Still testable: nothing.

## N2 — H1 inference-time cross-encoder re-ranking (E002)
- Change: top-K (K∈{4,8,16,32}) re-scoring with S + w·ITM logit (w∈{.25,.5,1,2,4,1e3}) from the jointly trained
  SignMatchHead (2-layer cross-attn over pose+RGB tokens), PH val, E002 best checkpoint (epoch 4).
- Plausible because: 80% of misses have the positive in top-10 (FAILURE_ANALYSIS.md).
- Result: same-checkpoint bi-encoder alone (w=0) PrimaryDev 75.14; w=1 → 73.70 (T2V fixes 9 / breaks 21);
  best w=0.25 → 75.24 (+0.10, within noise). Head train ITM accuracy 99.4% ⇒ the head overfits the 6.5k train pairs
  and its logits are not reliable on held-out queries.
- Conclusive: yes for this head size/data regime (single seed; sweep is on val so it is optimistic — still ~0).
- Do NOT try again: using this ITM head as an inference re-ranker on PH-sized data.
- Still testable: re-ranker with strong regularisation / cross-fitting; not a priority (see H1b).
