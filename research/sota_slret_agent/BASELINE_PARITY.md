# Baseline Parity

## 1. SEDS release-checkpoint parity (BASELINE RELEASE PARITY — official test touched once, 2026-09-23)

Command: `methods/sota_slret/scripts/release_eval.sh {ph|csl|h2s} runs/sota_slret/parity/<ds>_test`
(untouched upstream `main_task_retrieval.py --do_eval`, args copied from `scripts/eval_*.sh`, torchrun, 1 GPU).
Evaluator hash: upstream `metrics.py` (unchanged since import b2a87b5). GPU: RTX 5880 Ada. Env: conda `seds`.
Numbers re-derived from the saved raw sim matrices with our `evaluation.official_metrics` → identical.

| Dataset | Dir | Paper (SEDS Tab.1-3) R@1/R@5/R@10 | Local measured R@1/R@5/R@10/MedR | Δ R@1 | Sanity R@1 pessimistic / optimistic | runtime |
|---|---|---|---|---|---|---|
| PH-2014T (642/642) | T2V | 76.8 / 91.7 / 95.3 | 75.4 / 91.6 / 95.8 / 1 | −1.4 | 75.39 / 75.39 | 21 s |
| | V2T | 78.7 / 92.5 / 95.2 | 76.3 / 92.7 / 95.5 / 1 | −2.4 | 76.01 / 78.19 | |
| CSL-Daily (1176 vids / 798 sent.) | T2V | 85.8 / 94.4 / 95.6 | 84.5 / 94.1 / 95.5 / 1 | −1.3 | 84.46 / 84.46 | 31 s |
| | V2T | 85.4 / 93.8 / 95.8 | 84.0 / 94.0 / 95.8 / 1 | −1.4 | 84.01 / 84.01 | |
| How2Sign (2342 vids / 1964 sent.) | T2V | 62.5 / 75.1 / 80.1 | 61.7 / 75.2 / 79.6 / 1 | −0.8 | 61.71 / 61.71 | 61 s |
| | V2T | 57.9 / 70.4 / 74.9 | 57.0 / 69.9 / 75.3 / 1 | −0.9 | 56.83 / 57.51 | |

Checkpoint SHA-256: ph `6f07e5ab…f6af5`, csl `c4379c7b…70a2`, h2s `9c94d8f1…8372` (full hashes in ASSET_MANIFEST.json).

### Interpretation
- R@5/R@10 match the paper within ±0.5 everywhere; R@1 is consistently 0.8–2.4 lower. A systematic
  evaluator/data bug would be expected to move all recall levels; a different (e.g. re-trained, released
  2025-04) checkpoint vs the paper's run, or best-of-epochs-on-test selection noise, is consistent with
  the pattern. The PH V2T gap is partly tie handling: 12 duplicate captions in PH test create exact
  ties; the tie-optimistic V2T R@1 is 78.19 (vs paper 78.7).
- H2S test here has 2342 videos (paper says 2,348 after filtering) → 6 videos missing from the release.
- Bounded checks done: evaluator validated on synthetic cases; raw sims re-scored independently; path
  resolution verified (symlinks → same files); the 4-line legacy patch in upstream eval is inert.
- **Decision:** release parity accepted as "reproduced within 2.4 R@1 (R@5/10 within 0.5)". All SOTA
  comparisons will use the (higher) **paper** numbers, which is conservative for us.

## 2. Training parity (our trainer vs upstream semantics)
- `tests/test_fast_seds.py`: vectorised clip gather — eval outputs bitwise identical; train loss identical;
  grads ≤ 2.6e-6 relative. PASS.
- `tests/test_gradcache.py`: single-GPU gradcache vs full-batch reference (dropout off, BN eval):
  loss rel. diff 3e-4, global grad rel. err 0.9%, min per-param cosine 0.9998 (fp16 CLIP weights ⇒
  kernel-level differences between chunked and full-batch encoding). DDP scaling semantics (1/world for
  encoder params, full grad for logit_scale) verified. RNG-restored recompute bit-identical. PASS.
- Training-parity run on our val split: see experiments.jsonl E001 (50 ep) / E002 (200 ep) — the paper
  numbers cannot be compared directly to val numbers; parity at the test level is checked once for the
  final matched baseline under TEST_LOCK.
