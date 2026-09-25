# Claim–Evidence Ledger

| Claim | Evidence | Experiment IDs | Counterevidence | Status |
|---|---|---|---|---|
| Our trainer reproduces upstream SEDS training semantics on 1 GPU | fast-gather exact; gradcache grad rel.err 0.9%, cos ≥ 0.9998 | tests/test_fast_seds.py, test_gradcache.py | fp16-level differences only | supported |
| Release SEDS checkpoints reproduce paper within 2.4 R@1 (R@5/10 within 0.5) | release parity | parity/* | R@1 systematically lower | supported |
| SEDS val errors are mostly near-misses (positive in top-10) | 80% T2V misses in top-10 | E001 | — | supported (PH) |
| Cross-encoder re-ranking improves SEDS | — | E002 | w=1 −1.44 vs same ckpt; best +0.10 | **refuted (PH, 1 seed)** |
| ITM auxiliary training improves the SEDS bi-encoder | +2.31 PrimaryDev [0.29, 4.43] vs matched continuation | E002, E003 | single seed; warm-start only | provisional |
| Gain comes from ITM shaping encoder representations | — | E007 pending | — | unsupported |
| Hard negatives are necessary | — | E008 pending | — | unsupported |
| Native-language captions + multilingual text tower help | — | E004 pending | — | unsupported |
| Method is SOTA on any benchmark | — | — | — | unsupported |
