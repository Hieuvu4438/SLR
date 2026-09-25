# Benchmark & Evaluation Contract (frozen 2026-09-23, before any method development)

Changes only on discovery of a bug; any change invalidates affected experiments (log in STATE.md).

## Datasets, splits, modalities
| | PHOENIX-2014T (primary) | CSL-Daily | How2Sign |
|---|---|---|---|
| Train (development) | official train minus held-out val (6577 sent.) | official train minus 800 held-out sentences | official train minus 1500 held-out sentences |
| Validation (all selection/analysis) | 519 sentences carved from train, seed 0 (`artifacts/sota_slret_agent/splits/ph_val_s0/split.json`) | analogous, seed 0 | analogous, seed 0 |
| Test (final, locked) | official test: 642 videos / 642 sentences | official test: 1176 videos / 798 sentences | official test (SEDS release): 2342 videos / 1964 sentences |
| Why no official dev | SEDS release ships no dev features (PH: 0 dev I3D/pose files; raw videos unavailable) | no dev features | no dev features |
| Inputs | offline I3D RGB features + RTMPose 2D keypoints (SEDS release), English captions | same | same |
| Supervision | caption text only. **No gloss labels, no extra sign datasets.** | same | same |
| Allowed external weights | OpenAI CLIP ViT-B/32 (text+transformer init), SEDS `pretrain_signbert.pth`, CiCo/SEDS I3D features (as released) | same | same |

Final model protocol: hyperparameters, epoch budget and selection rule are frozen on val; the final model is
**retrained on the full official train split** with the frozen epoch budget (last checkpoint, no test-based
selection) OR, if retraining is not feasible, the val-selected checkpoint trained on train-minus-val. The choice is
recorded in TEST_LOCK.json before test is touched. Both are more conservative than upstream (test-selected).

## Evaluation code
- Headline ("official-compatible"): `methods/sota_slret/src/evaluation.py::official_metrics` — upstream
  `metrics.py` functions + upstream −inf group-padding reshape, verbatim logic. Verified to reproduce the upstream
  logs exactly on the three release checkpoints (BASELINE_PARITY.md).
- Sanity: `methods/sota_slret/src/sanity_eval.py` — independent, deterministic, pessimistic and optimistic
  tie handling, per-query ranks. Validated by `research/sota_slret_agent/tests/test_retrieval_metrics.py` (8 tests:
  perfect, reversed, orientation, multi-positive, ties, padding, blockwise, randomized agreement).
- Similarity orientation: score matrix `S[video, text_group]`.
  - **T2V**: query = each unique text; gallery = sentence groups scored by max over the group's videos.
  - **V2T**: query = each video; gallery = unique texts.
- Score = fusion-stream `0.5·I2T + 0.5·T2I` unless a method changes the scorer (then it must be a declared
  component and ablated). **No transductive inference normalisation** (dual-softmax / QB-Norm / inverted softmax
  over the test gallery) in headline numbers; if ever reported, in a separate, clearly marked row.
- Known quirks kept for comparability: tie ordering by index; V2T MedR = lower median. Duplicate captions inside
  a test set create exact V2T ties (PH test: 12 duplicate texts; release-checkpoint V2T R@1 pessimistic 76.01 /
  official 76.3 / optimistic 78.19). Tie-aware sanity numbers are reported alongside.

## Metrics
- Report T2V and V2T separately: R@1, R@5, R@10, MedR, MeanR.
- Pre-registered development score: **PrimaryDev = mean(T2V R@1, V2T R@1)** on val (official-compatible).
- A regression > 1.0 R@1 in either direction on val blocks promotion even if PrimaryDev improves.

## Forbidden
- Reading official test annotations/features for anything but (a) release-parity of official checkpoints
  (done once, labelled) and (b) the final locked evaluation after TEST_LOCK.json.
- Mining negatives, computing statistics, thresholds or text augmentations from val or test.
- Evaluating release checkpoints on our val split (val ⊂ their training data).
