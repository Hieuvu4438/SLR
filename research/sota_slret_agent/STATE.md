# Current Research State (updated 2026-09-23 09:40)

## Goal
New SLRet method that beats the strongest comparable published result (class b: RGB+pose, CLIP-B/32, gloss-free:
SEDS paper PH 76.8/78.7, CSL 85.8/85.4, H2S 62.5/57.9 T2V/V2T R@1) under a clean val-selected protocol;
overall-best reference C2RL (PH 78.7/77.6, CSL 90.3/88.4, H2S 62.4/57.5; RGB-only, MBart-large text encoders).

## Trusted baselines
- SEDS release ckpts, test parity (MEASURED): PH 75.4/76.3, CSL 84.5/84.0, H2S 61.7/57.0 (paper −0.8…−2.4 R@1; R@5/10 match).
- Our single-GPU trainer = upstream semantics (fast gather exact; gradcache = DDP 8×16 emulation, tests pass).

## Current best dev result
- E002 (cmr ITM auxiliary, warm-start ws10) bi-encoder: PrimaryDev 75.24 (last) / 75.14 (best ckpt), vs matched
  continuation E003 72.64/72.93 and E001 73.60. Paired Δ +2.31 [0.29, 4.43] (1 seed).

## Leading hypothesis
- H1b: ITM with in-batch hard negatives over typed pose/RGB clip tokens, as a TRAINING-ONLY auxiliary objective,
  improves the SEDS retrieval embedding. (H1 inference re-ranking refuted: N2.)

## Evidence / counterevidence
- For: RESULTS.md E002/E003 (both directions up, 66 improved vs 42 worsened queries).
- Against / open: single seed; warm-start protocol; mechanism untested (detach/random-neg pending).

## Active experiments
- Queue C (started 12:54; A/B cancelled to put the cheap H1b gate first): E005/E006 seed-43 pair, E007 detach,
  E008 random negs, then E004 H7 mtext native (50 ep, from scratch), then E009 cmr from scratch (50 ep).
- GPU intermittently shared with foreign jobs (carve.train / propose.py) → 2–2.5x slower when present.

## Last decision
- Cycle 1 review: REVISE H1 → H1b (REVIEW_LOG.md).

## Immediate next action
- Monitor E004; when queue B pilots finish, decide H1b (REPEAT/SCALE/KILL) with seed + ablation evidence.

## Hard blockers
- None. Constraints: single GPU (≈3.3 min/epoch PH), 79 GB disk free, no raw videos (features fixed).
