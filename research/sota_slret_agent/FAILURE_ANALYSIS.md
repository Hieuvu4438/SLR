# Failure Analysis — SEDS baseline (our trainer), PHOENIX-2014T val (519 held-out train sentences)

**FINAL (E001 best = epoch 40/50):** val PrimaryDev 73.60 (official T2V R@1 71.10 / V2T 76.11; R@5 91.71/91.52;
R@10 94.22/94.41). Final raw report: `FAILURE_ANALYSIS_E001_final.md`; per-case records: `failure_cases.jsonl` (final).
Final recoverability: T2V 150 misses — 71.3% in top-5, 80.0% in top-10; V2T 130 misses — 61.5% / 77.7%.
Final taxonomy T2V: low-overlap semantic 111 (74 in top-5), partial overlap 18, identical caption 8,
greeting 7, forecast-date 4, number 2. Caption ≤8 tokens: T2V R@1 49.4.
Conclusions below (from the epoch-20 preliminary pass) are unchanged by the final checkpoint.

Preliminary source: E001 best at epoch 20 (PrimaryDev 72.64, T2V 69.94, V2T 75.34).
Tools: `methods/sota_slret/analysis/{failure_analysis,taxonomy}.py`. Raw report: `FAILURE_ANALYSIS_E001_prelim.md`.

## 1. Recoverability (MEASURED)
| Dir | R@1 | R@5 | R@10 | misses | positive in top-5 | in top-10 |
|---|---:|---:|---:|---:|---:|---:|
| T2V | 69.94 | 89.40 | 92.87 | 156 | 64.7% | 76.3% |
| V2T | 74.18* | 89.40 | 94.41 | 134 | 59.0% | 78.4% |
*sanity evaluator (pessimistic ties); official V2T 75.34.
→ Most errors are "near misses": a K=10–16 re-scoring stage can in principle recover up to ~76% of them.

## 2. Error taxonomy of top-1 misses (pre-declared keyword rules, `taxonomy.py`)
| category | T2V (in top-5) | V2T (in top-5) |
|---|---|---|
| semantic confusion, low lexical overlap (Jaccard < 0.3) | 122 (75) | 107 (61) |
| partial lexical overlap, other | 14 (9) | 8 (6) |
| identical caption (unresolvable) | 9 (7) | 12 (7) |
| formulaic greeting/closing paraphrase | 4 (4) | 4 (3) |
| forecast-intro with different date | 4 (3) | 2 (1) |
| number/entity difference | 3 (3) | 1 (1) |
- Lexical Jaccard(query, hardest negative) is the same for misses (0.239) and hits (0.247): errors are **not**
  driven by lexically near-duplicate captions. The dominant error is a **visually/semantically plausible
  in-domain confusion** (same weather vocabulary: regions, precipitation, days) with different content.
- ~6% of T2V misses are irreducible (identical captions for different videos).

## 3. Where it fails
| bucket | n | T2V R@1 |
|---|---:|---:|
| caption ≤ 8 tokens | 83 | 53.0 |
| caption 9–12 | 133 | 66.9 |
| caption > 16 | 169 | 76.1 (pooled) |
| video ≤ 80 frames | 137 | 55.5 |
| video 81–120 | 183 | 69.9 |
| video > 120 | 199 | ~80 |
→ **Short clips / short captions** are the weakest regime (≈ 25 points below long ones): little signing
evidence, generic phrasing, and fewer tokens for the token-level softmax pooling to discriminate.

## 4. Calibration
- Median margin s(pos) − s(hardest neg): hits +4.7, misses −2.7; only 8% of misses are near-ties (|m|<0.5).
  Errors are *confident*: score sharpening/temperature tricks will not fix them; a different scoring function
  (re-scoring with interaction) or better representations are required.

## 5. Streams (control H2)
- Val R@1 T2V/V2T: fusion 69.94/75.34, RGB 62.81/67.44, pose 58.19/58.57.
- Best val-tuned late ensemble fusion + a·RGB + b·pose: PrimaryDev 72.93 (+0.29, optimistic) → negligible.

## 6. Pose quality (Probe B)
- PH hand confidence is narrow (5th–95th pct 0.60–0.81); not a dominant factor on PH (bucket analysis deferred to H2S).

## 7. Other observations
- Captions are machine translations of German (CiCo protocol); noticeable translation noise
  ("overigent ex hurricane", "clearly locally locally"). The original German (`ori_text`) exists in the
  annotations → hypothesis H7 (native-language text), but it changes the text side of the comparison class.

## Failure-mode → hypothesis mapping
- Near-miss, confident, in-domain confusions (§1,2,4) → H1 cross-encoder re-scoring (target: misses in top-K).
- Short clips/captions (§3) → H1 subgroup target; also H8 (to formulate): multi-granular/local evidence.
- Translation noise (§7) → H7.
