# Failure analysis: `/tmp/claude-1007/-home-haipd-SLR/b3c87030-4695-430a-8f5a-b899c4a8d757/scratchpad/e001ep17` (val, N_text=519, N_video=519)

## Aggregate (sanity evaluator, pessimistic ties)

| Dir | R@1 | R@5 | R@10 | MedR | MeanR |
|---|---:|---:|---:|---:|---:|
| T2V | 69.94 | 89.40 | 92.87 | 1.0 | 5.35 |
| V2T | 74.18 | 89.40 | 94.41 | 1.0 | 5.00 |

## Rank distribution / top-1 recoverability

- T2V: {'1': 363, '2': 54, '3-5': 47, '6-10': 18, '11-50': 26, '>50': 11}; of 156 top-1 misses, 64.7% have the positive in top-5, 76.3% in top-10.
- V2T: {'1': 385, '2': 47, '3-5': 32, '6-10': 26, '11-50': 20, '>50': 9}; of 134 top-1 misses, 59.0% have the positive in top-5, 78.4% in top-10.

## T2V errors: what the top-1 wrong candidate looks like

- lexical Jaccard(query, hardest negative): misses 0.239 vs hits 0.247
- misses where the hardest negative has the *identical* caption (unresolvable): 9 / 156
- misses where captions differ in numbers (temperatures, dates…): 25 / 156 (hits: 105 / 363)
- misses with Jaccard ≥ 0.3: 34 / 156
- misses with Jaccard ≥ 0.5: 17 / 156
- misses with Jaccard ≥ 0.7: 11 / 156

## Buckets (T2V R@1 / V2T R@1 by caption length and video length)

| bucket | n | T2V R@1 |
|---|---:|---:|
| caption tokens ≤8 | 83 | 53.0 |
| caption tokens ≤12 | 133 | 66.9 |
| caption tokens ≤16 | 134 | 75.4 |
| caption tokens ≤24 | 138 | 74.6 |
| caption tokens >24 | 31 | 83.9 |
| video frames ≤80 | 137 | 55.5 |
| video frames ≤120 | 183 | 69.9 |
| video frames ≤160 | 129 | 79.8 |
| video frames ≤220 | 57 | 80.7 |
| video frames >220 | 13 | 76.9 |

## Calibration (T2V margin = s(pos) − s(hardest neg))

- hits: median margin 4.736; misses: median margin -2.747
- fraction of misses with |margin| < 0.5 (logit units): 0.08

## Examples of T2V misses (rank 2–5, highest lexical overlap)

- r=3 | Q: Have a nice evening and do it well || top1: Have a nice evening and do it well
- r=5 | Q: Then gusts of wind are also possible || top1: Then gusts of wind are also possible
- r=2 | Q: Good evening dear spectators || top1: Good evening dear spectators
- r=5 | Q: Dear viewers good evening || top1: Dear viewers good evening
- r=4 | Q: Dear viewers good evening || top1: Dear viewers good evening
- r=2 | Q: Dear viewers good evening || top1: Dear viewers good evening
- r=3 | Q: Dear viewers good evening || top1: Dear viewers good evening
- r=3 | Q: And now I wish you a nice evening || top1: I wish you a nice evening now
- r=2 | Q: And now the weather forecast for tomorrow Tuesday, twenty -four May || top1: And now the weather forecast for tomorrow Tuesday the twenty -one December
- r=3 | Q: And now the weather forecast for tomorrow Saturday the twentieth February || top1: And now the weather forecast for tomorrow Saturday, July second
- r=3 | Q: I wish you a nice evening and do it well || top1: Have a nice evening and do it well
- r=2 | Q: I wish you a nice evening and do it well || top1: Have a nice evening and do it well
