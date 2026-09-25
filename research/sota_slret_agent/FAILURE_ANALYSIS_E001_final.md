# Failure analysis: `runs/sota_slret/ph/E001_baseline_ep50_s42` (val, N_text=519, N_video=519)

## Aggregate (sanity evaluator, pessimistic ties)

| Dir | R@1 | R@5 | R@10 | MedR | MeanR |
|---|---:|---:|---:|---:|---:|
| T2V | 71.10 | 91.71 | 94.22 | 1.0 | 4.26 |
| V2T | 74.95 | 90.37 | 94.41 | 1.0 | 4.46 |

## Rank distribution / top-1 recoverability

- T2V: {'1': 369, '2': 55, '3-5': 52, '6-10': 13, '11-50': 19, '>50': 11}; of 150 top-1 misses, 71.3% have the positive in top-5, 80.0% in top-10.
- V2T: {'1': 389, '2': 38, '3-5': 42, '6-10': 21, '11-50': 18, '>50': 11}; of 130 top-1 misses, 61.5% have the positive in top-5, 77.7% in top-10.

## T2V errors: what the top-1 wrong candidate looks like

- lexical Jaccard(query, hardest negative): misses 0.242 vs hits 0.248
- misses where the hardest negative has the *identical* caption (unresolvable): 8 / 150
- misses where captions differ in numbers (temperatures, dates…): 20 / 150 (hits: 107 / 369)
- misses with Jaccard ≥ 0.3: 38 / 150
- misses with Jaccard ≥ 0.5: 19 / 150
- misses with Jaccard ≥ 0.7: 10 / 150

## Buckets (T2V R@1 / V2T R@1 by caption length and video length)

| bucket | n | T2V R@1 |
|---|---:|---:|
| caption tokens ≤8 | 83 | 49.4 |
| caption tokens ≤12 | 133 | 65.4 |
| caption tokens ≤16 | 134 | 78.4 |
| caption tokens ≤24 | 138 | 79.7 |
| caption tokens >24 | 31 | 83.9 |
| video frames ≤80 | 137 | 54.7 |
| video frames ≤120 | 183 | 70.5 |
| video frames ≤160 | 129 | 82.2 |
| video frames ≤220 | 57 | 84.2 |
| video frames >220 | 13 | 84.6 |

## Calibration (T2V margin = s(pos) − s(hardest neg))

- hits: median margin 5.225; misses: median margin -2.639
- fraction of misses with |margin| < 0.5 (logit units): 0.12

## Examples of T2V misses (rank 2–5, highest lexical overlap)

- r=5 | Q: Dear viewers good evening || top1: Dear viewers good evening
- r=2 | Q: Then gusts of wind are also possible || top1: Then gusts of wind are also possible
- r=4 | Q: Fog forms locally || top1: Fog forms locally
- r=2 | Q: Good evening dear spectators || top1: Good evening dear spectators
- r=4 | Q: Dear viewers good evening || top1: Dear viewers good evening
- r=3 | Q: Dear viewers good evening || top1: Dear viewers good evening
- r=2 | Q: Dear viewers good evening || top1: Dear viewers good evening
- r=2 | Q: And now I wish you a nice evening || top1: I wish you a nice evening now
- r=2 | Q: And now the weather forecast for tomorrow Saturday the twentieth February || top1: And now the weather forecast for tomorrow Saturday, July second
- r=3 | Q: Have a nice evening and do it well || top1: I wish you a nice evening and do it well
- r=4 | Q: I wish you a nice evening and do it well || top1: Have a nice evening and do it well
- r=2 | Q: I wish you a nice evening and do it well || top1: Have a nice evening and do it well
