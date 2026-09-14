# PMGR implementation status

Recorded on 2026-09-14 for `feat/method6-pmgr`. This note separates software
correctness from measured retrieval evidence.

## Completion labels

- **Software ready:** implemented and locally verified on the real CSL-Daily train/dev assets.
- **Baseline reproduced:** the validation checkpoint loads strictly and its full-gallery dev
  result is recorded below; the corrected and legacy mask policies are reported separately.
- **Research supported:** **no**. The matched Phase-B seed-0 pilot completed, but C4 improved
  the strongest equal-input control by only 0.109 mean bidirectional R@1 points, below the
  predeclared 0.5-point planning threshold. No test split was accessed.

## Work packages

| Package | State | Evidence |
|---|---|---|
| WP0 runtime | complete | pinned CiCo commit, strict checkpoint/config hashes, CPU-safe imports, runtime lock |
| WP1 data | complete | canonical indexes: train 6,598 groups/18,401 videos; dev 797/1,077; disjoint IDs and complete feature paths |
| WP2 scorer/eval | complete | unscaled mixed score, explicit masks/IDs, stable full-gallery ranks and streamed group maximum tests |
| WP3 objectives | complete | C0--C8 objective modes, FP64 oracle, population factor, permutation and autograd checks |
| WP4 direct trainer | complete | real-data FP32 CPU smoke, finite gradients, exact resume, step-level recovery and zero-difference checkpoint reload |
| WP5 replay | complete | direct/replay loss, score, parameter-gradient and next-update parity; C0 dual-channel replay |
| WP6 distributed | complete | two-worker Gloo real/synthetic smokes, variable video counts and manual SUM equivalence |
| WP7 pilot | complete: no-go | CUDA preflight and C0/C1/C2/C3/stronger-positive-population/C4 completed; report and population/weak-performance diagnostics recorded |
| WP8 full study | stopped by gate | Phase C rank grid, replication, transfer and final test were not run because Phase B did not clear its predeclared threshold |

## Measured baseline-only results

On the complete CSL-Daily dev gallery (797 text groups, 1,077 videos), the unchanged
validation-selected baseline checkpoint gives:

| Mask policy | T2V R@1 | V2T R@1 | T2V R@5 | V2T R@5 |
|---|---:|---:|---:|---:|
| corrected valid positions | 66.374 | 63.045 | 82.058 | 80.501 |
| inherited legacy unmasked | 68.758 | 65.924 | 82.183 | 81.430 |

The corrected-minus-legacy changes (-2.384 T2V R@1 and -2.878 V2T R@1 points) are a
shared scorer/protocol correction, not a PMGR gain. The completed controls therefore all use the
same corrected mask policy. No tied scores were observed in this baseline evaluation.

## Measured Phase-B validation results

Every arm started from the same validation-selected baseline weights, reset its optimizer,
used 512 effective groups for 20 continuation epochs (240 updates), selected only on the full
CSL-Daily dev gallery, and reloaded its final recovery checkpoint with zero score difference.

| Arm | Objective | Best epoch (0-based) | T2V R@1 | V2T R@1 | Mean R@1 | Loaded videos | Accelerator seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| C0 | legacy branch-balanced, one representative | 14 | 66.876 | 64.067 | 65.471 | 122,880 | 612.7 |
| C1 | mixed-score CE, one representative | 17 | 66.248 | 64.345 | 65.297 | 122,880 | 614.3 |
| C2 | all-performance uniform-positive CE | 0 | 67.001 | 63.417 | 65.209 | 342,764 | 1,442.3 |
| C3 | all-performance set-positive CE | 15 | 66.499 | 64.717 | 65.608 | 342,764 | 1,439.9 |
| C23 | C3 with population-weighted V2T | 11 | 66.499 | 64.624 | 65.562 | 342,764 | 1,439.8 |
| C4 | group-max/population CE | 11 | 66.625 | 64.810 | **65.717** | 342,764 | 1,435.7 |

C4 is 1.008 points above the unchanged corrected checkpoint, but that is not the causal PMGR
comparison: continuation, candidate exposure and objective changes are confounded there. Against
the strongest equal-input control C3, C4 gains only 0.125 T2V R@1 and 0.093 V2T R@1, or 0.109
mean points. It also loses 0.251/0.627 T2V R@5/R@10 and 0.464/0.371 V2T R@5/R@10. The isolated
population weighting control C23 is 0.046 mean R@1 points below C3.

The all-performance arms encoded 342,764 videos and evaluated 175,495,168 candidate pairs each,
versus 122,880 videos and 62,914,560 pairs for C0/C1 (2.789x exposure). This is why their gains
over one-representative arms cannot be attributed to PMGR alone.

Weak-performance diagnostics do not show a broad collapse: 181 of 1,077 V2T queries improved,
99 degraded and 797 were unchanged. Mean worst-member V2T rank improved from 12.184 to 11.402
for singleton groups and from 19.621 to 18.886 for size-two groups. However, 19 groups combined
a T2V improvement with degradation of their worst V2T member. These diagnostics do not override
the failed causal threshold.

**Decision:** `population_hypothesis_no_go`. Per Sections 21 and 23 of the specification, the
rank grid must not be used to rescue a population intervention that failed its matched positive
controls. C6--C8, additional seeds, transfer and final-test access therefore remain intentionally
unrun. This is a valid negative research result, not a software failure.

## Generated, ignored evidence

Indexes, audits, smoke checkpoints, logs, evaluation matrices and reports live under
`artifacts/` and `runs/`, both intentionally ignored by Git. The completed queue log is
`artifacts/pmgr/logs/csl_phase_b_seed0_queue.log`; the measured report is
`artifacts/pmgr/csl_phase_b_seed0_report.json`, with diagnostics under
`artifacts/pmgr/diagnostics/csl_phase_b_seed0/`. Checkpoint format v2 records the sampler
permutation/cursor, validation history, cumulative data exposure and wall time, runtime packages,
implementation diff identity, and hashes of every configured manifest/index/baseline resource.
