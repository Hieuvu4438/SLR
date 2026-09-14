# PMGR implementation status

Recorded on 2026-09-14 for `feat/method6-pmgr`. This note separates software
correctness from measured retrieval evidence.

## Completion labels

- **Software ready:** implemented and locally verified on the real CSL-Daily train/dev assets.
- **Baseline reproduced:** the validation checkpoint loads strictly and its full-gallery dev
  result is recorded below; the corrected and legacy mask policies are reported separately.
- **Research supported:** **not established**. The matched Phase-B arms are queued, not yet
  complete, and no test split has been accessed.

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
| WP7 pilot | queued | GPU preflight and C0/C1/C2/C3/C3-population/C4 run automatically after GPU 0 is idle |
| WP8 full study | gated | C6--C8, seeds, transfer and final test remain blocked on the Phase-B stop/go result |

## Measured baseline-only results

On the complete CSL-Daily dev gallery (797 text groups, 1,077 videos), the unchanged
validation-selected baseline checkpoint gives:

| Mask policy | T2V R@1 | V2T R@1 | T2V R@5 | V2T R@5 |
|---|---:|---:|---:|---:|
| corrected valid positions | 66.374 | 63.045 | 82.058 | 80.501 |
| inherited legacy unmasked | 68.758 | 65.924 | 82.183 | 81.430 |

The corrected-minus-legacy changes (-2.384 T2V R@1 and -2.878 V2T R@1 points) are a
shared scorer/protocol correction, not a PMGR gain. The queued controls therefore all use the
same corrected mask policy. No tied scores were observed in this baseline evaluation.

## Generated, ignored evidence

Indexes, audits, smoke checkpoints, logs, evaluation matrices and reports live under
`artifacts/` and `runs/`, both intentionally ignored by Git. The queue log is
`artifacts/pmgr/logs/csl_phase_b_seed0_queue.log`; the eventual measured report is
`artifacts/pmgr/csl_phase_b_seed0_report.json`. Checkpoint format v2 records the sampler
permutation/cursor, validation history, cumulative data exposure and wall time, runtime packages,
implementation diff identity, and hashes of every configured manifest/index/baseline resource.
