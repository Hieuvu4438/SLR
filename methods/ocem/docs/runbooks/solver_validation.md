# Solver validation runbook

This runbook validates the OCEM mathematical solver only. It does not produce
a sign-language retrieval result, reproduce CiCo, or establish the proposed
failure mechanism.

## Immutable oracle

`src/ocem/scoring/reference.py` is a byte-identical copy of the proposal
oracle. Both files have SHA-256
`5579b07c5fa6e4f4e15563d43919f6d3a7ff0336969be6c86d81812daf687773`.
Do not refactor the oracle together with the GPU solver.

## Commands

From `methods/ocem`:

```bash
PYTHONPATH=src pytest -q
ruff check src/ocem/scoring src/ocem/cli.py tests/test_geometry.py tests/test_solver.py tests/test_cli.py
PYTHONPATH=src python -m ocem solver validate \
  --device cpu --dtype float64 \
  --output runs/wp06_solver/cpu_float64_final.json
PYTHONPATH=src python -m ocem solver validate \
  --device cuda --dtype float64 \
  --output runs/wp06_solver/cuda_float64_final.json
PYTHONPATH=src python -m ocem solver validate \
  --device cuda --dtype float32 --profile --profile-pairs 64 \
  --output runs/wp06_solver/cuda_float32_p64_final.json
```

The CLI accepts only the bundled oracle path through `--reference`. Generated
reports live under ignored `runs/`; the compact immutable result is
`locks/solver_validation.json`.

## Acceptance result on 2026-09-10

- 44 tests passed, including independent constrained-primal bracketing,
  all-entry finite differences, duplicate/permutation invariance, low-kappa
  null-heavy behavior, distinct-geometry buckets, CPU/CUDA plan and gradient
  parity, and FP16-autocast isolation.
- CPU and CUDA float64 maximum oracle value error: `3.94e-11`; maximum
  all-entry gradient error: `1.86e-10`.
- CUDA float32 maximum oracle value error: `7.43e-8`; maximum all-entry
  gradient error: `4.10e-8`.
- Three deterministic CUDA float32 64-pair profiles were certified with no
  fallback. Throughput was 154.4, 251.5, and 195.2 pairs/s for `(m,M)` of
  `(8,16)`, `(16,32)`, and `(32,64)` respectively. Peak allocated CUDA memory
  was below 17 MB for the solver tensors.

This passes the synthetic S6/G4 solver gate. These timings are not an
end-to-end latency claim: preprocessing, affinity construction, real shape
distribution, score tiling, and full-gallery seconds/query must be measured
again after a dataset feature lock exists.

## Failure policy

Never return an uncertified pair. Retry through the declared ladder; an
enabled CPU reference fallback is counted explicitly. A negative gap beyond
rounding tolerance is a technical error. Do not relax the training or
evaluation certificate thresholds to rescue a run.
