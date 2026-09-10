# ADR-0001: Use measured backtracking and extend the retry ladder

- Status: Accepted
- Date: 2026-09-10
- Scope: WP-06 solver engineering; no change to the OCEM objective or gates

## Context

The implementation specification proposes initial retry checkpoints of 128,
512, and 2,048 iterations and explicitly requires benchmarking them. During
an initial conservative-step CUDA float32 profile, two deterministic
`(m=8, M=16)` random pairs reached
centered-score widths below `1e-4` but retained pre-repair capacity violations
of approximately `2.8e-5` and `3.9e-5` after 2,048 iterations. Returning those
scores as certified would violate the `1e-5` training contract.

## Decision

Add 4,096 as a final measured retry tier. Keep all certificate thresholds
unchanged. Do not treat the extra iterations as a license to return an
uncertified score. CPU reference fallback remains explicit and counted.

Use a spectral-norm initial trial step of
`1000 * epsilon / (||A||_2^2 + machine_epsilon)`. This deliberately aggressive
trial is not assumed valid: every pair halves its own step until the smooth
dual descent inequality passes. Adaptive acceleration restart remains enabled,
and primal/dual certificates—not optimizer state—decide convergence.

## Consequences

After the measured step/backtracking change, the 64-pair synthetic profiles
converge within 424 iterations without fallback. A separate low-kappa,
null-heavy regression requires 4,016 iterations, so retaining the final tier
still prevents false success or unnecessary fallback. Full-gallery latency on
locked real feature shapes must still be reported; later experiments may not
weaken certificate tolerances to improve throughput.
