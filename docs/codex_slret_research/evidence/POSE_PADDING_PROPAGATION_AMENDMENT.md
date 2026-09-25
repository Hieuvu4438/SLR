# Cycle21 instrumentation amendment after first execution

2026-09-23. Initial command exited0 and found the registered downstream
signature on all four fixed inputs. Its hook captured only the body output,
not the complete concatenation requested in the protocol. The final native
pooled outputs were correctly computed from all three parts; no arithmetic,
sample, threshold or decision rule is changed.

Preserve initial `pose_padding_propagation_20260923.json` and exact initial
script `pose_padding_propagation_v1_source.txt`. Replace the hook with a transparent
wrapper around native gcn_emb to record concatenated pre-window summaries,
while retaining the body-slice parent cross-check. Schema becomes2. Run twice
with the same original command flags and timeout, changing only outputs to
`pose_padding_propagation_20260923_v2.json` and
`pose_padding_propagation_20260923_v2_replay.json`.

Require their exact replay and require every previously recorded per-sample
field to equal the original result. This amendment completes instrumentation;
it is not a threshold rescue or independent confirmation population.
