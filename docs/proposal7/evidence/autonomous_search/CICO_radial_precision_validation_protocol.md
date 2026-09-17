# Posthoc precision validation for the radial check

2026-09-16. Declared after reading CICO-RADIAL-CONTRACT.json. The real TRAIN
mask-valid p95 error is11.770844%, while the synthetic FP64 identity passes.
The learned text projection has condition number3824.98. This follow-up tests a
possible numerical explanation, not a new method, tolerance adjustment or rescue.
The original locked1% screen remains failed.

Use64 fixed synthetic512-dimensional pre-LayerNorm rows, seed42, and the SAME
checkpoint gamma/beta/projection. Execute the actual source-extracted LayerNorm
class (no upstream module imports). Compare:

1. FP64 LayerNorm and projection: exact-reference control.
2. Source LayerNorm on FP16 inputs (internally FP32, then FP16), promote its
   output to FP64 before projection: isolates the rounded LayerNorm output.
3. Source LayerNorm on FP16 inputs and FP16 projection/matmul: CPU emulation of
   dtype stages, not a claim to reproduce the GPU kernel or real hidden states.

All use the same fixed inputs and native checkpoint coefficient values. Report
norm-reconstruction median/p95/max, affine-constraint residuals, and check the
independent identity r_hat/r=1/(1+delta), delta=(z*a-b)/b. Require the FP64
reference max error<1e-9; no pass threshold for other cases. Nonfinite values are
explicitly counted. No fitted correction, DEV/TEST, scoring, training or encoder.
These synthetic rows cannot establish that this mechanism explains every actual
token's error. Save a separate JSON; do not overwrite the original certificate.
