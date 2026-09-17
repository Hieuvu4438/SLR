# Visual radial realizability at the final LayerNorm/projection

2026-09-16, locked before numerical feasibility results. This is a mathematical
function-class check, not a reconstruction retry or a proposed method. The text
inverse's failed1% screen remains unchanged. AI-assisted ARS source verification.

For u of zero mean and ||u||²<D, any such u is realizable as LayerNorm(h) before
its affine transform: choose h=u sqrt(epsilon/(1−||u||²/D)). With native gamma,
beta and P, visual output is z=(gamma*u+beta)P. Set
B=[(diag(gamma)P)^T; ones^T/sqrt(D)]. For a target z, solve minimum-norm
B u=[z−betaP;0]. Full row rank and ||u_min||²<D imply a feasible input.
The minimum solution is orthogonal to ker(B); a nullspace component can make
two solutions have equal norm without changing output or zero mean.

Read only existing seed42 checkpoint and TRAIN cache, verify hashes. Use first
256 TRAIN rows in immutable cache order, each first real video slot(index1,
not CLS/padding). Test exactly .9 and1.1 times each stored output. No searched
direction, scale, example, fitted model, alternate checkpoint or DEV query.
Report feasibility at fixed ||u||²=D*(1−1e−5), solve residual and counts.

For row0 ONLY, if both scales feasible, construct both u with that same norm by
adding the same unit nullspace direction from a seed42 Gaussian vector. Verify
FP64 LayerNorm(eps1e−5)→native affine/projected values against targets, their
normalized directions and the1.1/.9 radius ratio, tolerance1e−9. Also report
FP16 source-LayerNorm CPU-stage residuals as approximate, not GPU parity.
Do not switch to another row on failure. Rank failure/infeasibility is a result,
not permission to change the construction or loosen tolerances.

Interpretation: success demonstrates a noninjective local map on feasible
LayerNorm inputs around these output directions. It does NOT establish that
both constructed hidden states are reachable from actual videos through the
preceding transformer, that norm distinguishes meanings, or that a retrieval
method should retain it. Hence no norm-gating/precision/geometry rescue or GO.
Runtime120s CPU2threads, no encoder, scorer, training, GPU or upstream edits.
