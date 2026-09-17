# Source-led UPRet transport-gradient audit

2026-09-15; user-directed repository weakness research, academic-research-suite.
Source-only synthetic CPU check, not AS-C46 or a retrieval efficacy experiment.

At pinned committed UPRet046366227417e1d8ec14145965403462df345984,
`flip_similarity_softmax` computes a detached Sinkhorn plan, multiplies the
sample similarities by the plan, then takes row/column maxima and averages.
Test whether the backward derivative equals finite differences of the full
forward computation. An intentional surrogate derivative is not automatically
a bug; a mismatch alone does not authorize a novel method or claim harm.

Extract the committed Sinkhorn function and the exact source AST block from
`wdist = 1.0-sim` through the averaged `sim_ot`, with no upstream imports/assets.
Use sample_num2, eps.1, max_iter100 from that source. CPU FP64, seed20260915,
32 uniform random2x2 similarities in[-.5,.5]. Each is realizable as dot products
between two orthonormal video vectors and two normalized text vectors in R4.

Compare autograd with central finite differences at1e-5 and1e-6, report stability.
Also compare autograd against finite differences holding the detached plan fixed;
this is a control for transcription/finite-difference error. Independent formula
for symmetric2x2 uniform Sinkhorn plan checks whether any discrepancy merely
comes from the implementation's early-stopping criterion. Record full fixed bank,
not just the largest discrepancy; no searched counterexample or trained state.

No claimed PH failure burden, input-gradient harm, lost retrieval recall, direct
author-paper contradiction or new method. Next gate, if discrepancy exists:
inspect intended paper objective and determine whether it has a distinct open
causal consequence before any benchmark training. No generic OT, teacher,
preservation or distributional-positive rebrand; no external writes/downloads.
