# CMCM Gaussian alignment lower-bound check

2026-09-16; academic-research-suite / deep-research, inline source verification.
This is distinct from the completed covariance square-root derivative audit.

The source GaussianAlignmentModule uses Softplus variances and epsilon1e-8 in
two denominators but not both corresponding numerators. Check its loss on equal
means and equal strictly positive variances, where exact Gaussian KL is zero.
Derive its limit along a shared variance-decreasing path before execution.

For variance v>0 and e=1e-8, the source expression reduces to
L(v)=0.5*[log(v/(v+e))+v/(v+e)-1]. It is negative and tends to minus infinity
as v tends to zero. A common Softplus bias b realizes v=softplus(b)>0 for finite b.
This is an admissible parameter path, not a claim that actual optimization finds it.

Run unchanged source classes isolated via AST, CPU FP64, B1,T1024,D8,heads2,K3.
T1024 respects the existing causal-mask slicing defect without patching it.
Zero attention linear parameters, Gaussian means and Gaussian projection weights;
set both variance biases to fixed b in [0,-10,-18,-25,-40,-80]. Zero inputs make
the attention residual exactly zero. Measure source loss, its independent scalar
formula, and summed gradient with respect to both variance-bias vectors (shared
b directional derivative). Derive the independent analytic derivative. Require
finite values and formula/gradient agreement within1e-10 absolute.

Report exact unregularized KL and a consistently variance-shifted KL reference,
both zero on this fixture. These are diagnostic references, not implemented fixes
or efficacy candidates. No arbitrary covariance fixture, retrieval scores, trained
CMCM weights, dataset/TEST/SEDS asset access, GPU, optimization or training.

If verified, record the loss-contract failure and its narrow scope. Do not infer
trained collapse, published-result invalidity, retrieval gain, causal identification
failure, or novel uncertainty regularization. The unavailable trained integration
remains a limitation; do not repeat resource searches or build a replacement.
