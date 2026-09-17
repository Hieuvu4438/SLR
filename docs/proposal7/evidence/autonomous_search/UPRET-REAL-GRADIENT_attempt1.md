# Failed engineering attempt 1

2026-09-15, execution session64008, terminal exit1. Command and data selection
as in UPRET_real_gradient_protocol.md. Model built and calibration batch returned
c=.2279624044895172 at CLIP initialization. No screen metrics, historical retrieval
checkpoint loads, parameter updates or TEST access occurred before failure.
Attempt codeSHA6c29ef602569784d092655a2e0391dd0ea29ed3d389dc5d13e7b2d1acbd53fef.

Failure: torch.autograd.grad of the three chosen parity parameters raised
`RuntimeError: One of the differentiated Tensors appears to not have been used
in the graph.` This is a probe assumption failure, not a UPRet training result.

Recovery under autonomous-loop §41: allow unused gradients for both reference
and candidate, require identical None/non-None patterns, and compare every used
chosen gradient. Independently require nonzero full visual/text encoder gradients.
Add visual.proj to ensure a used visual-encoder parameter is checked as well.
Keep sample bank, calibration rule, arms and sensitivity threshold unchanged.
Announced to the user before rerun; no result-based tuning.
