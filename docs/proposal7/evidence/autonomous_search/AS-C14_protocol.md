# AS-C14 — factor the scorer's inner-mask intervention

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C14-v1

## Rationale and fixed design

AS-C01 jointly masking invalid inner-softmax positions lost1.54pp; it did not
separate video CLS, video padding and text padding. AS-C13's mean-token control
still retained original CLS/padding. Q05 remains causally under-specified at the
scorer. This eight-cell factorial ablation uses the unchanged frozen R0 seed42
dev cache, full519 gallery, unchanged outer masks and official metrics.

Three binary factors: exclude video CLS (slot0) from B's video-axis softmax;
exclude video padding (mask1 except slot0) from B; exclude text padding (mask0)
from A's text-axis softmax. Run ALL8 combinations, not a sweep-selected winner.
Video CLS is historically outer-invalid; text special tokens remain valid and
are NOT ablated. Other tokens and .07 temperature/normalization unchanged.
No train fitting, new checkpoint, inference cohort, or changed positives.

Verify default channel delta≤2e-5 and exact ranks against R0; all-excluded
channels≤2e-5 against the existing independent masked scorer. Unit tests must
verify text-mask changes cannot change B and video-mask changes cannot change A.
Report each factor cell's directional R1/R5/R10, persistent rank changes and
score effects; no p-value, dev selector or three-seed method claim. Any observed
gain is only a diagnostic lead and cannot bypass novelty or matched training.
Inference ablations are not proof that a training-consistent implementation
would improve; a bug fix or generic global/local mixture is not novel by itself.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.mask_factor_probe`
from `/home/haipd/SLR`. Hard timeout30minutes, monitor run JSON and session≤60s.
Outputs `AS-C14-MASK_run.json`, eight metric files and eight small score arrays.
Preserve failures, no overwrite of completed experiment.
