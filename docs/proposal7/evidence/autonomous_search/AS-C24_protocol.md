# AS-C24 — inner token-competition analytic endpoints

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C24-v1

Question: does the deployed inner soft expectation suppress rank-critical
evidence relative to its hard-max or uniform-mean endpoints, without changing
contextual representations, normalization, masks, or outer averaging? AS-C01
pooled representations instead; AS-C14 changed masks; AS-C03 changed channel mix.
No learned temperature, sweep, candidate, new loss or token-weighting novelty.

Fixed R0seed42, full519 PH dev gallery, existing frozen cache. For normalized
token dot products a, inner expectation sum(a*softmax(a/.07)) is compared to
max(a) (tau→0) and mean(a) (tau→infinity). ALL inner tokens remain competitors,
including legacy CLS/PAD; outer video/text validity masks remain unchanged.
Use both endpoints in both channels, and change A-only or B-only with the other
channel fixed soft: six fixed nonidentity variants plus exact soft replay.
Final score remains .5(A+B). No data/model/positive changes, dev fitting or test.

All variants compute from the same blockwise dot-product tensor, float32,
block64. Require soft channel error<=2e-5 and exact directional rank parity.
Unit tests check analytic limiting behavior, endpoint order(mean<=soft<=max),
legacy mask treatment and unchanged dimensions. Save per-channel score matrices,
full metrics/ranks and fixed-confuser margin decomposition. Pointwise higher
similarity does not imply improved ranking; report both gains and losses.

Each endpoint lead gate: >=+.5pp mean R1 versus soft; neither directional R1
loss>.25pp; no directional R5/R10 loss>.5pp; both persistent mean ranks improve.
This one-checkpoint screen is not a three-seed pilot or GO. All six variants
reported, no winner-based significance claim. Even a passed endpoint would
require targeted collision analysis before proposing a mechanism; ordinary
max/mean pooling and scalar temperature changes are not new methods.
If none passes, reject these endpoints, not every intermediate temperature or
the information content of features. Do not rescue with a dev-temperature sweep.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.token_competition_probe`
Cwd `/home/haipd/SLR`; hard timeout300s. Monitor process and`AS-C24-ENDPOINTS_run.json`.
Only task-specific small matrices/metrics, no large new feature cache. Preserve
numerical/parity failures separately from scientific negatives.
