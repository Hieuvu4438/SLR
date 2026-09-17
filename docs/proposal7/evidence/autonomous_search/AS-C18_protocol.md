# AS-C18 — video-encoder parameter-gradient pathway audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C18-v1

Q14/Q16: do AS-C12's valid-token directional alignment and duplicate-video-row
concentration persist after the shared video encoder Jacobian? Include the
previously omitted CLS/padding paths. No optimizer step, dev/test access,
gradient surgery, changed negatives/positives, or new method.

## Fixed computation

Use R0 checkpoint/cache; first512-row batch for each AS-C12 seed42/1337/2026,
clean/deployed-augmentation seed at epoch0: six conditions, not six models.
Reuse the exact batch indices and compare directional loss/valid-interface
cosine against AS-C12. Score A clean, B clean or augmented; directional loss
V=(CE(A)+CE(B))/2,T=(CE(A.T)+CE(B.T))/2. Other modalities/logit scale fixed.

Compute unmasked video-token covectors gV,gT; then vector-Jacobian products
through ALL `core.clip.visual` parameters. Additional covectors:
`(gV+gT)/2` restricted to within-batch duplicate-input VIDEO ROWS; and restricted
to video CLS/padding slots. These are PATHWAY partitions, NOT duplicate-query
loss partitions: a video's coordinate participates in other queries' negatives.
Global duplicate membership does not replace within-batch membership.

Report parameter-space directional cosine; norms and signed projection of
duplicate/other and invalid/valid pathway components onto the total gradient;
cross-component cosine and squared norms. Squared norms are NOT additive
fractions of a shared update. Adam/optimizer preconditioning, text parameters,
learning dynamics and effects on dev performance are not measured.

## Precision and verification

Retain checkpoint-native parameter precision (many tensors FP16). Fixed
power-of-two scale4096 in backward; convert parameter gradients to CPUfloat32
and divide by4096 before accumulation; float64 scalar reductions. No tuning of
loss scale after results. Reject nonfinite gradients. Disclose precision limits.

Encoder chunks have historical shape128 (last canonical group56) to preserve
cache numerics. Selected rows may be reordered within full128 shape; incomplete
groups are padded by repeated inputs with ZERO cotangents. Handle final56-row
cache group at its original shape. Assert exact token/mask parity for every
participating video; do not relax if this fails. Encoder has per-sequence
attention/normalization, not cross-example training batch statistics, in eval.

Before six conditions, an8-active-example/128-encoder-row smoke compares direct
end-to-end backward to the detached-token chain rule. Both use scale4096;
relative L2 error≤1e-3 and cosine≥.99999. Synthetic double tests check chunk
accumulation and partition reconstruction. Recorded inactive visual parameters
must be explicit, not silently counted as measured nonzero pathways.

No gradient arrays/checkpoints persisted; only scalar results/provenance and
small parameter-name metadata. RAM accumulation avoids new multiGB disk assets.
No candidate follows from conflict/concentration alone; PMGR/RPCA remain closed.

Command `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.parameter_gradient_probe`
from `/home/haipd/SLR`. Hard timeout30min; process/progress checks≤60s. Output
`AS-C18-PARAMETER_run.json`. Preserve failures and disclose corrections; no
silent retry, overwrite, or reinterpretation of a failed parity gate.
