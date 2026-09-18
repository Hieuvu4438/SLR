# Cross-model checkpoint moment exposure screen

2026-09-18, academic-research-suite inline experiment audit. CPU only,
no model execution, updates, DEV selection or TEST. Maximum300s, output<=1MiB.
This is not a new optimizer method or reopening of closed augmentation,
residual-head, gradient-conflict or UPRet transport experiments.

Question: is SEDS's measured nonzero-first/zero-second-moment phenotype also
present in actually retained CiCo PH/CSL optimizer states? Inspect those
selected checkpoints and the historical partial UPRet checkpoint as an explicit
FP32 negative comparison. Include completed SEDS native and FP32 controls.
Different optimizers (CiCo local AdamW versus SEDS/UPRet BertAdam), training
stages and datasets preclude comparing magnitudes as causal effects.

Verify checkpoint SHA against each selection/run provenance before loading.
Count moment tensors/elements, nonzero first moments, zero second moments with
nonzero first moments, nonfinite values, per stored moment dtype. Report both
all-element and nonzero-first denominators. Do not call zeros proof of numerical
underflow without matched-gradient arithmetic evidence; no gradients retained
in these snapshots. Native source uses zeros_like but local wrappers differ.

Decision gate: >1% of FP16 moment elements with nonzero first/zero second moment
on CiCo as well as SEDS, on PH and CSL, admits planning a bounded causal study;
it does not admit a claim, automatic new training or a precision sweep. Absent
exposure means do not pursue this explanation on that checkpoint. UPRet partial
FP32 state is not evidence for its default native mixed-precision training.
Recall causality on CiCo/CSL and independent confirmation remain UNKNOWN.

Only bounded CPU artifacts change; preserve every checkpoint. Runtime record
and experiment ledger will contain source identities and counts. No published
paper ranking can be revised based on this screen.
