# CiCo channel-objective identity: fixed algebra check

2026-09-16; academic-research-suite, source fact-check and numerical fixtures.
Recorded before execution. No model/data/checkpoint/DEV/TEST experiment.

Scope: at dual_mix=.5, compare the actual source CrossEn four-term mean to
bidirectional CrossEn of the mean channel logits. Verify the exact difference
as the mean row/column negative log Bhattacharyya coefficient between channel
softmax distributions. This is a known ensemble-loss structure, not novelty.

Four deterministic 4x4 fixtures: identical channels; row-constant offset;
opposite fixed perturbations around4I; opposite global offsets±1000 around4I.
FP64 CPU, no optimizer. Source class extracted by AST, avoiding module imports
and their side effects. Verify source loss against an independent logsumexp
expression and verify derivatives of the algebraic identity w.r.t. both logit
matrices. Absolute tolerance1e-10; nonnegative-gap tolerance1e-12. Equality
checks for identical/global-offset cases; row-offset equality applies only to
row normalization and must not be incorrectly applied to column normalization.

Command: `/home/haipd/miniconda3/bin/python -m methods.information_probe.channel_objective_identity`.
Stdout JSON will be retained as CICO-CHANNEL-IDENTITY.json. No performance gate,
seed sweep, candidate declaration or model-level realizability claim. A failure
blocks use of the numeric check, not permission to relax tolerance.

Decision consequence: even if verified, no joint-loss training follows without
an open mechanism, measured real-model harm and primary-source novelty distance.
Earlier C12/C18 directional gradients are not this channel-axis comparison.
