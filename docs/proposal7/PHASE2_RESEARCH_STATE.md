# Phase 2 empirical research state

Origin Skill: academic-research-suite / experiment-agent. Mode: run/validate.
Origin date: 2026-09-14. Verification: UNVERIFIED (experiments in progress).
Protocol version: 1. No publication method is proposed here.

## Scope and fixed gates

Proposal 7 remains the historical checkpoint; proposals 1–6 remain closed.
Only existing PH train/dev manifests and permitted PH RGB are in scope.
No test split, SEDS assets, new pretraining, or broad literature search.
Seeds 42, 1337, 2026 are probe optimization seeds against one fixed R0:
the Proposal 7 corrected-feature baseline's seed-42 best-dev checkpoint.
This estimates readout-training variation, NOT backbone-retraining variation.
Persistent errors remain the intersection of the three historical baseline seeds.

Success requires mean bidirectional dev R@1 >= +0.5 percentage points versus
the strongest matched control, at least 2/3 seeds, source-cluster paired
bootstrap lower 95% CI > 0, neither direction losing >0.25 points, no R@5/R@10
loss >0.5 points, improved persistent ranks, learned gain beyond initialization,
and failure of invalidated/uninformative controls to explain the gain.
All four regimes use the same paired positives, full 519-item dev gallery,
fixed historical persistent population, and frozen baseline encoders.

## Cycle 1: registered before new outcomes

- P2-C1-R0: independently replay the selected checkpoint and save contextual
  video/text tokens, masks, full-gallery scores, and per-query ranks. Preserve
  upstream scoring and tie conventions; padding corrections are NOT silently
  folded into R0. Compare replay numerically with stored baseline scores.
- P2-C1-R1: train a small expressive readout of the exact frozen contextual
  features. Diagnostic only; no new visual input or encoder learning.
- P2-C1-R2: permitted RGB passed through the SAME frozen agnostic I3D already
  used by R0, retaining coarse spatial activations before global pooling.
  Controlled readout and the baseline symmetric contrastive objective.
  This is a bounded recoverability probe, not access to all possible raw cues.
- P2-C1-R3: crossed raw-input and expressive-readout probe. Empirical envelope
  only, never a formal oracle or proof of irreducibility.
- Controls: same architecture/parameter count with existing-feature additional
  processing, spatially pooled RGB, zero information, and train-only shuffled
  RGB. Report actual runtime and parameter count, not presumed compute parity.
- Training hard pairs come only from train retrieval; dev hard pairs are fixed
  from R0's strongest incorrect candidate and are never training examples.
- Cache all PH train examples; use source-independent deterministic ordering.
  Raw sampling: eight uniformly placed 16-frame windows, same baseline image
  preprocessing; 2x2 spatial cells from the last I3D feature map. No localization
  claim is made before attributable retrieval improvement.

## Experiment ledger

Status update: R0 replay and nine R1 screening runs completed; no learned gain.
Same-I3D raw extraction is running after exact batch-parity validation.
The authoritative live search and completed experiment ledger have moved to
`AUTONOMOUS_RESEARCH_STATE.md` and `evidence/autonomous_search/` under the newer
user-selected research-loop instructions. Earlier entries below are preregistration,
not claims that those measurements are still pending.
Commit, manifest/checkpoint/code hashes, commands, seeds, timing, initial/final
metrics, rank deltas, controls, confidence intervals, and failed runs will be
written under `evidence/phase2/`; tensor caches under
`artifacts/proposal7/phase2/` (separate from earlier evidence).

Current A–E classification: **E / pending measurement**, not a scientific NO-GO.
Next experiment: P2-C1-R0 replay and cache integrity checks, then crossed probes.

## Failure and decision policy

Record failures before repair; do not treat a broken or underpowered probe as
evidence for semantic irreducibility. The subsequent user-selected autonomous
research loop supersedes the original three-cycle limit; see
`AUTONOMOUS_RESEARCH_STATE.md` for the broader search and diversification rule.
Only supported A/B/C permits mechanism candidates and targeted prior-art search.
No proposal 8 before empirical localization and minimum causal falsification.
