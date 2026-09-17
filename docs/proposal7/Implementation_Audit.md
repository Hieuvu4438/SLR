# Prior implementation audit

Evidence date: 2026-09-14. Scope: source reading and small synthetic checks, not a new baseline training campaign. Reports/specifications/checkpoints are covered in `AUDIT_PROGRESS.md`; this file records implementation-specific conclusions. Historical benchmark values are context only and are not used to select a new mechanism.

## Reuse boundaries

[V] `methods/README.md`, `shared/README.md` and `pyproject.toml` establish method isolation. `slr_common` is method-neutral; it must not import a method. Existing ELSC modules re-export some shared helpers. A new method must live under its own `methods/<name>/` and must not revive a closed method through direct dependency on its objective.

[V] Fully read shared `upstream/cico_bridge.py`, `upstream/factory.py`, `data/cico_dataset.py`, `data/tokenize.py`, `evaluation/cico_eval.py`. The bridge maps [B,F,1024] to [B,1024,F,1], prepends an excluded visual CLS mask, and exposes raw token encodings. Both score channels are video×text matrices; their mixture precedes any group max. Factory uses weights-only load, explicit architecture keys and a limited missing-key allowlist—not blanket strict checkpoint equality. Imports use generic upstream `modules.*`; separate baseline processes avoid CiCo/UPRet module-cache collisions.

[V] Shared feature loading uses pickle and a 1024-dimension orientation heuristic. That is not safe decoding for untrusted arbitrary assets, and a square 1024×1024 array is ambiguous. Reuse only known local provenance-checked feature files, validate exact layouts and finite values; do not describe a successful import as a provenance audit.

## Evaluator separation

[V] Shared `evaluate_score_matrix` implements flat-video best-positive ranking for non-singleton relevance, NOT CiCo's max-over-existing-groups T2V protocol. Its singleton branch deliberately expands T2V ties and uses double torch argsort for V2T. The diagnostic best-positive output is optimistic at ties. It must not be relabeled as stable one-rank-per-query evaluation.

[V] PMGR `metrics.py` implements existing-group max and stable persisted-candidate-order ranks. Its `evaluate.py` encodes every group text and every performance, scores the complete rectangle, and checks expected split counts. The CLI test flag is an explicit flag, not by itself an immutable dev-selection provenance lock. OCEM's error selector uses lexical candidate-ID tie ordering, another distinguishable convention.

[M, synthetic] Added `tools/test_protocol_boundaries.py`: one distractor group with six videos pushes the positive from grouped rank 2 to flat rank 7. Thus T2V R@5 becomes 100% versus 50% on the two-query fixture, with unchanged V2T. A second fixture distinguishes tie expansion and optimistic diagnostic recall from stable ranks. No dataset efficacy claim follows.

Executed focused suite: proposal7 reconciliation/protocol tests, `tests/test_evaluation_contract.py`, PMGR metric tests, package-boundary tests: **17 passed in 0.76 s**. Ruff on proposal7 tools: passed. Existing implementations were not changed.

## Actual learning paths and attribution

| Family | Directly inspected execution evidence | Implication |
|---|---|---|
| ELSC | Entire model/adapter/local-head and all loss kernels; train forward/loss/backward/dev-selection paths; base/min configs; compatibility patch | Zero-initialized residual adapter; training-only lexical head; detached teacher supports/word targets; coarse CLCL remains. Initialization is a legitimate selected candidate, so epoch −1 is not a trained gain. Local baseline uses AdamW and cosine, not pinned BertAdam; alpha .9 means 90% agnostic. Shared scorer retains legacy inner padding behavior. |
| DIVE | Full evidence encoder, score composer and loss kernels; actual H2 base/reproduction configs | Pose/RGB local projector and reference subtraction are already closed. Four-way paired margins and sampled-contrast denominator are implemented; fixture correctness does not prove native SEDS-backed efficacy. No SEDS pretrained file loaded during this audit. |
| SSSC | Complete baseline wrapper, shared-support kernel and training-step wrapper; full UPRet repair patch; all six method ledgers/runbook | Corrected masked late interaction plus original-order PDE sampling/OT logits; new hinge sends gradient to student video only. Actual baseline is not interchangeable with the shared CiCo bridge. Source-order parity repair excludes the invalid earlier run, not proof of why all baseline errors occurred. |
| PMGR | Full model/scorer/loss/evaluator/metrics; trainer objective dispatch/update/selector; config/status | Current-batch complete groups, exact population coefficient, branch-specific C0 versus mixed-score C1, all-positive controls. Phase B failed; rank grid intentionally unrun. Same-exposure C3—not unchanged checkpoint—is causal comparator. |
| OCEM | Stage-A affinity/statistic/error-selection/controlled-bootstrap/gate path, error helpers, amendment and baseline locks | Attempt 0 failed before outcomes from double-added CLS; amendment changed shape handling only. Rerun's actual scientific gate is no-go. Adjusted CI and ≥10% coverage checks are conjunctive. Historical released-checkpoint test parity is not independent dev-selected training reproduction. |

## Resource and uncertainty boundaries

[V] PH ELSC saved scores use H2-transfer-aware features; OCEM/SSSC use locally PH-adapted aware features. SSSC's release audit reports agnostic agreement but substantial aware-stream disagreement. Its official feature archive contains test only. Source reuse does not eliminate these resource differences.

[V] PMGR unchanged CSL checkpoint corrected masks score 66.374/63.045 T2V/V2T R@1 versus legacy 68.758/65.924. These are historical dev evaluations, not the effect of population training. Both mask policies must be named in comparisons.

[U] Source-line reading plus selected tests does not certify all hardware paths, all data identities, complete pose provenance, every config variant, or published baseline reproduction. Such boundaries remain explicit. New implementation claims will need function-specific tests and provenance checks, not inference from these passing fixtures.

## Pre-search adversarial checkpoint

Verdict: **PASS to literature investigation; no candidate approved.** The aspirational target is a fair SOTA path, not guaranteed SOTA. Major risk: most intuitive local alignment, distributional, sampler, nuisance, context-preservation and extra-stream ideas are already closed. Resolution: generate RQs only from measured residuals, search by mechanism, and permit NO-GO only after both required diagnostic/search cycles. The strongest counterargument is that observed residuals are primarily resource/protocol/optimization effects; this remains an alternative to test, not grounds to invent a module.
