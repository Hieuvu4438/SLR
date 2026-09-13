# Method 1 PH campaign runbook

This is the execution order for reportable runs. Run commands from the repository
root with `PYTHONPATH=methods/sssc:shared`. Large artifacts remain ignored; every
stage validates the hashes produced by the previous stage.

## Frozen implementation

- Branch: `feat/method1-sssc`
- Starting implementation commit: `bf59de5`
- Semantic source-tree SHA-256:
  `ffdb3b671b0e594a756d8c364a12b98e8fb138c5d60d12317ef5c0189c8c909f`
- Host deviation is recorded per run: Python 3.13.5 / torch 2.11.0+cu128,
  rather than the target Python 3.10 / torch 2.5.1 lock.

Changing a Method 1 Python file, a pinned UPRet module, `requirements.lock`, or
the UPRet repair patch changes the semantic hash. Resume and reference-cache
validation then fail closed. Documentation-only commits do not change it.

## S1: corrected baseline per seed

The seed-42 clean run is active in tmux session
`method1_ph_s1_seed42_clean`; its log is
`artifacts/logs/method1_ph_seed42_base_clean.log`.

```bash
PYTHONPATH=methods/sssc:shared python -m method1.cli train-base \
  --config methods/sssc/configs/method1/ph_seed42_base_initial.yaml \
  --device cuda
```

Repeat only after the previous GPU job finishes for seeds 43 and 44 by changing
the config suffix. A stopped run may resume only from its own validated `last.pt`:

```bash
PYTHONPATH=methods/sssc:shared python -m method1.cli train-base \
  --config methods/sssc/configs/method1/ph_seed42_base_initial.yaml \
  --device cuda \
  --resume runs/method1/ph/base/seed42/last.pt
```

Do not use the preserved `seed42_invalid_rng_order_20260913` attempt.

After S1 releases the GPU, run the fixed real-batch overfit gate once. It is an
engineering check from permitted CLIP initialization, not a retrieval result or
an alternate checkpoint source:

```bash
PYTHONPATH=methods/sssc:shared python methods/sssc/tools/run_real_overfit_gate.py \
  --config methods/sssc/configs/method1/ph_seed42_base_initial.yaml \
  --device cuda --steps 25 --batch-size 2 \
  --output artifacts/method1/audit/ph_real_overfit_gate.json
```

## S2–S4: frozen reference, miner, diagnostics and K=1 pilot

For seed 42, after `training_complete.json` and the completed `best_dev.pt`
exist:

```bash
PYTHONPATH=methods/sssc:shared python -m method1.cli cache-reference \
  --config methods/sssc/configs/method1/ph_seed42_span_shared.yaml \
  --device cuda
PYTHONPATH=methods/sssc:shared python -m method1.cli mine-negatives \
  --config methods/sssc/configs/method1/ph_seed42_span_shared.yaml \
  --device cuda
PYTHONPATH=methods/sssc:shared python -m method1.cli audit \
  --config methods/sssc/configs/method1/ph_seed42_span_shared.yaml \
  --stage method
PYTHONPATH=methods/sssc:shared python -m method1.cli diagnose \
  --config methods/sssc/configs/method1/ph_seed42_span_shared.yaml \
  --split dev --kind support --device cuda
PYTHONPATH=methods/sssc:shared python -m method1.cli train-method \
  --config methods/sssc/configs/method1/ph_seed42_span_independent_k1_pilot.yaml \
  --device cuda --max-steps 200
PYTHONPATH=methods/sssc:shared python -m method1.cli train-method \
  --config methods/sssc/configs/method1/ph_seed42_span_shared_k1_pilot.yaml \
  --device cuda --max-steps 200
```

The K=1 outputs are engineering pilots and cannot be promoted to completed
reportable runs or teachers. Do not tune the miner after inspecting them.

## S5–S7: paired full arms

Run all arms from the same seed-specific `best_dev.pt`, cache and 20-epoch
budget. The minimum reportable order is:

```text
base_continuation
span_independent
span_shared
span_random_support
caption_hn
fsc_local
fsc_local_caption_hn
```

For each arm:

```bash
PYTHONPATH=methods/sssc:shared python -m method1.cli train-method \
  --config methods/sssc/configs/method1/ph_seed42_ARM.yaml \
  --device cuda
```

Replace `ARM` with each value above. Repeat the complete baseline/cache/arm
chain for seeds 43 and 44; never share a teacher or reference cache across
seeds. Compare the decisive pair and controls on dev:

```bash
PYTHONPATH=methods/sssc:shared python -m method1.cli compare-runs \
  --runs runs/method1/ph/span_independent/seed42 \
         runs/method1/ph/span_shared/seed42
```

The command rejects mismatched resources, protocol, implementation, seed,
query order or group identities and emits the 10,000-draw paired group-cluster
bootstrap report.

## Selection, test-once and export

Select one arm/hyperparameter policy from dev results across all three declared
seeds. Do not select the best seed. Only then create one immutable lock for each
predeclared seed replicate, evaluate its test split once, and export the same
checkpoint:

```bash
PYTHONPATH=methods/sssc:shared python -m method1.cli lock-selection \
  --config CONFIG --checkpoint RUN/best_dev.pt
PYTHONPATH=methods/sssc:shared python -m method1.cli evaluate \
  --config CONFIG --checkpoint RUN/best_dev.pt --split test --device cuda
PYTHONPATH=methods/sssc:shared python -m method1.cli export \
  --config CONFIG --checkpoint RUN/best_dev.pt --output RUN/student_inference.pt
```

The test command consumes the lock and a second supported test evaluation is
rejected. Report matched-resource conclusions separately from the broader
SEDS/C²RL system frontier in `benchmark_ledger.md`.

## Currently blocked confirmation datasets

How2Sign lacks complete CiCo paired features/dev assets. CSL-Daily lacks an
established UPRet test-feature release and a resolved dev-group provenance.
Their configs are fail-closed readiness contracts; do not substitute random
splits, copied features or a new translation to fill the result table.
