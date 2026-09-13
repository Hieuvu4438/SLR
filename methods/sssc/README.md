# Shared-Support Sign Contrast (Method 1)

This directory is the isolated implementation of proposal 5, Method 1. The normative
contract is `docs/proposal5/Method1_Implementation_Spec_2026-09-13.md` at the repository
root. The Python package is deliberately named `method1` so the required interface is:

```bash
PYTHONPATH=methods/sssc:shared python -m method1.cli --help
```

Implemented data/reference entry points are:

```bash
PYTHONPATH=methods/sssc:shared python -m method1.cli build-manifests --config CONFIG
PYTHONPATH=methods/sssc:shared python -m method1.cli cache-reference --config CONFIG --device cuda
PYTHONPATH=methods/sssc:shared python -m method1.cli mine-negatives --config CONFIG --device cuda
PYTHONPATH=methods/sssc:shared torchrun --standalone --nproc_per_node=W -m method1.cli train-base --config CONFIG
PYTHONPATH=methods/sssc:shared torchrun --standalone --nproc_per_node=W -m method1.cli train-method --config CONFIG
PYTHONPATH=methods/sssc:shared python -m method1.cli evaluate --config CONFIG --checkpoint CKPT --split dev
PYTHONPATH=methods/sssc:shared python -m method1.cli export --config CONFIG --checkpoint CKPT --output OUTPUT
```

The latter two fail closed unless the configured checkpoint is a dev-selected
`base_initial` checkpoint and every manifest/cache identity matches.

`configs/method1/ph_base_initial.yaml` is the explicit S1 configuration. A bounded
`--max-steps` run is labeled `pilot_complete`; its checkpoint is intentionally rejected as a
teacher. Only completion of the full configured epoch/step budget adds the
`training_run_complete` gate required by reference caching and Method 1 arms.

`tools/generate_ph_configs.py` deterministically materializes 24 standalone PH configs: all
eight implemented base/core/strong-control arms for seeds 42/43/44. The generated YAML files
contain no runtime inheritance; paired files differ only in seed, arm/support mode, and their
seed/arm-specific checkpoint/cache/output paths.

The implementation starts from pinned UPRet commit
`046366227417e1d8ec14145965403462df345984`. The local audit checkout is kept at
`third_party/UPRet/` and ignored by the parent repository; tracked compatibility changes are
stored as a patch below `patches/` with an attribution ledger in `docs/`.

Large manifests, caches, checkpoints, and experiment results belong below ignored
`artifacts/method1/` and `runs/method1/`. No SEDS weights, outputs, or feature caches are
allowed in this method.

## Current stage

M0 has passed, the real PH manifest has passed its complete input audit, and the corrected
baseline/model, shared-support loss, two-rank reductions, optimizer, and checkpoint contracts
have CPU tests. Reference caching, mining, complete-pool training/evaluation orchestration, and
real training remain gated. Commands without a completed correctness milestone fail closed.
