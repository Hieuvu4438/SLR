# Shared-Support Sign Contrast (Method 1)

This directory is the isolated implementation of proposal 5, Method 1. The normative
contract is `docs/proposal5/Method1_Implementation_Spec_2026-09-13.md` at the repository
root. The Python package is deliberately named `method1` so the required interface is:

```bash
PYTHONPATH=methods/sssc:shared python -m method1.cli --help
```

The implementation starts from pinned UPRet commit
`046366227417e1d8ec14145965403462df345984`. The local audit checkout is kept at
`third_party/UPRet/` and ignored by the parent repository; tracked compatibility changes are
stored as a patch below `patches/` with an attribution ledger in `docs/`.

Large manifests, caches, checkpoints, and experiment results belong below ignored
`artifacts/method1/` and `runs/method1/`. No SEDS weights, outputs, or feature caches are
allowed in this method.

## Current stage

M0/M1 implementation is in progress. Synthetic commands must work without CUDA or private
assets. Real training commands fail closed until the requested resource and provenance checks
pass.
