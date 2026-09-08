# DIVE-SLR v2

This directory is the canonical home of proposal 2. The implementation follows
[`docs/proposal2/DIVE_SLR_End_to_End_Implementation_Spec.md`](../../docs/proposal2/DIVE_SLR_End_to_End_Implementation_Spec.md)
and keeps method-specific code, configs, tests, and documentation separate from ELSC.

The current vertical slice provides a strict resolved config, stage-aware environment/resource
diagnostics, an ID-based full-gallery evaluator, the exact masked late-interaction scorer, centered
residual composition, and the core retrieval/local/pair losses. All fixture results are correctness
checks only and must never be reported as How2Sign reproduction or benchmark metrics.

From the repository root:

```bash
python -m pip install -e '.[dev]'
bash methods/dive/scripts/setup_seds.sh
dive doctor --config methods/dive/configs/fixture.yaml --stage fixture \
  --output artifacts/dive/doctor_fixture.json
dive smoke --config methods/dive/configs/fixture.yaml --output-dir artifacts/dive/smoke
pytest -q methods/dive/tests
```

The main How2Sign config intentionally retains unresolved SEDS resources as `null`. The doctor
command fails closed for a stage whose prerequisites are missing; it does not substitute the local
CiCo checkpoint for SEDS.
