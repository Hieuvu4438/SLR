# DIVE-SLR v2

This directory is the canonical home of proposal 2. The implementation follows
[`docs/proposal2/DIVE_SLR_End_to_End_Implementation_Spec.md`](../../docs/proposal2/DIVE_SLR_End_to_End_Implementation_Spec.md)
and keeps method-specific code, configs, tests, and documentation separate from ELSC.

The current vertical slice provides strict configuration and resource diagnostics; versioned data,
text-unit, temporal and relation contracts; an ID-based full-gallery evaluator; exact centered
scoring and loss kernels; evidence warm-up/reference export; deterministic mining, support and batch
planning; versioned checksummed tensor caches; an exact student step; and checkpoint/resume plus
fixture smoke coverage. The evaluation layer includes dev-only opportunity bounds and gamma
calibration, duplicate-query ceilings, paired bootstrap intervals and multi-seed aggregation. All
fixture results are correctness checks only and must never be reported as How2Sign reproduction or
benchmark metrics.

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
