# DIVE-SLR v2

This directory is the canonical home of proposal 2. The implementation follows
[`docs/proposal2/DIVE_SLR_End_to_End_Implementation_Spec.md`](../../docs/proposal2/DIVE_SLR_End_to_End_Implementation_Spec.md)
and keeps method-specific code, configs, tests, and documentation separate from ELSC.

The current vertical slice provides strict configuration and resource diagnostics; versioned data,
text-unit, temporal and relation contracts; an ID-based full-gallery evaluator; exact centered
scoring and loss kernels; evidence warm-up/reference export; deterministic mining, support and batch
planning; versioned checksummed tensor caches; an exact student step; and checkpoint/resume plus
fixture smoke coverage. A checksummed run-state resolver owns the required shared and per-variant
artifact namespaces and resolves only registered parent outputs, never arbitrary files discovered
on the machine. The evaluation layer includes dev-only opportunity bounds and gamma
calibration, duplicate-query ceilings, paired bootstrap intervals and multi-seed aggregation. All
fixture results are correctness checks only and must never be reported as How2Sign reproduction or
benchmark metrics.

The SEDS integration is an adapter around the pinned upstream model, not a forked copy of its code.
It can construct the official model when the external SEDS checkpoint and CLIP initialization are
supplied; it otherwise fails closed. The checked native reproduction artifact records every flag
from the pinned How2Sign train/eval scripts, verifies their source hashes and types, and explicitly
records the missing-upstream-parser `freeze_exfusion=false` compatibility value. The fixture tests
cover score orientation, prelogit scaling, padding behavior, feature-tap layouts and three-stream
pose flow, but do not constitute released-checkpoint parity.

Controlled manifests feed the native model through `SedsManifestInputBuilder`. It executes the
verified pinned eval loader/tokenizer source in memory (so no generated files dirty the upstream
checkout), delegates pose/RGB/text transforms and collation to those source methods, restores the
upstream loader's otherwise-unused Python RNG draw, and rejects ID, feature-root, temporal-grid,
mask or tensor-layout drift. This avoids inheriting the released loader's hard-wired test split
while retaining its native preprocessing.

From the repository root:

```bash
python -m pip install -e '.[dev]'
bash methods/dive/scripts/setup_seds.sh
dive doctor --config methods/dive/configs/fixture.yaml --stage fixture \
  --output artifacts/dive/doctor_fixture.json
dive prepare-data --config methods/dive/configs/how2sign_base.yaml --workers 16
dive validate-data --config /path/to/resolved_config.yaml
dive smoke --config methods/dive/configs/fixture.yaml --output-dir artifacts/dive/smoke
pytest -q methods/dive/tests
```

`prepare-data` uses the pinned SEDS train/test identities but repairs its protocol leakage by using
the disjoint local `labels.dev.json` as dev; the upstream `test.pkl` stays reserved for final test.
It probes every video/pose pair and records exact container-time frame maps. `validate-data`
requires real train/dev/test manifests, exact per-split relevance JSONL, the
train excluded-negative JSONL, and every referenced video/pose/RGB-key/frame-map binding. It also
rejects cross-split source-video leakage and translation hash drift, then registers its audit in
the shared seed run state. Prepared paths can resolve from that state, so users do not copy paths
between commands. The fixture config intentionally does not pretend to supply these files.

The main How2Sign config names the intended SEDS resource paths even when the external files are
absent. The doctor command resolves null prepared-data fields only through checksummed
`prepare_data` run-state outputs, rehashes those parents, and reports concrete missing SEDS/I3D
paths. It does not discover arbitrary files or substitute the local CiCo checkpoint for SEDS.
