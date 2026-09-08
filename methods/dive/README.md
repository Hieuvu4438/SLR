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
dive baseline train --config methods/dive/configs/how2sign_base.yaml --device cuda:0
dive baseline validate --config methods/dive/configs/how2sign_base.yaml --split dev
dive evidence warmup --config methods/dive/configs/how2sign_base.yaml --device cuda:0
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

For the real How2Sign/SEDS profile, validation also runs every RGB/pose pair through the pinned
native preprocessing path. It rejects nonfinite or malformed RGB tensors and RGB/pose clip-count
disagreement, records the exact selected raw pose frames and clip starts after native filtering,
binds the exact native tokenizer, and registers this lineage directory alongside the audit. The
prepared `text_model` is the exact FTfy + double-HTML-unescape + whitespace + lowercase string
seen by SEDS, while `text_original` remains preserved. Validation instruments the pinned byte-level
BPE path, proves token-ID/mask parity with the native loader, records character spans and selected
subword indices for every unit, and marks units made partial by the native uniform truncation rule.

The main How2Sign config names the intended SEDS resource paths even when the external files are
absent. The doctor command resolves null prepared-data fields only through checksummed
`prepare_data` run-state outputs, rehashes those parents, and reports concrete missing SEDS/I3D
paths. It does not discover arbitrary files or substitute the local CiCo checkpoint for SEDS.

`baseline train` reproduces the native fusion/pose/RGB/rgb-pose objective in FP32 on one GPU with
the published effective batch size and six pinned BertAdam parameter groups. Each deterministic
epoch selects one video view per unique training text, applies the vendored one-swap EDA algorithm
with a caller-owned RNG, and evaluates only the controlled dev gallery. It retains one rolling,
checksummed resume checkpoint plus the earliest best mean bidirectional R@1 checkpoint. Both input
features and native frame lineage are revalidated against the registered data audit before model
training; `--resume` rejects changed code, config, data, initialization weights, or provenance.
The configured `baseline.locked_checkpoint` is intentionally null so later stages can resolve only
the registered dev-selected `baseline_train.locked_checkpoint`.

`baseline validate` is a real controlled-dev runner, not a shape-only stub. It requires the
registered data audit and native lineage, verifies and loads the locked checkpoint, replays that
lineage exactly, encodes the full controlled dev
manifest, compares a real unpadded probe against the released score dispatcher at FP32 tolerances,
computes the complete prelogit `[video,text]` gallery with stable IDs, evaluates multi-positive
retrieval, and registers the score archive and report under the shared run state. It permits only
`--split dev`; final test access belongs to the separately locked evaluation command.

`evidence warmup` requires that controlled baseline result plus registered native frame and text
lineage. It rechecks feature hashes and exact native token IDs, caches only frozen contextual text
units, and streams native pose/RGB batches through the local encoder. Training uses deterministic
one-view-per-text ordinary batches and the registered positive/candidate relations for exactly five
FP32 epochs. Dev selection scores the complete gallery in bounded video/text blocks, so it never
materializes the full video-by-text-by-clip-by-unit interaction tensor. Epoch checkpoints restore
model, optimizer, scheduler, RNG and candidate history under `--resume`; the earliest best mean
bidirectional R@1 winner is exported and registered as an immutable FP32 reference.
