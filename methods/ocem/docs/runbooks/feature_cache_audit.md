# P14T feature-cache audit

This module checks whether an existing self-extracted I3D cache can be reused
without silently losing samples or temporal support. It does not adapt I3D,
materialize the final paired shards, or issue a complete feature lock.

## Command

From `methods/ocem`:

```bash
PYTHONPATH=src python -m ocem features audit-cache \
  --dataset phoenix2014t \
  --manifest-dir runs/wp03_phoenix/manifests \
  --feature-root ../../artifacts/features_reextracted/ph_domain_agnostic \
  --temporal-root ../../artifacts/features_reextracted/ph_temporal_metadata \
  --checkpoint-sha256 6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f \
  --workers 8 \
  --output runs/wp04_phoenix/agnostic_cache_audit.json
```

The cache pickle loader allows only the NumPy globals required by the observed
protocol-5 array encoding. It rejects arbitrary pickle globals. Every expected
ID must have one feature, metadata, and support file; extras and omissions are
reported rather than ignored.

## Result on 2026-09-10

All 8,257 P14T samples passed. The audit covered 823,901 local windows and
3,376,876,288 bytes of feature pickles. Features are finite float32 arrays with
1,024 channels. Supports are half-open input-frame intervals with 16-frame
windows and stride 1; counts, FPS, source paths, and frame counts agree with
the locked protocol manifests. The compact result is
`locks/feature_cache_audit.phoenix2014t.agnostic.json`.

Raw videos were checked by manifest path, byte size, frame count, FPS, and the
SHA-256 recorded consistently in both extractor metadata files.
This audit did not reread every raw video byte.

## Why this is not a feature-lock PASS

The verified stream uses the Oxford `bsl5k` checkpoint and is domain-agnostic.
The only complete locally discovered second P14T stream was extracted with the
H2S-adapted checkpoint. It is not evidence of P14T train-only adaptation and
must not be relabeled as such.

The pinned CiCo source cannot be executed unchanged for this requirement:

- its published pseudo-label launcher is hard-coded to How2Sign and includes
  the test split;
- the P14T trainer dataset expects an untracked `train_val_info.json`, reads
  the H2S class file, and hard-codes a pseudo-video root;
- therefore the missing P14T adaptation input/provenance must be reconstructed
  explicitly from the locked train IDs before any checkpoint is trained.

No dev or test example may enter pseudo-label generation or adaptation.
