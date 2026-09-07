# ELSC for sentence-level sign-language retrieval

This repository implements the design in
[`docs/proposal1/ELSC_End_to_End_Implementation.md`](docs/proposal1/ELSC_End_to_End_Implementation.md):
a CiCo-compatible retrieval model with a shared pointwise visual adapter and train-only,
teacher-mined local lexical supervision.

The implementation intentionally separates a measured result from a proposed method. The code,
contracts, and structural tests are present; a SOTA claim is only valid after the official assets,
dev-selected baselines, ablations, and locked full-gallery test runs have been completed. Generated
reports never fill missing metrics with zero.

## Current asset status

Phoenix14T raw videos and official train/dev/test annotations already exist on this server and are
referenced directly. They are not copied into this repository. The local CiCo annotation artifact
contains English model captions; its `dev.pkl` contains 7,615 entries (train + official dev), so
`elsc.prepare` filters it by the 519 official dev IDs and records the extra-count audit.

The official CiCo PH release checkpoint is available locally at
`artifacts/pretrained/ph_sota.pth` and pinned by SHA-256 in `configs/ph_base.yaml`. It is used only
for release parity and initialization because its historical selection provenance is unknown.
After explicit user approval, the official `sign_features.zip` was downloaded and verified as
SHA-256 `9ba1956cf416df9a31ae3d1a71a3fa9a2d1e2b3724670288b608c8d4eb895c51`. It contains
Phoenix **test only**: exactly 642 float32 files for each I3D stream. Only those PH files were
extracted under `artifacts/sign_features`; unrelated CSL/H2S content was skipped. A separate local
root is used to generate Phoenix train/dev/test features from the already-local videos. The
domain-agnostic BSL5K checkpoint reproduces the release extraction closely; the downloadable
domain-aware checkpoint targets How2Sign and is recorded explicitly as a transfer stream rather
than misidentified as the unavailable Phoenix target encoder.

## Reproducible setup

The checked upstream source is CiCo from SLRT commit
`38a4f7b00da7a858d59b7fabe5093876a84db8e0`. Recreate its small sparse checkout with:

```bash
bash scripts/setup_upstream.sh
python -m pip install -e '.[dev]'
pytest -q
```

The current runtime is captured in `environment.lock.yaml`. It is a PyTorch 2.x compatibility
environment, not a claim that the historical PyTorch 1.7 runtime is identical. Upstream source is
not vendored into the main Git repository; its commit is audited at runtime. CiCo code remains under
the terms/provenance of the upstream SLRT repository.

## End-to-end commands

Feature extraction is resumable and can wait in the background until the shared GPU has enough
free VRAM; training and parity checks still require an idle device:

```bash
tmux new-session -d -s elsc_ph_i3d 'bash scripts/run_ph_i3d_extraction.sh'
tail -f artifacts/logs/ph_i3d_extraction.log
```

After all-split local extraction completes (do not mix official test features into this run):

```bash
python -m elsc.release_eval --config configs/ph_release.yaml
python -m elsc.audit --config configs/ph_base.yaml --stage assets
python -m elsc.audit --config configs/ph_base.yaml --stage checkpoint
python -m elsc.prepare --config configs/ph_base.yaml --splits train dev test
python -m elsc.audit --config configs/ph_base.yaml --stage parity
python -m elsc.training_preflight --config configs/ph_base.yaml \
  --batch-size 512 --min-free-after-gib 4 \
  --output artifacts/preflight/ph_base_b512.json
python -m elsc.train --config configs/ph_base.yaml --run-dir runs/ph_base_s42

# Bind Min to the dev-selected baseline teacher/student initialization.
python -m elsc.configure_stage --template configs/ph_min.yaml \
  --teacher-run runs/ph_base_s42 --cache-path artifacts/cache/ph_min_s42_v1 \
  --output artifacts/campaign/ph_min_s42.yaml
python -m elsc.audit --config artifacts/campaign/ph_min_s42.yaml --stage teacher
python -m elsc.mining.build_cache --config artifacts/campaign/ph_min_s42.yaml --split train
python -m elsc.train --config artifacts/campaign/ph_min_s42.yaml --run-dir runs/ph_min_s42
python -m elsc.train --config configs/ablation_caption.yaml --run-dir runs/ph_caption_s42
python -m elsc.train --config configs/ablation_random_span.yaml --run-dir runs/ph_random_span_s42

# Test is a separate, locked action after dev selection.
python -m elsc.evaluate --run-dir runs/ph_min_s42 --split test --checkpoint best_dev
python -m elsc.export --run-dir runs/ph_min_s42 --checkpoint best_dev --output exports/ph_min

# Aggregate only measured, ID-paired runs; bootstrap resamples video groups.
python -m elsc.report --baseline-runs runs/ph_base_s42 --method-runs runs/ph_min_s42 \
  --split dev --output artifacts/reports/ph_min_vs_base_dev.json
python -m elsc.gate --report artifacts/reports/ph_min_vs_base_dev.json \
  --output artifacts/reports/ph_min_vs_base_dev_gate_g.json
```

The checked-in PH batch size is 512. It was accepted only after a real BF16
optimizer-step preflight on the RTX 5880 Ada measured 44.73 GB peak reserved
memory and 5.70 GB free memory after the step. Do not run another GPU workload
alongside this training configuration.

After starting the baseline, `scripts/run_ph_b512_followup.sh` can wait for its
validated dev-selected checkpoint, build a run-isolated train-only cache, train
ELSC-Min, and emit the paired dev report. It times out instead of retrying a
failed experiment and never accesses the test split.

`ph_full.yaml` additionally requires verified receptive-field metadata and a selected ELSC-Min
checkpoint. Resolve it with the baseline kept as teacher and Min used only for student
initialization:

```bash
python -m elsc.configure_stage --template configs/ph_full.yaml \
  --teacher-run runs/ph_base_s42 --student-run runs/ph_min_s42 \
  --cache-path artifacts/cache/ph_full_s42_v1 \
  --output artifacts/campaign/ph_full_s42.yaml
python -m elsc.train --config artifacts/campaign/ph_full_s42.yaml \
  --run-dir runs/ph_full_s42
```

Full fails rather than inferring receptive fields from sequence length.

Transfer datasets can be audited without downloading data, extracting features, or reading test
content into a tuning decision:

```bash
python -m elsc.transfer_audit \
  --dataset how2sign \
  --root /home/shared_data/sign_language/How2Sign \
  --auxiliary-label-root /home/dongvk/datasets/How2Sign/from_uni_sign_source \
  --subset-root /home/shared_data/sign_language/How2Sign/train/subset_2000 \
  --output artifacts/transfer/how2sign_asset_audit.json

python -m elsc.transfer_audit \
  --dataset csl_daily \
  --root /home/dongvk/datasets/CSL_Daily_Sentence_Crop \
  --output artifacts/transfer/csl_daily_asset_audit.json
```

The current local audit marks CSL-Daily ready for feature extraction. How2Sign remains blocked:
its annotations reference 118 train, 2 validation, and 6 test clips absent from both the extracted
directories and the local train/test ZIP inventories. The directory named `subset_2000` contains an
internally consistent 100-clip train-only engineering subset; it is never labeled as Gate X or an
official benchmark evaluation.

## What is enforced

- canonical `[B,F,1024]` features, dense indices, `True=valid`, and explicit CiCo CLS/padding masks;
- exact CiCo BPE token identity including its `linspace` long-caption subsampling;
- teacher frozen in eval mode while student gradients pass through the frozen transformer;
- train-only cache with teacher/manifest/tokenizer/feature-fusion hashes;
- generated feature SHA/checkpoint/recipe sidecars and verified receptive-field metadata before
  manifest creation;
- a real train-only optimizer-step GPU preflight before committing a long-run batch size;
- exact balanced/depart CLCL losses, lexical log-sum-exp margin, Huber evidence loss, and KL direction;
- FP32/FP16/BF16 training with GradScaler state, accumulation-aware clipping and exact resume;
- full `video × text` score orientation, multi-positive ID mappings, exact CiCo tie behavior,
  per-query ranks, and tie statistics;
- dev-only checkpoint selection and a hash-locked config/dev/checkpoint contract before test;
- inference export containing core + adapter, with local head/cache/teacher removed and score parity checked.

See `artifacts/asset_audit.json` for the current machine-readable readiness report.
See `docs/proposal1/IMPLEMENTATION_STATUS.md` for the requirement-by-requirement completion audit.
