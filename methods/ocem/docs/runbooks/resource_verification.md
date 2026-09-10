# Resource verification runbook

Set explicit roots and run the verifier from `methods/ocem`:

```bash
export OCEM_PROJECT_ROOT=/home/haipd/SLR/methods/ocem
export OCEM_DATA_ROOT=/home/shared_data/sign_language
export OCEM_RUN_ROOT=/home/haipd/SLR/methods/ocem/runs
PYTHONPATH=src python -m ocem resources verify \
  --config configs/resources.yaml \
  --output locks/resource_lock.json
```

The verifier hashes a file before invoking a model loader. A hash mismatch,
size mismatch, missing required path, annotation row/header mismatch, or model
content failure is blocking. Git source verification pins the checkout commit;
local source changes are reported separately from commit identity and are not
silently accepted as upstream parity.

## Current measured state

- Oxford I3D matches the audited SHA-256 and loads as a checkpoint containing
  more than 300 state tensors, including `module.logits.conv3d.weight`.
- The canonical `artifacts/pretrained/ViT-B-32.pt` matches OpenAI's official
  SHA-256 and loads as TorchScript with the required CLIP state keys. It was
  seeded byte-for-byte from a matching local file already present in the
  workspace, avoiding a redundant download. OCEM's registry and runtime path
  do not reference the SEDS tree.
- The pinned SLRT checkout is at the required CiCo commit.
- P14T video counts and official annotation row counts match 7,096/519/642.
- H2S annotations match all three audited hashes and row counts.
- H2S raw train is blocked: it has 31,047 named `.mp4` files plus one valid MP4
  stored under a corrupt undecodable filename (`�~`). The same corrupt entry is
  present in the existing 33 GB ZIP. The verifier intentionally does not rename
  it or guess its sample ID.
- SEDS/Baidu is represented only as an excluded resource with
  `required=false`; the verifier never resolves or loads an excluded path.

`locks/resource_lock.json` verifies registered files and content only. It does
not promote dataset protocol/feature gates, baseline reproduction, or any
scientific result.

