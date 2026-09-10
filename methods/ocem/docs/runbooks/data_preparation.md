# Data preparation runbook

From `methods/ocem`, prepare each dataset independently:

```bash
export OCEM_PROJECT_ROOT=/home/haipd/SLR/methods/ocem
export OCEM_DATA_ROOT=/home/shared_data/sign_language
export OCEM_RUN_ROOT=/home/haipd/SLR/methods/ocem/runs

PYTHONPATH=src python -m ocem data prepare \
  --dataset phoenix2014t \
  --protocol configs/protocols/phoenix2014t.yaml \
  --output-dir runs/wp03_phoenix/manifests \
  --workers 16 \
  --report runs/wp03_phoenix/prepare_report.json

PYTHONPATH=src python -m ocem data prepare \
  --dataset how2sign \
  --protocol configs/protocols/how2sign.yaml \
  --output-dir runs/wp03_how2sign/manifests \
  --workers 24 \
  --report runs/wp03_how2sign/prepare_report.json
```

The adapters preserve annotation order and raw caption spelling/case. They join
video files by explicit sample name, probe FPS/frame count from each MP4, retain
the source annotation interval separately, and write every failed join to both
the original manifest and an exclusion manifest. They never infer missing IDs
from row order.

## Measured result

P14T passes protocol preparation:

| Split | Annotation | Named/valid video | Excluded |
|---|---:|---:|---:|
| train | 7,096 | 7,096 | 0 |
| validation | 519 | 519 | 0 |
| test | 642 | 642 | 0 |

All included P14T clips report 25/1 FPS. The locked manifest retains the
official gallery order and has no blank translations. This proves the local
sample/protocol join, not CiCo reproduction or feature equivalence.

How2Sign remains blocked and fully accounted for:

| Split | Annotation snapshot | Named/valid video | Published CiCo count | Missing named video |
|---|---:|---:|---:|---:|
| train | 31,165 | 31,047 | 31,085 | 118 |
| validation | 1,741 | 1,739 | 1,739 | 2 |
| test | 2,357 | 2,343 | 2,348 | 14 |

The validation count matching 1,739 is not treated as protocol-equivalence
evidence: the published ID set is unavailable. H2S therefore remains
`UNVERIFIED` even where counts coincide. The existing videos also contain
multiple source frame rates (24, 24000/1001, 30, 50, and 60 FPS), which the
manifest preserves instead of forcing to the SEDS 24-FPS convention.

Generated full manifests live under ignored `runs/`; the compact lock snapshots
are committed under `locks/`.

