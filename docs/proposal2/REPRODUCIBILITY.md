# DIVE-SLR v2 reproducibility contract

The verified initial environment is recorded in `methods/dive/requirements.lock`. Correctness runs
use FP32, one process, explicit Python/NumPy/torch/DataLoader seeds, and a dedicated sampler RNG.
Future checkpoints must include model weights and buffers, optimizer/scheduler state, all RNG
states, sampler plan/cursor, resolved config hash, data/unit/grid/bank fingerprints, software/git
revision, and optimizer parameter-group manifest.

Generated data, checkpoints, logs, caches, and bundles remain under ignored `artifacts/`, `runs/`,
and `exports/` paths. Only small configs, schemas, tests, and documentation are committed. A fixture
test pass proves numerical/software contracts only; real adapter parity and benchmark results are
reported separately and cannot be inferred from it.
