# DIVE-SLR v2 reproducibility contract

The verified initial environment is recorded in `methods/dive/requirements.lock`. Correctness runs
use FP32, one process, explicit Python/NumPy/torch/DataLoader seeds, and a dedicated sampler RNG.
Future checkpoints must include model weights and buffers, optimizer/scheduler state, all RNG
states, sampler plan/cursor, resolved config hash, data/unit/grid/bank fingerprints, software/git
revision, and optimizer parameter-group manifest.

Evidence warm-up uses two audited semantic parameter owners (projector at `1e-4`, trainable pose at
`1e-5`). They are represented as decay/no-decay AdamW groups so biases and normalization parameters
can have zero decay without name-substring heuristics. The scheduler is fixed from planned optimizer
steps: linear warm-up for the configured fraction, then cosine decay to the configured LR fraction.
Every epoch must yield exactly its planned step count. Selection uses only full-gallery dev
`E_local` retrieval, with the earliest optimizer step winning exact endpoint ties; both the selected
reference and its source checkpoint have SHA-256 provenance.

Tensor caches use a versioned full-provenance fingerprint rather than filenames or shapes. The
artifact-specific fingerprint contracts include grid/view, tokenizer/unit mapping, reference or
student weights, preprocessing, precision and scoring policy as applicable. Writers publish each
CPU-contiguous tensor shard atomically, record its checksum and shape/dtype schema, then publish the
index last. Readers verify the namespace, complete fingerprint, checksum, shard/index schema,
ordered sample IDs, masks and timestamps before returning tensors. Train/dev/test namespaces and
requested manifest order are explicit; correctness reference caches reject non-FP32 precision.

Run `bash methods/dive/scripts/setup_seds.sh` to create or verify the ignored official SEDS
worktree. The script refuses a dirty/non-Git target and enforces detached commit
`434e3f714fcb6a7d1f4001fb9a246bbd93ec0246`.

Generated data, checkpoints, logs, caches, and bundles remain under ignored `artifacts/`, `runs/`,
and `exports/` paths. Only small configs, schemas, tests, and documentation are committed. A fixture
test pass proves numerical/software contracts only; real adapter parity and benchmark results are
reported separately and cannot be inferred from it.
