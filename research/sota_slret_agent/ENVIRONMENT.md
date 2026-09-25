# Environment (recorded 2026-09-23)

| Item | Value |
|---|---|
| Host OS | Linux 5.15.0-187-generic |
| GPU | 1x NVIDIA RTX 5880 Ada Generation, 48 GiB (49140 MiB), driver 570.195.03, CUDA 12.8 (driver) |
| Other GPU users | one foreign `python3` process holding ~1.5 GiB (not touched) |
| CPU / RAM | 24 cores / 62 GiB |
| Disk | / : 3.6 TB, **79 GB free (98% used)** — keep checkpoints lean |
| Repo | /home/haipd/SLR @ `de2f07ecac9ce65563316fde0860f22e4d585acf` (branch main), dirty files are legacy docs only |
| Python env | conda `seds`: Python 3.10, torch 2.3.1+cu121, torchvision 0.18.1, numpy 2.2.6, nltk 3.10.3, textaugment, pytest |
| Base env (not used for training) | Python 3.13.5, torch 2.11.0+cu128 |

## Upstream code state
- `third_party/SEDS` = official SEDS code (commit b2a87b5 import). One local 4-line legacy patch in
  `main_task_retrieval.py::eval_epoch` (optional `geometry_windows/geometry_valid` keys) — inert unless those
  keys exist in a batch (they never do with upstream loaders). Release parity reproduced with it (see BASELINE_PARITY.md).
- `third_party/SLRT/CiCo` = official CiCo code (not modified by this run).

## Compute-relevant facts (measured)
- Upstream SEDS scripts assume 8 GPUs x 16 = global contrastive batch 128. On this single GPU, batch 128 OOMs
  with the upstream path (>48 GB). We use `methods/sota_slret/src/gradcache.py` (exact DDP-semantics emulation,
  chunk 16 = per-rank BN statistics) → 12.2 GB peak.
- Upstream `get_sign_output` Python loop replaced by an equivalent gather (`fast_seds.py`): eval outputs bitwise
  identical, grads ≤2.6e-6 rel. → 3.5x faster.
- Throughput (PH, batch 128, chunk 16): 3.05 s/step deterministic cudnn, 2.73 s/step with cudnn.benchmark
  (default in our trainer). ≈ 52 steps/epoch on the 6577-sentence train split → ≈ 2.4 min/epoch, 200 epochs ≈ 8 h.
- Dataloader: 0.058 s/batch with 20 workers (not a bottleneck).
