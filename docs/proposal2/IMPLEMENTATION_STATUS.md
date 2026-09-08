# DIVE-SLR v2 implementation status

Updated: 2026-09-08. Status describes repository evidence, not intended work. Synthetic fixtures
are never counted as a benchmark replication.

| Milestone | Status | Evidence / blocker |
|---|---|---|
| M00 — inventory | passed | `dive doctor` records Python/PyTorch/CUDA and stage resources. RTX 5880 Ada (49,140 MiB), driver 570.195.03, PyTorch 2.11.0+cu128/CUDA 12.8 were detected. How2Sign raw video/pose roots exist. Pinned SEDS source, reproduction config, and SEDS initial/locked checkpoints are unresolved. |
| M01 — skeleton | passed | Method-owned package/config/CLI and deterministic fixture exist under `methods/dive`; editable install and `dive --help` work. Unknown/missing keys and semantic violations fail closed. |
| M02 — evaluator oracle | passed | ID-based `[video,text]` evaluator passes asymmetric-direction, exact-tie, duplicate-ID, multi-positive and nonfinite/full-gallery guards. |
| M03 — score/loss core | passed | Dense/chunked masked late interaction, centered composition and retrieval/local/pair kernels pass padding, empty evidence, cosine-bound, four-margin, H-denominator, gradient and FP64 gradcheck oracles. |
| M04 — data contracts | passed | Versioned manifest/split guards, normalized word/numeric units, native subword offset mapping with partial-target rejection, deterministic canonical/shifted grids, raw-time RF mapping, per-step pose normalization, P/C construction and bool-mask collation have independent fixtures. |
| M05 — SEDS adapter | in_progress | Official source is reproducibly checked out at pinned commit `434e3f7...` by `methods/dive/scripts/setup_seds.sh`; typed adapter protocol exists. Compatible SEDS checkpoint/features and real score/tap parity remain unresolved. |
| M06 — B0 reproduction | blocked | No controlled SEDS How2Sign reproduction config/checkpoint has been verified. Existing `H2S_sota.pth` is a CiCo checkpoint and is not relabeled as SEDS. |
| M07 — evidence/reference | in_progress | Pointwise local RGB+pose projector, frozen BN mode/affine, deep-copied reference/student, state hash including buffers, storage independence, identity score, nonzero student gradient and raw-window locality tests pass. Warm-up stage runner/checkpoint artifacts remain. |
| M08 — mining/audit | not_started | Train-only shortlist, strict numeric proposal bank and human audit export remain. |
| M09 — support | not_started | Two-view support, rebin/JSD/RF gates and bank remain. |
| M10 — sampler/step | not_started | Reproducible shared plans and student step remain. |
| M11 — end-to-end smoke | not_started | Requires M04–M10. |
| M12 — primary pilot | blocked | Requires completed core, controlled B0/reference, accepted audit and real features. |
| M13 — mechanism | blocked | Requires a successful primary pilot and support-common artifacts. |
| M14 — final protocol | blocked | Requires passed research gates, locked experiment plan and actual final resources. |
| M15 — optimization | not_started | Optional; core correctness remains single-process FP32. |

## Inventory details

- How2Sign root: `/home/shared_data/sign_language/How2Sign` (86 GiB), with train/eval/test raw
  videos and poses present, subject to the existing missing-clip audit documented by proposal 1.
- Other reusable dataset roots are present: Phoenix14T (48 GiB) and CSL-Daily (42 GiB).
- Existing upstream checkouts are CiCo/SLRT at `38a4f7b...` and official SEDS at the required
  detached commit `434e3f7...`; both live under ignored `third_party/` paths.
- Available `artifacts/pretrained/H2S_sota.pth` is tracked in existing provenance as CiCo.
- GPU was idle enough at inventory time (48,519 MiB free), but no DIVE training is authorized by
  resource presence alone: the controlled SEDS baseline and correctness gates must come first.

## Verification log

- `python -m pip install -e '.[dev]'`: editable package and `dive` entry point installed.
- `pytest -q methods/dive/tests`: 24 passed on CPU correctness fixtures.
- `pytest -q`: 148 passed across shared, ELSC and DIVE suites; no regression.
- `ruff check methods/dive/src methods/dive/tests conftest.py`: all checks passed.
- `dive doctor --config methods/dive/configs/fixture.yaml --stage fixture --output
  artifacts/dive/doctor_fixture.json`: ready=true; PyTorch CUDA and RTX 5880 Ada detected.
- `pytest -q methods/dive/tests`: 31 passed after M04 data contracts.
- `pytest -q`: 155 passed after M04; no ELSC/shared regression.
- `bash methods/dive/scripts/setup_seds.sh`: official SEDS checkout verified exactly at
  `434e3f714fcb6a7d1f4001fb9a246bbd93ec0246` with a clean detached worktree.
- `pytest -q methods/dive/tests`: 35 passed after evidence/reference contracts.
- `pytest -q`: 159 passed; no ELSC/shared regression.
