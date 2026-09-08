# DIVE-SLR v2 implementation status

Updated: 2026-09-08. Status describes repository evidence, not intended work. Synthetic fixtures
are never counted as a benchmark replication.

| Milestone | Status | Evidence / blocker |
|---|---|---|
| M00 — inventory | passed | `dive doctor` records Python/PyTorch/CUDA and stage resources. RTX 5880 Ada (49,140 MiB), driver 570.195.03, PyTorch 2.11.0+cu128/CUDA 12.8 were detected. How2Sign raw video/pose roots exist and both official SEDS Baidu shares are live. Pinned SEDS source is present; reproduction config and local SEDS initial/locked checkpoints remain unresolved because current Baidu routes these >50 MB anonymous downloads through its authenticated desktop-client flow. |
| M01 — skeleton | passed | Method-owned package/config/CLI and deterministic fixture exist under `methods/dive`; editable install and `dive --help` work. Unknown/missing keys and semantic violations fail closed. |
| M02 — evaluator oracle | passed | ID-based `[video,text]` evaluator passes asymmetric-direction, exact-tie, duplicate-ID, multi-positive and nonfinite/full-gallery guards. |
| M03 — score/loss core | passed | Dense/chunked masked late interaction, centered composition and retrieval/local/pair kernels pass padding, empty evidence, cosine-bound, four-margin, H-denominator, gradient and FP64 gradcheck oracles. |
| M04 — data contracts | passed | Versioned manifest/split guards, normalized word/numeric units, native subword offset mapping with partial-target rejection, deterministic canonical/shifted grids, raw-time RF mapping, per-step pose normalization, P/C construction and bool-mask collation have independent fixtures. Real `dive prepare-data` completed for 34,888 controlled How2Sign samples with frame-exact video/pose maps, disjoint sources, multi-video positives and 22,785 ambiguity exclusions. `dive validate-data` enforces exact bidirectional relevance coverage, legal excluded negatives, source-video split isolation, media/pose/RGB-key/frame-map presence and translation provenance, and registers a checksummed audit; it currently stops at the absent SEDS I3D cache as intended. |
| M05 — SEDS adapter | in_progress | Code is complete for clean pinned-checkout/model-state validation, lazy official construction, native contextual encoding, prelogit fusion directional mixing, mask-before-softmax, explicit CLS/text-mask conversion, projected subword unit pooling, pre-Transformer RGB tap, cloned local GCN/sign-conv pose path and conservative raw RF metadata. Unpadded native-formula and padded-invariance fixtures pass; compatible real checkpoint/features and real score/tap parity remain unresolved. |
| M06 — B0 reproduction | blocked | The released SEDS loader has no dev annotation and selects training checkpoints on `test.pkl`; this is now isolated as published/debug only. Controlled manifests use 31,019 pinned-upstream train videos, an independent 1,527-sample dev split, and reserve all 2,342 pinned-upstream test videos for final evaluation, with zero source overlap. A controlled SEDS config/checkpoint and matching I3D features are still unverified. Existing `H2S_sota.pth` is CiCo and is not relabeled as SEDS. |
| M07 — evidence/reference | in_progress | Code is complete for the pointwise local RGB+pose projector, frozen BN mode/affine, audited AdamW ownership/decay groups, fixed-step warm-up/cosine schedule, symmetric `E_local` retrieval-only training, full-dev teacher evaluation, earliest-tie checkpoint selection, checksummed reference export, deep-copy independence and identity initialization. A real warm-up/reference artifact remains blocked on the SEDS checkpoint/features. |
| M08 — mining/audit | blocked | Code is complete for bidirectional pooled shortlist, S0 four-margin reranking, deterministic dedup, exact shortlist-coverage audit, strict numeric schema, blinded audit export, train-only finalize and fingerprint/checksum bank I/O. A real accepted human audit plus SEDS train representations are required to produce the main semantic bank. |
| M09 — support | blocked | Code is complete for differential own-minus-rival support, absolute/tiny-distance gates, raw-time rebin, JSD, concentration/RF gates, g calculation, and retaining failed records with g=0 in the bank. Actual reference features and two real shifted views are required for the main support artifact. |
| M10 — sampler/step | passed | Dedicated-RNG plans enforce unique effective batches, unordered-pair uniqueness, endpoint quota, ordinary fill, pre-rejection H and shared-control fingerprints. The exact centered student step uses explicit ID remaps, filters g=0 before nullable fields, computes global/local/pair losses, clips gradients and detaches B0/reference/text/q/g inputs. |
| M11 — end-to-end smoke | in_progress | `dive smoke` runs a deterministic CPU fixture through student forward, H=2/H_active=1 losses, optimizer update, checkpoint+checksum, reload parity and ID-based evaluation while emitting `benchmark_claim_allowed=false`. The run-state resolver now enforces checksummed shared/per-variant namespaces, immutable stage records, parent revalidation and no filesystem discovery. Tiny real SEDS integration remains blocked by checkpoint/features. |
| M12 — primary pilot | blocked | Dev opportunity-bound code is complete for both directions, stable IDs, multi-positive best targets and the exact `2*gamma` repairability gate. Running A0/A1/A2/A3/A6 still requires the controlled B0/reference, accepted audit and real features. |
| M13 — mechanism | blocked | Requires a successful primary pilot and support-common artifacts. |
| M14 — final protocol | blocked | Code is complete for checkpoint-first dev gamma calibration (including gamma=0 and smaller-gamma ties), selection provenance, duplicate-query ceilings, paired query/source-cluster bootstrap and seed mean/std. Final execution still requires passed research gates, a locked experiment plan and actual resources. |
| M15 — optimization | not_started | Optional; core correctness remains single-process FP32. |

## Inventory details

- How2Sign root: `/home/shared_data/sign_language/How2Sign` (86 GiB), with train/eval/test raw
  videos and poses present, subject to the existing missing-clip audit documented by proposal 1.
- Other reusable dataset roots are present: Phoenix14T (48 GiB) and CSL-Daily (42 GiB).
- Existing upstream checkouts are CiCo/SLRT at `38a4f7b...` and official SEDS at the required
  detached commit `434e3f7...`; both live under ignored `third_party/` paths.
- The official SEDS model share was verified to list the 804,176,411-byte How2Sign checkpoint and
  353,976,522-byte `ViT-B-32.pt`; the official feature share lists the 9,527,083,922-byte How2Sign
  archive. The files are not claimed as acquired: the current anonymous web policy forces files
  over 50 MB into the desktop client, while the current direct-link API requires account-derived
  credentials. The 40 GiB free filesystem is sufficient for the two model files but extraction of
  the feature archive is not attempted without a measured space plan.
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
- `pytest -q methods/dive/tests`: 50 passed after strict slot and support kernels.
- `pytest -q`: 174 passed; no ELSC/shared regression.
- `pytest -q methods/dive/tests`: 57 passed after neighbor/bank/audit completion.
- `pytest -q`: 181 passed; no ELSC/shared regression.
- `pytest -q methods/dive/tests`: 63 passed after sampler and exact student-step integration.
- `pytest -q`: 187 passed; no ELSC/shared regression.
- `dive smoke --config methods/dive/configs/fixture.yaml --output-dir artifacts/dive/smoke`:
  finite loss, nonzero gradient norm, checkpoint round-trip=true, explicit fixture-only report.
- `pytest -q methods/dive/tests`: 66 passed after checkpoint/resume and smoke CLI.
- `pytest -q`: 190 passed; no ELSC/shared regression.
- `pytest -q methods/dive/tests`: 71 passed after the evidence warm-up runner, optimizer ownership,
  schedule, dev-selection and reference-export contracts.
- `pytest -q`: 195 passed; no ELSC/shared regression.
- `pytest -q methods/dive/tests`: 78 passed after versioned tensor-cache fingerprints, atomic
  checksummed shards, explicit ID/mask/timestamp loading and T06 invalidation coverage.
- `pytest -q`: 202 passed; no ELSC/shared regression.
- `pytest -q methods/dive/tests`: 87 passed after opportunity, calibration, duplicate-ceiling,
  paired-bootstrap and seed-aggregation evaluation contracts.
- `pytest -q`: 211 passed; no ELSC/shared regression.
- `pytest -q methods/dive/tests`: 93 passed after the pinned SEDS adapter, mask/score/tap/RF
  contracts and synchronized three-stream pose integration.
- `pytest -q`: 217 passed; no ELSC/shared regression.
- `pytest -q methods/dive/tests`: 100 passed after the atomic run-state/artifact resolver, exact
  shared/variant layout, content mutation checks and no-discovery parent resolution.
- `pytest -q methods/dive/tests`: 105 passed after the executable prepared-data validator and its
  relevance, excluded-negative, source-leakage, asset and run-state contracts.
- `pytest -q`: 229 passed; no ELSC/shared regression.
- `dive prepare-data --config methods/dive/configs/how2sign_base.yaml --workers 16`: passed on
  34,888 real samples; all video/RTM-pose frame counts match. Generated train/dev/test counts are
  31,019/1,527/2,342 with 30,786/1,527/1,964 text IDs, zero cross-split sample/source overlap and
  22,785 excluded ambiguous negatives. The manifests are registered under ignored
  `runs/dive_v2/shared/seed17/`.
- Real prepared-artifact reload verified exact manifest/relevance/frame-map ordering and legal
  exclusions. `dive validate-data` resolves its prepared parents, then exits 2 with only
  `MISSING_RESOURCE: data.rgb_cache_root`, accurately exposing the next resource gate.
- `pytest -q methods/dive/tests`: 108 passed after controlled preparation; `pytest -q`: 232 passed.
