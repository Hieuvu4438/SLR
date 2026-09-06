# ELSC implementation status

This file distinguishes implemented contracts from experiment results. It must not be used as a
SOTA claim. Last audited: 2026-09-07 (Asia/Ho_Chi_Minh).

| Acceptance requirement | Current evidence | Status |
| --- | --- | --- |
| Pinned upstream and compatibility patch | SLRT commit `38a4f7b00da7a858d59b7fabe5093876a84db8e0`; `patches/cico_compat.patch` passes `git apply --check` | Implemented |
| Environment and checkpoint provenance | `environment.lock.yaml`; release/I3D hashes; official feature archive SHA and split scope recorded | Implemented; generated all-split features pending |
| PH dev routing/path/fusion | Official split audit is disjoint and exact; release test features are isolated from locally extracted all-split features | Partially verified; all-split extraction scheduled |
| Bridge/scorer golden parity | Real `ph_sota.pth` component audit has max error `0.0`; official full-gallery metrics reproduce the paper after matching direction-specific tie kernels | Verified for the release artifact |
| Test isolation | Trainer reads/hashes only train+dev; test requires selected checkpoint, checkpoint SHA, config hash, and dev-manifest hash | Implemented and tested |
| BPE and one-span replacement | Upstream tokenizer identity, Unicode/HTML/repeated-word/truncation tests | Implemented and tested |
| Frozen teacher and train-only cache | Dev-selection provenance and all cache hashes are fail-fast; cache CLI accepts only `train` | Implemented and tested |
| Raw local branch | Pointwise zero-init adapter and pre-Transformer local head; padding/locality tests | Implemented and tested |
| Auxiliary gradient flow | Two-step zero-init test and periodic lexical-only adapter/head gradient diagnostics | Implemented and tested |
| ELSC-Min and matched controls | Configs and execution paths exist; measured PH runs have not started | Not run |
| ELSC-Full | RF closure/control matching and per-video cap exist; deterministic extractor emits verified input-frame RF for generated train/dev features | Implemented gate; experiment not run |
| DDP/AMP math | Global-count DDP auxiliary normalization test; FP16/BF16 scaler/resume code; BF16 real-checkpoint GPU backward smoke | Implemented for single-GPU MVP; multi-GPU run not claimed |
| Full-gallery evaluation | Blockwise CiCo score, direction-specific singleton tie behavior, multi-positive IDs, explicit per-query artifact | Implemented and tested |
| SAN fine-grained protocol | Missing official artifact returns `official_artifact_missing`, never a fabricated zero | Implemented gate |
| Inference export | Core+adapter export excludes training-only head/cache/teacher and reload parity is enforced | Implemented; real run export pending |
| Results integrity/statistics | `elsc.report` accepts only measured local artifacts, enforces paired seeds/gallery IDs, and performs video-group hierarchical bootstrap | Implemented; no measured comparison yet |

Current execution gate: the approved official archive contains only Phoenix test features (642 per
stream). The release checkpoint reproduces the published T2V/V2T metrics. The public PH loader's
256-pixel GPU spatial path is now reproduced closely for the BSL5K domain-agnostic encoder. The
downloadable domain-aware encoder is explicitly a How2Sign target checkpoint, not the unavailable
Phoenix target encoder, so its release comparison is diagnostic rather than a parity claim.
Training will use locally extracted features from the same two checkpoints for train/dev/test and
will not mix those distributions with official test features. Baseline selection, ELSC
mining/ablations, and any SOTA comparison remain unverified until extraction and measured runs
finish.
