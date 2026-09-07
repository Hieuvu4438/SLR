# ELSC implementation status

This file distinguishes implemented contracts from experiment results. It must not be used as a
SOTA claim. Last audited: 2026-09-07 17:52 (Asia/Ho_Chi_Minh).

| Acceptance requirement | Current evidence | Status |
| --- | --- | --- |
| Pinned upstream and compatibility patch | SLRT commit `38a4f7b00da7a858d59b7fabe5093876a84db8e0`; `patches/cico_compat.patch` passes `git apply --check` | Implemented |
| Environment and checkpoint provenance | `environment.lock.yaml`; release/I3D hashes; official feature archive SHA and split scope; completed extraction report hashes and measured batch preflight | Implemented and verified |
| PH dev routing/path/fusion | Official split audit is disjoint and exact; generated roots contain exactly train/dev/test = 7,096/519/642 in both streams; all 8,257 GPU-aware files are byte-identical to the serial extraction | Verified |
| Bridge/scorer golden parity | Real `ph_sota.pth` component audit has max error `0.0`; official full-gallery metrics reproduce the paper after matching direction-specific tie kernels | Verified for the release artifact |
| Test isolation | Trainer reads/hashes only train+dev; test requires selected checkpoint, checkpoint SHA, config hash, and dev-manifest hash | Implemented and tested |
| BPE and one-span replacement | Upstream tokenizer identity, Unicode/HTML/repeated-word/truncation tests | Implemented and tested |
| Frozen teacher and train-only cache | Dev-selection provenance and all cache hashes are fail-fast; cache CLI accepts only `train` | Implemented and tested |
| Raw local branch | Pointwise zero-init adapter and pre-Transformer local head; padding/locality tests | Implemented and tested |
| Auxiliary gradient flow | Two-step zero-init test and periodic lexical-only adapter/head gradient diagnostics | Implemented and tested |
| ELSC-Min and matched controls | Canonical batch-512 A0/A1/A2/A3/A4 runs for seeds 42/1337/2026 are hash-validated. Across seeds, Min reaches mean dev R@1 75.080 versus baseline 74.984 (+0.096 pp), adapter-only 75.048 (+0.032 pp), matched-caption 74.984 (+0.096 pp), and random-support 75.112 (-0.032 pp). True support wins both matched controls only for seed 42 | Three-seed screen complete; Gate G `no_go`, Gate M `no_go` |
| ELSC-Full | RF closure/control matching and per-video cap exist; deterministic extractor emits verified input-frame RF for generated train/dev features | Implemented gate; experiment not run |
| DDP/AMP math | Global-count DDP auxiliary normalization test; FP16/BF16 scaler/resume code; BF16 real-checkpoint GPU backward smoke | Implemented for single-GPU MVP; multi-GPU run not claimed |
| Full-gallery evaluation | Blockwise CiCo score, direction-specific singleton tie behavior, multi-positive IDs, explicit per-query artifact | Implemented and tested |
| SAN fine-grained protocol | Missing official artifact returns `official_artifact_missing`, never a fabricated zero | Implemented gate |
| Inference export | The selected seed-42 Min checkpoint has a real 352.5 MB core+adapter export; local head/teacher/cache are absent and reload score parity has max absolute error 0.0 | Implemented and verified |
| Results integrity/statistics | `elsc.report` rederives metric summaries from full-gallery per-query ranks and requires matching evaluation/training-source contracts, config/dev-manifest/selection/checkpoint hashes, paired seeds/gallery IDs, and video-group hierarchical bootstrap; Gate G and Gate M are executable dev-only contracts | Implemented; canonical three-seed reports measured and revalidated |
| Dataset-transfer asset gate | `elsc.transfer_audit` hashes ordered IDs and annotations, checks video/text coverage and split overlap without extracting features or using test feedback. Local CSL-Daily is exact at 18,401/1,077/1,176 annotations and 20,654 referenced videos. Local How2Sign annotations reference 118/2/6 unavailable train/dev/test clips; its engineering subset is exactly 100 train clips and is explicitly non-benchmark | CSL-Daily ready for feature extraction; How2Sign Gate X blocked on a protocol decision for missing release clips |

Current execution gate: the approved official archive contains only Phoenix test features (642 per
stream). The release checkpoint reproduces the published T2V/V2T metrics. The public PH loader's
256-pixel GPU spatial path is reproduced closely for the BSL5K domain-agnostic encoder. The
downloadable domain-aware encoder is explicitly a How2Sign target checkpoint, not the unavailable
Phoenix target encoder, so its release comparison is diagnostic rather than a parity claim. Local
train/dev/test features are complete and consistently use those same two encoders; official test
features remain isolated. A real batch-512 BF16 optimizer-step preflight passed with 44.73 GB peak
reserved and 5.70 GB free. The canonical seed-42 Baseline completed 200 epochs and selected epoch 0
at mean dev R@1 75.241. Its train-only cache contains 9,778 final auxiliary records with verified
artifact hashes. ELSC-Min selected epoch 18 at 75.530 for seed 42 but retained the initialization
checkpoint for seeds 1337 and 2026. Across all three registered seeds, Min improves mean dev R@1
from 74.984 to 75.080 (+0.096 pp; hierarchical-bootstrap 95% CI [-0.353, 0.706]), with T2V
+0.321 pp and V2T -0.128 pp. It beats the matched-caption mean by +0.096 pp but trails
duration-matched random support by 0.032 pp, so both Gate G and Gate M are `no_go`. The registered
seed-42 lower-LR/keep corrective screen is running; post-screen diagnostics are queued and will only
start after a clean corrective exit. Full and locked test evaluation remain blocked by their gates;
dataset/backbone transfer and any SOTA claim also remain pending.

Transfer audits are stored locally under `artifacts/transfer/`. They do not download data, hash
large video contents, extract features, or expose test captions in their output. The How2Sign
train/test ZIP inventories contain the same absence as the extracted clip directories, so selective
re-extraction cannot recover the 124 missing train/test references. The official project distributes
clips and English annotations separately and documents that manually re-aligned clips require
re-segmentation from full videos; no silent intersection/filtering is accepted as an official Gate X
protocol. CSL-Daily has no missing/extra video references or cross-split pair-ID overlap and is the
only full transfer dataset currently ready for a disk-budgeted feature extraction plan.
