# ELSC implementation status

This file distinguishes implemented contracts from experiment results. It must not be used as a
SOTA claim. Last audited: 2026-09-07 20:53 (Asia/Ho_Chi_Minh).

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
| ELSC-Min and matched controls | Canonical batch-512 A0/A1/A2/A3/A4 runs for seeds 42/1337/2026 are hash-validated. Across seeds, Min reaches mean dev R@1 75.080 versus baseline 74.984 (+0.096 pp), adapter-only 75.048 (+0.032 pp), matched-caption 74.984 (+0.096 pp), and random-support 75.112 (-0.032 pp). A5/A6/A7 diagnostics are complete at seed 42. Lower-LR and generic local word-video three-seed screens reach +0.257 pp and +0.225 pp over baseline respectively. Registered weight-0.02 and five-epoch lexical-ramp screens both retain initialization at seed 42 (delta 0.000 pp) | Registered Gate G `no_go`, Gate M `no_go`; all four corrective Gate G screens `no_go` |
| ELSC-Full | RF closure/control matching and per-video cap exist; deterministic extractor emits verified input-frame RF for generated train/dev features | Implemented gate; experiment not run |
| DDP/AMP math | Global-count DDP auxiliary normalization test; FP16/BF16 scaler/resume code; BF16 real-checkpoint GPU backward smoke | Implemented for single-GPU MVP; multi-GPU run not claimed |
| Full-gallery evaluation | Blockwise CiCo score, direction-specific singleton tie behavior, multi-positive IDs, explicit per-query artifact | Implemented and tested |
| SAN fine-grained protocol | Missing official artifact returns `official_artifact_missing`, never a fabricated zero | Implemented gate |
| Inference export | The selected seed-42 Min checkpoint has a real 352.5 MB core+adapter export; local head/teacher/cache are absent and reload score parity has max absolute error 0.0 | Implemented and verified |
| Results integrity/statistics | `elsc.report` rederives metric summaries from full-gallery per-query ranks and requires matching evaluation/training-source contracts, config/dev-manifest/selection/checkpoint hashes, paired seeds/gallery IDs, and video-group hierarchical bootstrap; Gate G and Gate M are executable dev-only contracts | Implemented; canonical three-seed reports measured and revalidated |
| Dataset-transfer asset gate | `elsc.transfer_audit` hashes ordered IDs and annotations, checks video/text coverage and split overlap without using test feedback. Local CSL-Daily is exact at 18,401/1,077/1,176 annotations and 20,654 referenced videos. Its train/dev extraction plan, English-caption provenance, grouped sampler, baseline/Min configs, and queued Gate X runner are implemented. Local How2Sign annotations reference 118/2/6 unavailable train/dev/test clips; its engineering subset is exactly 100 train clips and is explicitly non-benchmark | CSL-Daily train+dev feature extraction running; How2Sign Gate X blocked on a protocol decision for missing release clips |

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
duration-matched random support by 0.032 pp, so both Gate G and Gate M are `no_go`. The seed-42
post-screen diagnostics are complete: shuffled lexical and head-only retain initialization at
75.241, random lexical neighbors reach 75.626, and generic local word-video contrast reaches
75.915. A source-matched Min reproduction exactly matches the registered seed-42 Min metrics. The
registered lower-LR three-seed screen improves mean dev R@1 by 0.257 pp and the local word-video
screen by 0.225 pp; both fail the 0.5 pp Gate G threshold. KEEP reaches 75.723 at seed 42 but was not
expanded after lower-LR failed Gate G. Reducing the lexical weight from 0.1 to 0.02 and ramping the
0.1 weight over five epochs were each screened under the same registered seed-42 Gate G contract;
both selected initialization at 75.241 and were not expanded. This exhausts the registered PH
corrective screens without using test feedback. Full and locked test evaluation remain blocked by
their gates; dataset/backbone transfer and any SOTA claim also remain pending.

Transfer audits are stored locally under `artifacts/transfer/`. They do not download data, hash
large video contents, extract features, or expose test captions in their output. The How2Sign
train/test ZIP inventories contain the same absence as the extracted clip directories, so selective
re-extraction cannot recover the 124 missing train/test references. The official project distributes
clips and English annotations separately and documents that manually re-aligned clips require
re-segmentation from full videos; no silent intersection/filtering is accepted as an official Gate X
protocol. CSL-Daily has no missing/extra video references or cross-split pair-ID overlap and is the
only full transfer dataset currently ready for a disk-budgeted feature extraction plan. The exact
CSL train+dev plan contains 19,478 videos and 2,883,145 temporal windows per stream. Its two-stream
write bound is 23.103 GiB; the launch projected 41.969 GiB free after extraction and enforces a
32 GiB reserve. The official `csl_sota.pth` is pinned at SHA-256
`bdf32b5083db7039f0b5678812154ee39e03541d1eea0f614571cedeb174886f`. English train captions
come only from pinned upstream `data_csl/train.pkl`; 797 official dev sentence groups were
translated from the local Chinese dev CSV with `Helsinki-NLP/opus-mt-zh-en` revision
`cf109095479db38d6df799875e34039d4938aaa6`. The translation/build artifacts record
`test_annotation_accessed=false`. No CSL test feature was extracted or evaluated. Two batch-128
I3D processes were launched under tmux with 24-hour hard timeouts, and a separate process-alive
queue will automatically run manifest validation and the seed-42 baseline/ELSC-Min dev screen once
both streams complete.
The registered multiseed queue expands only after a passing seed-42 Gate G. A second conditional
queue is defined for the CSL Full pilot: it requires the completed three-seed Gate X scope artifact,
passing G and M gates, the train-only RF/cache Gate F contract, a real batch-512 GPU preflight, and
a 28 GiB launch-disk reserve. The Full row is compared with a ten-epoch continued-Min row from the
same selected Min initialization; all reports remain dev-only and explicitly record no test access.

The source tree now supports multiple proposal methods without cross-contamination. ELSC-owned
Python, configs, campaign scripts, and tests live under `methods/elsc/`. Reusable feature/data/
evaluation/CiCo/transfer/resource utilities live under `shared/slr_common/`, while `third_party/`
remains external. Import-boundary tests reject dependencies from shared code back into a method;
thin root compatibility shims preserve historical commands and active run provenance.
