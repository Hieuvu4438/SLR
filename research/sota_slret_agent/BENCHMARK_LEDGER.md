# SLRet Benchmark Ledger (as of 2026-09-23)

Every number below is **REPORTED BY PAPER** and was copied from the primary PDF. The URL and table are in each row. Metric order is R@1 / R@5 / R@10 / MedR. "—" means the paper does not report that metric.
Full per-paper details are in `LITERATURE_MATRIX.md`, and the BibTeX is in `LITERATURE_REFERENCES.bib`.

## Comparison classes

- **(a) RGB-only, gloss-free.**
  - **(a1)** The visual encoder is trained on target data plus generic (ImageNet/CLIP) pretraining only.
  - **(a2)** A frozen I3D pretrained on *external sign data* is added. CiCo, UPRet and SAN-on-CiCo use an I3D from BSL-1K / Varol et al. 2021 "Read and Attend".
- **(b) RGB + pose.** SEDS also uses the external BSL-1K I3D and a SignBERT-initialised hand GCN.
- **(c) Uses gloss labels or large external sign corpora / multi-dataset sign pretraining.** This covers SPOT-ALIGN (WLASL/MSASL lexicons, BOBSL init, mouthing spotting) and Scaling-up (SL-1.5M pose-text, including gloss-labelled ISLR sets and BOBSL; pose-only).

## Checkpoint-selection legend

- **VAL** means the paper states that it selects on validation.
- **n.s.** means the paper does not state how it selects.
- **[code: TEST]** means the official GitHub code evaluates the test split every epoch and keeps the best epoch by test T2V R@1. The paper does not disclose this. Code checked:
  - `FangyunWei/SLRT/CiCo/CLCL/main_task_retrieval.py`
  - `longtaojiang/SEDS/main_task_retrieval.py` (it also calls `save_best_model`)
  - `xua222/UPRet/main_task_retrieval.py`
  - The val-evaluation lines are commented out in all three.

---

## How2Sign (test, 2,348 pairs after CiCo/SPOT-ALIGN filtering)

| Class | Method | Venue | T2V R@1/5/10/MedR | V2T R@1/5/10/MedR | Modalities | Extra supervision | Split | Ckpt selection | Source (table) |
|---|---|---|---|---|---|---|---|---|---|
| a1 | C2RL | TCSVT 2025 | **62.4** / 75.9 / 80.1 / — | **57.5** / 68.4 / 73.0 / — | RGB (ResNet-18 IN) | none; MBart-large-cc25 text encoders | test | n.s. | arxiv.org/pdf/2408.09949 Tab. VI |
| a2 | UPRet | ECCV 2024 | **59.1** / 71.5 / 75.7 / 1.0 | **53.4** / 65.4 / 70.0 / 1.0 | RGB (I3D feats) | BSL-1K I3D + domain-aware I3D; CLIP B/32 | test | n.s. [code: TEST] | arxiv.org/pdf/2405.19689 Tab. 1 (= ECVA camera-ready) |
| a2 | CiCo | CVPR 2023 | 56.6 / 69.9 / 74.7 / 1.0 | 51.6 / 64.8 / 70.1 / 1.0 | RGB (I3D feats) | BSL-1K I3D + ~64K target pseudo-labels; CLIP B/32 | test | n.s. [code: TEST] | arxiv.org/pdf/2303.12793 Tab. 1 |
| a2 | CiCo* (repro by UPRet) | — | 56.4 / 69.4 / 74.1 / 1.0 | 50.3 / 63.6 / 69.3 / 1.0 | RGB | as CiCo | test | n.s. | arxiv.org/pdf/2405.19689 Tab. 1 |
| b | **SEDS** | ACM MM 2024 | **62.5** / 75.1 / 80.1 / 1.0 | **57.9** / 70.4 / 74.9 / 1.0 | RGB (I3D) + pose (RTMPose GCN) | BSL-1K I3D; SignBERT hand init; CLIP B/32 | test | n.s. [code: TEST] | arxiv.org/pdf/2407.16394 Tab. 1 |
| c | SPOT-ALIGN SR+CM (arXiv v1) | CVPR 2022 | 34.2 / 48.0 / 52.6 / 8.0 | 23.6 / 47.0 / 53.0 / 7.5 | RGB | mouthing + dictionary spotting (WLASL, MSASL), BOBSL init; SR+CM late fusion | test | **VAL** (geo-mean R@1/5/10) | arxiv.org/pdf/2201.02495v1 Tab. 6 |
| c | SPOT-ALIGN SR+CM (v2 / CVF camera-ready) | CVPR 2022 | 32.8 / 47.7 / 52.9 / 7.0 | 23.3 / 48.5 / 53.7 / 7.0 | RGB | same | test | **VAL** | arxiv.org/pdf/2201.02495v2 Tab. 6; CVF open-access PDF |
| ? | CMCM | CVIU 2026 | UNVERIFIED | UNVERIFIED | RGB (per repo) | UNVERIFIED | ? | ? | doi 10.1016/j.cviu.2025.104631 (closed access) |

Strongest per class:
- **(a1)** C2RL: 62.4 T2V / 57.5 V2T.
- **(a2)** UPRet: 59.1 / 53.4.
- **(b)** SEDS: 62.5 / 57.9. This is the overall H2S best; C2RL is within 0.1 T2V and 0.4 V2T.
- **(c)** SPOT-ALIGN: 34.2 / 23.6 (v1).
- The Scaling-up paper (TPAMI 2025) does **not** report H2S retrieval.

---

## PHOENIX-2014T (test, 642 pairs)

| Class | Method | Venue | T2V R@1/5/10/MedR | V2T R@1/5/10/MedR | Modalities | Extra supervision | Split | Ckpt selection | Source (table) |
|---|---|---|---|---|---|---|---|---|---|
| a1 | **C2RL** | TCSVT 2025 | **78.7** / 92.2 / 94.9 / — | **77.6** / 91.3 / 94.2 / — | RGB | none; MBart text encoders | test | n.s. (all ablations on test) | arxiv.org/pdf/2408.09949 Tab. VI |
| a1 | GFSLT-VLP + SAN | ACL 2026 | 70.2 / 89.3 / 94.4 / — (MRR 78.7) | 67.4 / 85.4 / 90.5 / — (MRR 75.5) | RGB | gloss-free; SAN negatives from a trained GFSLT-VLP retrieval model | test (coarse) | n.s.; hyper-params swept on test | aclanthology.org/2026.acl-long.1302.pdf Tab. 1 |
| a1 | GFSLT-VLP (SAN's baseline, CLCL) | ACL 2026 | 67.9 / 88.4 / 93.8 / — (MRR 77.5) | 69.4 / 88.7 / 93.3 / — (MRR 77.9) | RGB | gloss-free | test | n.s. | same, Tab. 1 |
| a2 | UPRet | ECCV 2024 | **72.0** / 89.1 / 94.1 / 1.0 | **72.0** / 89.4 / 93.3 / 1.0 | RGB (I3D) | BSL-1K I3D; CLIP | test | n.s. [code: TEST] | arxiv.org/pdf/2405.19689 Tab. 2 |
| a2 | CiCo* (repro by UPRet) | — | 70.4 / 88.2 / 92.7 / 1.0 | 70.9 / 87.2 / 92.5 / 1.0 | RGB | as CiCo | test | n.s. | arxiv.org/pdf/2405.19689 Tab. 2 |
| a2 | CiCo | CVPR 2023 | 69.5 / 86.6 / 92.1 / 1.0 | 70.2 / 88.0 / 92.8 / 1.0 | RGB (I3D) | BSL-1K I3D + pseudo-labels; CLIP; Google-translated DE->EN text | test | n.s. [code: TEST] | arxiv.org/pdf/2303.12793 Tab. 2 |
| a2 | CiCo (SAN's run) | ACL 2026 | 69.2 / 87.2 / 92.2 / — (MRR 77.3) | 70.1 / 87.7 / 92.9 / — (MRR 78.2) | RGB | as CiCo | test | n.s. | aclanthology.org/2026.acl-long.1302.pdf Tab. 1 |
| a2 | CiCo + SAN | ACL 2026 | 68.1 / 87.4 / 91.7 / — (MRR 76.6) | 67.8 / 87.4 / 91.7 / — (MRR 76.2) | RGB | as CiCo + SAN hard negatives | test | n.s. | same, Tab. 1 |
| b | **SEDS** | ACM MM 2024 | **76.8** / 91.7 / 95.3 / 1.0 | **78.7** / 92.5 / 95.2 / 1.0 | RGB + pose | BSL-1K I3D; SignBERT; CLIP | test | n.s. [code: TEST] | arxiv.org/pdf/2407.16394 Tab. 2 |
| c | Scaling-up SLP (pose, SL-1.5M) | TPAMI 2025 | **74.5** / 93.3 / 95.6 / 1.0 | **75.1** / 92.1 / 95.3 / 1.0 | pose only | SL-1.5M (~1.55M pose-text incl. gloss-labelled ISLR sets, BOBSL, P14T/CSL/H2S); test-split exclusion **not stated** | test | n.s. | arxiv.org/pdf/2408.08544 Tab. X |
| c | SPOT-ALIGN Combination | CVPR 2022 | 55.8 / 79.6 / 87.2 / 1.0 | 53.1 / 79.4 / 86.1 / 1.0 | RGB (features of [10,31]) | CM + SLT-based text retrieval (SLT trained with P14T supervision) | test | VAL (stated for H2S; generic) | arxiv.org/pdf/2201.02495 Tab. 7 |
| c | SPOT-ALIGN Cross-modal | CVPR 2022 | 48.6 / 76.5 / 84.6 / 2.0 | 50.3 / 78.4 / 84.4 / 1.0 | RGB | as above | test | VAL | same, Tab. 7 |
| ? | CMCM | CVIU 2026 | UNVERIFIED | UNVERIFIED | — | — | — | — | closed access |

Strongest per class:
- **(a1)** C2RL: 78.7 / 77.6. This is the overall best T2V.
- **(a2)** UPRet: 72.0 / 72.0.
- **(b)** SEDS: 76.8 / 78.7. This is the overall best V2T, tied with C2RL T2V R@1 at 78.7.
- **(c)** Scaling-up: 74.5 / 75.1.
- SAN does **not** raise the standard (coarse) benchmark. With CiCo, V2T R@1 falls from 70.1 to 67.8; with GFSLT-VLP, T2V R@1 rises from 67.9 to 70.2 and V2T R@1 falls from 69.4 to 67.4. SAN's contribution is the fine-grained stress-test gain (V2T R@1 16.8 -> 49.1 for GFSLT-VLP; 17.9 -> 39.4 for CiCo), which is not comparable to other papers.

---

## CSL-Daily (test, 1,176 pairs)

| Class | Method | Venue | T2V R@1/5/10/MedR | V2T R@1/5/10/MedR | Modalities | Extra supervision | Split | Ckpt selection | Source (table) |
|---|---|---|---|---|---|---|---|---|---|
| a1 | **C2RL** | TCSVT 2025 | **90.3** / 96.4 / 97.7 / — | **88.4** / 95.7 / 97.1 / — | RGB | none; MBart text encoders (native Chinese) | test | n.s. | arxiv.org/pdf/2408.09949 Tab. VI |
| a2 | UPRet | ECCV 2024 | **78.4** / 89.1 / 92.0 / 1.0 | **77.0** / 89.2 / 92.7 / 1.0 | RGB (I3D) | BSL-1K I3D; CLIP | test | n.s. [code: TEST] | arxiv.org/pdf/2405.19689 Tab. 3 |
| a2 | CiCo* (repro by UPRet) | — | 76.3 / 88.6 / 92.1 / 1.0 | 73.9 / 87.9 / 92.0 / 1.0 | RGB | as CiCo | test | n.s. | arxiv.org/pdf/2405.19689 Tab. 3 |
| a2 | CiCo | CVPR 2023 | 75.3 / 88.2 / 91.9 / 1.0 | 74.7 / 89.4 / 92.2 / 1.0 | RGB (I3D) | BSL-1K I3D; CLIP; Google-translated ZH->EN text | test | n.s. [code: TEST] | arxiv.org/pdf/2303.12793 Tab. 3 |
| b | **SEDS** | ACM MM 2024 | **85.8** / 94.4 / 95.6 / 1.0 | **85.4** / 93.8 / 95.8 / 1.0 | RGB + pose | BSL-1K I3D; SignBERT; CLIP | test | n.s. [code: TEST] | arxiv.org/pdf/2407.16394 Tab. 3 |
| c | Scaling-up SLP (pose, SL-1.5M) | TPAMI 2025 | **87.5** / 95.2 / 97.6 / 1.0 | **87.2** / 95.0 / 97.2 / 1.0 | pose only | SL-1.5M; test-split exclusion not stated | test | n.s. | arxiv.org/pdf/2408.08544 Tab. X |
| ? | CMCM | CVIU 2026 | UNVERIFIED | UNVERIFIED | — | — | — | — | closed access |

Strongest per class:
- **(a1)** C2RL: 90.3 / 88.4. This is the overall best.
- **(a2)** UPRet: 78.4 / 77.0.
- **(b)** SEDS: 85.8 / 85.4.
- **(c)** Scaling-up: 87.5 / 87.2.
- The SEDS pose-only ablation reaches 80.5 / 80.0 (SEDS Tab. 4).

---

## Protocol and fairness flags

1. **Inference-time score normalisation.** None of the published SLRet results above uses Dual-Softmax, QB-Norm, inverted softmax or CSLS at inference. None of the papers describes it.
   - In the CiCo, SEDS and UPRet code, the softmax runs over tokens within a video-text pair (τ = 0.07). The final score averages the V2T-pooled and T2V-pooled scores (`dual_mix = 0.5`). There is no normalisation over the test gallery.
   - There are no hidden DSL gains to discount. Any new method that adds DSL or QB-Norm must report numbers both with and without it.
2. **Late fusion.** SPOT-ALIGN "SR+CM" / "Combination" averages the similarities of two separately trained models: an ensemble plus a recognition/translation text matcher. This is disclosed.
3. **Test-set checkpoint selection.**
   - The CiCo, SEDS and UPRet official code choose the best epoch by **test T2V R@1**, with no validation evaluation. None of the three papers discloses this.
   - C2RL, Scaling-up and SAN do not state their selection. All three run ablations or hyper-parameter sweeps on the test split.
   - Only SPOT-ALIGN explicitly selects on validation.
   - Expect all post-2023 numbers to be optimistically biased by the best-of-epochs-on-test effect. It is largest on the small P14T test set (642).
4. **SPOT-ALIGN number drift.** CiCo, SEDS, UPRet and C2RL quote the arXiv v1 H2S numbers (34.2 / 23.6). The CVPR camera-ready (arXiv v2) reports 32.8 / 23.3.
5. **Copy errors in later papers' baseline tables.**
   - UPRet Tab. 2 gives SA-CM V2T MedR as 14.0 (the original is 1.0) and SA-COMB T2V R@10 as 92.1 (the original is 87.2).
   - SEDS Tab. 2 labels the "Translation" baseline "SA-SR", and its text says "+6.3" H2S T2V where the tables give +5.9.
   - C2RL's text says OpenASL T2V 62.6 while its table says 62.2.
6. **Capacity and data confounds.**
   - C2RL uses two MBart-large encoders for retrieval; the others use CLIP ViT-B/32.
   - CiCo, UPRet and SEDS rely on external BSL-1K I3D features, and SEDS also on SignBERT and RTMPose.
   - Scaling-up pretrains on about 1.55M pose-text pairs whose sources include the target datasets.
   - CiCo-family methods use Google-translated English text for P14T and CSL; C2RL and Scaling-up use native-language MBart.
7. **Seeds.** No SLRet paper after SPOT-ALIGN reports multiple seeds or variance. SPOT-ALIGN reports 3-seed mean ± std only for its CM ablations.
8. **Released artefacts.**
   - CiCo: code, features and checkpoints.
   - SEDS: code, features, keypoints and checkpoints (BaiduDrive).
   - UPRet: code only.
   - SAN: repo says "Coming Soon".
   - CMCM: partial modules only.
   - C2RL and Scaling-up: none found.

## Headline: strongest published comparable results (2026-09)

| Dataset | Best overall (T2V / V2T R@1) | Best RGB-only, no external sign data (a1) | Best CLIP-B/32 + external I3D family (a2 / b) |
|---|---|---|---|
| How2Sign | SEDS 62.5 / 57.9 (b); C2RL 62.4 / 57.5 (a1) | C2RL 62.4 / 57.5 | SEDS 62.5 / 57.9 (b); UPRet 59.1 / 53.4 (a2) |
| PHOENIX-2014T | C2RL 78.7 / 77.6 (a1); SEDS 76.8 / 78.7 (b) | C2RL 78.7 / 77.6 | SEDS 76.8 / 78.7 (b); UPRet 72.0 / 72.0 (a2) |
| CSL-Daily | C2RL 90.3 / 88.4 (a1) | C2RL 90.3 / 88.4 | SEDS 85.8 / 85.4 (b); UPRet 78.4 / 77.0 (a2) |

CMCM (CVIU 2026) could not be verified. If its full text shows higher numbers, this table must be updated.
