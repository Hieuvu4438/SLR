# SLRet Literature Matrix (as of 2026-09-23)

Scope: sign-language text<->video retrieval (SLRet) on How2Sign (H2S), PHOENIX-2014T (P14T), CSL-Daily (CSL).
All numbers are **REPORTED BY PAPER**. Each is copied from the primary source (arXiv PDF, CVF, ECVA or ACL Anthology) and the source URL and table are given with it.
Nothing was taken from the local repo. Where a primary source could not be read, the entry says **UNVERIFIED**.

Metric order everywhere is R@1 / R@5 / R@10 / MedR (T2V = text->video, V2T = video->text).

Legend used below
- "Ext. sign data" means a visual encoder pretrained on sign data from outside the target dataset (e.g. BSL-1K/BOBSL I3D, WLASL/MSASL, SL-1.5M).
- "Checkpoint selection" records only what the **paper text** says. Where official code was inspected on GitHub, that is marked separately as "[code]".

---

## 1. SPOT-ALIGN: Duarte, Albanie, Giro-i-Nieto, Varol. "Sign Language Video Retrieval with Free-Form Textual Queries." CVPR 2022.

- Sources: arXiv v1 https://arxiv.org/pdf/2201.02495v1 ; arXiv v2 / CVF camera-ready https://arxiv.org/pdf/2201.02495v2 , https://openaccess.thecvf.com/content/CVPR2022/papers/Duarte_Sign_Language_Video_Retrieval_With_Free-Form_Textual_Queries_CVPR_2022_paper.pdf ; project page https://imatge-upc.github.io/sl_retrieval/
- Venue/year: CVPR 2022 (pp. 14074-14084, DOI 10.1109/CVPR52688.2022.01370)
- Datasets: How2Sign (main), PHOENIX-2014T (baseline). H2S split used: 31,075 / 1,739 / 2,348 in v2, and 31,085 in CiCo's restatement, after invalid pairs are removed.
- Modalities: RGB only.
- Extra supervision: no gloss. It does use a lot of external sign data. Automatic sign annotations come from mouthing-based spotting (BSL-1K-style keyword spotter). Dictionary spotting uses WLASL and MSASL exemplars. The I3D is initialised from a BOBSL-pretrained BSL recognition model, and three spot-and-retrain rounds follow (M+D1..D3, 1887-sign vocabulary).
- Pretraining data: Kinetics, then BOBSL, then How2Sign sign recognition on the automatic annotations.
- Video encoder: I3D sign-recognition embedding (1024-d, frozen for retrieval). It is applied in a sliding window, average-pooled over time, and projected with a gated embedding unit (MoEE/CE style).
- Pose encoder: none.
- Text encoder: frozen word embeddings (GrOVLE best; W2V, GPT, GPT-2-xl and ALBERT-XL were ablated), NetVLAD, and a gated embedding unit. P14T uses German GPT-2.
- Temporal representation: temporal average pooling of clip features into one global vector.
- Local alignment: none.
- Global alignment: joint embedding with cosine similarity.
- Scoring: cosine similarity in the joint space. The "COMB" model late-fuses, by equal-weight averaging, the CM similarity and a text-based IoU similarity between recognised sign words (H2S) or SLT output (P14T) and the query.
- Losses: bidirectional max-margin ranking loss (m = 0.2) for CM. Cross-entropy for the sign classifier.
- Negative sampling: in-batch negatives (margin loss); no hard-negative mining is described.
- Uncertainty modeling: none.
- Training-only vs inference: the spotting pipeline is training-time only. The SR classifier and the IoU text matching are used at inference in COMB.
- Reported results. H2S test (Table 6), **numbers differ between arXiv v1 and v2/CVF**.

| version | model | T2V R@1/5/10/MedR | V2T R@1/5/10/MedR |
|---|---|---|---|
| arXiv v1 (Tab. 6) | SR | 18.9 / 32.1 / 36.5 / 62.0 | 11.6 / 27.4 / 32.5 / 69.0 |
| arXiv v1 (Tab. 6) | CM | 24.3 / 40.7 / 46.5 / 16.0 | 17.9 / 40.1 / 46.9 / 14.0 |
| arXiv v1 (Tab. 6) | SR+CM (COMB) | 34.2 / 48.0 / 52.6 / 8.0 | 23.6 / 47.0 / 53.0 / 7.5 |
| arXiv v2 = CVF (Tab. 6) | SR | 18.4 / 32.2 / 36.5 / 68.0 | 11.5 / 27.9 / 33.3 / 66.0 |
| arXiv v2 = CVF (Tab. 6) | CM | 24.7 / 39.6 / 46.0 / 17.0 | 17.9 / 40.8 / 46.6 / 15.0 |
| arXiv v2 = CVF (Tab. 6) | SR+CM (COMB) | 32.8 / 47.7 / 52.9 / 7.0 | 23.3 / 48.5 / 53.7 / 7.0 |

  - P14T test (Table 7, same in v1/v2): Translation-based 30.2/53.1/63.4/4.5, V2T 28.8/52.0/60.8/56.1. Cross-modal 48.6/76.5/84.6/2.0, V2T 50.3/78.4/84.4/1.0. Combination 55.8/79.6/87.2/1.0, V2T 53.1/79.4/86.1/1.0.
  - CSL: not reported.
  - NOTE: CiCo, SEDS, UPRet and C2RL all quote the **arXiv v1** H2S numbers (34.2 / 23.6), not the camera-ready ones (32.8 / 23.3).
- Evaluation protocol: test split. **The checkpoint is selected on the validation set**. "We use the validation set to tune parameters (i.e. training epoch)" (Sec. 4.1). Appendix B.4 adds: "the epoch achieving the highest geometric mean of R@1, R@5 and R@10 on the validation set was used to select the final model."
- Seeds: CM ablations report mean ± std over **3 random seeds**. The final Table 6 numbers are single values.
- Code/checkpoints: project page with the manually verified recognition test set. A full training-code release was not verified.
- Limitations (stated): cannot discover signs outside the queried lexicon vocabulary; fails on generic, less detailed queries; interpreted rather than conversational signing.
- Solves: defines the free-form text SLRet task and benchmark, and shows that sign-embedding quality is the bottleneck.
- Unsolved: weak text encoder, no token-level alignment, heavy multi-round spotting pipeline.

---

## 2. CiCo: Cheng, Wei, Bao, Chen, Zhang. "CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning." CVPR 2023.

- Sources: https://arxiv.org/pdf/2303.12793 (v1). Code: https://github.com/FangyunWei/SLRT/tree/main/CiCo
- Venue/year: CVPR 2023, pp. 19016-19026, DOI 10.1109/CVPR52729.2023.01823
- Datasets: H2S (31,085 / 1,739 / 2,348 after filtering), P14T (7,096 / 519 / 642), CSL (18,401 / 1,077 / 1,176).
- Modalities: RGB only (offline pre-extracted I3D clip features).
- Extra supervision: no gloss. It does use **external sign data**:
  - The "domain-agnostic" I3D was pretrained on BSL-1K. The paper cites Varol et al. CVPR 2021 "Read and Attend", and the README checkpoint is `bsl5k.pth.tar`.
  - A "domain-aware" I3D is fine-tuned on about 64K target-set pseudo-labels (1,220 words, threshold λ = 0.6).
  - The two I3Ds are fused as H(v) = α·h_agnostic + (1-α)·h_aware, with α = 0.8 (README uses 0.9 for P14T and 0.8 for CSL).
  - P14T and CSL texts are **Google-translated to English** to reuse CLIP's text encoder.
- Pretraining data: BSL-1K / BOBSL-derived I3D, and CLIP ViT-B/32 (image and text towers).
- Video encoder: I3D sliding window (16 frames, stride 1), up to 64 clip features, then a 12-layer Transformer initialised from **CLIP ViT-B/32 image encoder** (it operates on feature tokens, not patches).
- Pose encoder: none.
- Text encoder: CLIP ViT-B/32 text Transformer (full fine-tuning is best; freezing layers hurts, Table 9). Max 32 tokens.
- Temporal representation: sequence of M ≤ 64 clip tokens (no pooling before matching).
- Local alignment: clip-word similarity matrix E = S·Wᵀ (M×L).
- Global alignment: E is turned into a scalar by a softmax-weighted sum and then an average. This is the "cross-lingual contrastive learning" (CLCL) step.
- Similarity:
  - V2T: z = mean_i Σ_j E_ij·softmax_row(E)_ij.
  - T2V: the same with column-wise softmax and sum.
  - [code]: the token softmax has temperature 0.07, and inference scores are 0.5·z_V2T + 0.5·z_T2V (`dual_mix = 0.5`).
  - The softmax is **within a pair over tokens**, not over the gallery.
- Losses: symmetric InfoNCE on Z_V2T and Z_T2V, with L = β·L_V2T + (1-β)·L_T2V and β = 0.5. Text augmentation is random word swap only; delete and synonym replacement hurt (Table 7).
- Negative sampling: in-batch only (batch 512).
- Uncertainty modeling: none.
- Training-only vs inference: the pseudo-labelling and domain-aware fine-tune happen offline, but both I3Ds are used at inference (feature extraction).
- Reported results (test):
  - H2S (Tab. 1): T2V 56.6 / 69.9 / 74.7 / 1.0; V2T 51.6 / 64.8 / 70.1 / 1.0.
  - P14T (Tab. 2): T2V 69.5 / 86.6 / 92.1 / 1.0; V2T 70.2 / 88.0 / 92.8 / 1.0.
  - CSL (Tab. 3): T2V 75.3 / 88.2 / 91.9 / 1.0; V2T 74.7 / 89.4 / 92.2 / 1.0.
  - The README table has identical numbers.
- Evaluation protocol:
  - The paper does not describe how the checkpoint is selected.
  - All ablations (Tables 4-9) are on the H2S **test** split.
  - **[code]**:
    - `main_task_retrieval.py` builds only a *test* dataloader.
    - It runs `eval_epoch(..., test_dataloader)` every epoch and tracks `best_epoch` / `best_score` by **test T2V R@1**.
    - The val-set evaluation lines are commented out.
    - Released checkpoints are named `*_sota.pth`.
  - This test-set selection is **not disclosed in the paper**.
- Seeds: not reported. Results come from single runs.
- Code/checkpoints: yes. Code, I3D features, domain-aware I3D, and CLCL checkpoints (H2S / P14T / CSL) are released.
- Limitations: two-stage training with an offline, frozen encoder; English-translation dependency for DGS/CSL text; no uncertainty or hard-negative modelling.
- Solves: fine-grained sign-to-word soft alignment without annotations. Large gains over SPOT-ALIGN; CLCL contributes +20.7 T2V R@1 in the ablation.
- Unsolved: visual confusability; one-to-many ambiguity; evaluation hygiene (test-based epoch selection).

---

## 3. UPRet: Wu, Li, Luo, Cheng, Zhuang, Cao, Fu. "Uncertainty-aware Sign Language Video Retrieval with Probability Distribution Modeling." ECCV 2024.

- Sources: https://arxiv.org/pdf/2405.19689 (v1); ECVA camera-ready https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06074.pdf (numbers identical to arXiv); supplement ..._06074-supp.pdf. Code: https://github.com/xua222/UPRet
- Venue/year: ECCV 2024, LNCS pp. 390-408, DOI 10.1007/978-3-031-72784-9_22
- Datasets: H2S, P14T, CSL (the paper quotes the raw split sizes 31,164 / 1,740 / 2,356; the CiCo-filtered splits are presumably used since the code is CiCo-based).
- Modalities: RGB only (CiCo's I3D features).
- Extra supervision: no gloss. It uses the same external sign-pretrained I3D (BSL-1K) plus the domain-aware fusion as CiCo.
- Pretraining data: BSL-1K I3D and CLIP ViT-B/32.
- Video encoder: CiCo sign encoder, then a CLIP ViT-B/32 image Transformer.
- Pose encoder: none.
- Text encoder: CLIP ViT-B/32 text encoder.
- Temporal representation: clip-token sequence (max 64).
- Local alignment:
  - Token-wise alignment matrix a_ij = V_i·T_j.
  - Weighted max-pooling in both directions, with MLP-softmax token weights ω (DRL/X-Pool style).
  - Plus an optimal-transport (Sinkhorn) plan over clip and word tokens with cost 1 - E and uniform marginals.
- Global alignment: InfoNCE on the fused score.
- Similarity (Eq. 24): S = ½[(Σ_i ω_v^i max_j a_ij + λ_ot·S_ot) + (Σ_j ω_t^j max_i a_ij + λ_ot·S_ot)], with λ_ot = 1.0. **"S_ot is calculated only during training."**
- Losses:
  - Symmetric InfoNCE on S (Eq. 26).
  - OT-distance cross-entropy L_D with a fixed OT plan (Eq. 22).
  - Distribution modelling: features are split in half and passed through attention+MLP to predict μ and σ. There are K = 2 reparameterised samples, and features are averaged with the samples (Eq. 12).
- Negative sampling: in-batch (batch 512).
- Uncertainty modeling: diagonal Gaussian per token/feature with Monte-Carlo sampling. This follows PCME / UATVR, but the distributions are used to *augment* the features rather than to score with a probabilistic distance.
- Training-only vs inference: distribution modelling, sampling and OT are **inactive at inference** (Sec. 4.3, Table 5). Inference time is 2.55 vs 2.54 for the baseline.
- Reported results (test; paper also reports MnR):
  - H2S (Tab. 1): T2V 59.1 / 71.5 / 75.7 / 1.0 (MnR 54.4); V2T 53.4 / 65.4 / 70.0 / 1.0 (MnR 76.4).
  - P14T (Tab. 2): T2V 72.0 / 89.1 / 94.1 / 1.0 (MnR 4.4); V2T 72.0 / 89.4 / 93.3 / 1.0 (MnR 4.6).
  - CSL (Tab. 3): T2V 78.4 / 89.1 / 92.0 / 1.0 (MnR 6.7); V2T 77.0 / 89.2 / 92.7 / 1.0 (MnR 5.5).
  - Their CiCo reproduction (CiCo*):
    - H2S 56.4/69.4/74.1/1.0, V2T 50.3/63.6/69.3/1.0.
    - P14T 70.4/88.2/92.7/1.0, V2T 70.9/87.2/92.5/1.0.
    - CSL 76.3/88.6/92.1/1.0, V2T 73.9/87.9/92.0/1.0.
  - Transcription errors in UPRet's Table 2 baselines: SA-CM V2T MedR is given as 14.0 (the original is 1.0), and SA-COMB T2V R@10 as 92.1 (the original is 87.2).
- Evaluation protocol:
  - The paper does not describe checkpoint selection.
  - Ablations (Tables 4-7) are on H2S (test).
  - **[code]**: `main_task_retrieval.py` has the same CiCo loop, i.e. per-epoch evaluation on the **test** dataloader with `best_epoch` taken by test R@1. The val evaluation is commented out.
- Seeds: not reported.
- Code/checkpoints: training code is released. No pretrained UPRet checkpoints are linked in the README.
- Limitations (stated): cross-lingual sign retrieval.
- Solves: models one-to-many polysemy (e.g. COLD/WINTER). The OT regulariser gives about +2.7 T2V R@1 over CiCo* on H2S.
- Unsolved: the gains are about 1.6-2.7 R@1 and single-seed. Uncertainty is not used at test time.

---

## 4. SEDS: Jiang, Wang, Li, Fang, Zhou, Li. "SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval." ACM MM 2024.

- Sources: https://arxiv.org/pdf/2407.16394 (v1, 23 Jul 2024). Code: https://github.com/longtaojiang/SEDS
- Venue/year: ACM MM 2024, pp. 5141-5150, DOI 10.1145/3664647.3681237
- Datasets: H2S (31,019 / 1,738 / 2,348, stated), P14T (7,096 / 519 / 642), CSL (18,401 / 1,077 / 1,176).
- Modalities: **RGB + pose**.
- Extra supervision: no gloss.
  - External sign data: the offline I3D is pretrained on **BSL-1K**. SEDS uses only the BSL-1K I3D; the paper does not mention CiCo's domain-aware fine-tune.
  - The hand GCN is initialised from **SignBERT** (itself pretrained on external hand-pose sign data).
  - RTMPose is used for keypoints.
- Pretraining data: BSL-1K I3D, SignBERT hand GCN, and CLIP ViT-B/32.
- Video encoder: frozen offline I3D (16-frame window, stride 1, T ≤ 64 clips), then a 12-layer "RGB interaction Transformer" initialised from the CLIP ViT-B/32 image encoder.
- Pose encoder:
  - Online GCN over 21 left-hand, 21 right-hand and 7 body keypoints (the hand GCN is shared).
  - 1D temporal convolution within each clip.
  - A 12-layer "Pose interaction Transformer" (CLIP-initialised).
- Fusion: Cross Gloss Attention Fusion (CGAF). This is deformable local cross-attention: N sampled offset positions around each clip, pose<->RGB, 2 layers, followed by MLP([f_p, f_r]) + f_p + f_r.
- Text encoder: CLIP ViT-B/32 text (max 32 tokens). The paper does not say how P14T and CSL text is handled; the code is CiCo-derived.
- Temporal representation: clip-token sequences.
- Local alignment: CiCo CLCL fine-grained matrices for the pose, RGB and fusion streams against words. There is also a **pose-RGB clip-clip matching objective**: a diagonal-sum of the softmax-reweighted T×T matrix, trained with InfoNCE across videos.
- Global alignment: CLCL scalar score. **Inference uses the fusion stream only.**
- Losses: L = L_t-v + α(L_t-p + L_t-r) + β·L_p-r, with α = 0.8 and β = 0.4. All terms are InfoNCE.
- Negative sampling: in-batch (batch 128).
- Uncertainty modeling: none.
- Training-only vs inference:
  - Training only: the pose and RGB auxiliary text losses and the pose-RGB matching loss.
  - Inference: the fused stream (it still needs the I3D and pose extraction).
- Reported results (test):
  - H2S (Tab. 1): T2V **62.5 / 75.1 / 80.1 / 1.0**; V2T **57.9 / 70.4 / 74.9 / 1.0**.
  - P14T (Tab. 2): T2V **76.8 / 91.7 / 95.3 / 1.0**; V2T **78.7 / 92.5 / 95.2 / 1.0**.
  - CSL (Tab. 3): T2V **85.8 / 94.4 / 95.6 / 1.0**; V2T **85.4 / 93.8 / 95.8 / 1.0**.
  - Single-stream ablation (Tab. 4). The column tick marks are lost in the PDF text; the assignment follows the paper's text ("Pose better on CSL, RGB better on P14T").

| stream | dataset | T2V R@1/5/10 | V2T R@1/5/10 |
|---|---|---|---|
| Pose | H2S | 55.9 / 69.6 / 74.9 | 50.9 / 64.7 / 69.3 |
| Pose | P14T | 65.0 / 86.4 / 92.1 | 65.3 / 86.0 / 92.2 |
| Pose | CSL | 80.5 / 90.9 / 94.0 | 80.0 / 89.5 / 92.9 |
| RGB | H2S | 54.3 / 68.8 / 74.4 | 48.3 / 62.6 / 68.7 |
| RGB | P14T | 70.4 / 88.6 / 94.4 | 70.1 / 88.8 / 94.5 |
| RGB | CSL | 75.4 / 88.3 / 92.4 | 73.5 / 87.7 / 92.3 |

  - Inconsistencies in the paper:
    - The text claims "+6.3" T2V R@1 over CiCo on H2S, but the table gives 62.5 - 56.6 = +5.9.
    - The P14T table labels the SLT-based "Translation" baseline row as "SA-SR".
- Evaluation protocol:
  - The paper does not describe checkpoint selection. Ablations (Tables 4-6, Fig. 5 α/β sweeps) are on the test set.
  - **[code]**: `main_task_retrieval.py` (master) evaluates the *test* dataloader every epoch and calls `save_best_model` when test R@1 improves. **Not disclosed in the paper.**
- Seeds: not reported.
- Code/checkpoints: code is released (2024-04). Processed I3D features, RTMPose keypoints and pretrained models were released on BaiduDrive (2025-04-27).
- Limitations: the RGB encoder stays frozen/offline; relies on an external pose estimator and external pretraining; single-run results; α/β were tuned on test.
- Solves: adds pose to reduce signer/background bias and recover hand detail; fuses the two streams through local cross-attention.
- Unsolved: visual confusability of similar signs; hubness/normalisation; honest validation protocol.

---

## 5. C2RL: Chen, Zhou, Huang, Wan, Hu, Shi, Liang, Lei, Zhang. "C2RL: Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieval." IEEE TCSVT 2025.

- Sources: https://arxiv.org/pdf/2408.09949 (v1, 19 Aug 2024). Published in IEEE TCSVT vol. 35(9), pp. 8533-8544, 2025, DOI 10.1109/TCSVT.2025.3553052 (Crossref).
- Datasets: P14T, CSL, H2S (CiCo filtering, 31,085 / 1,739 / 2,348, with Faster R-CNN crop), and OpenASL.
- Modalities: **RGB only** (raw frames, end-to-end).
- Extra supervision: **gloss-free**. Pretraining uses "only the corresponding dataset" (Fig. 1 caption): no external sign data, no pose. It does use ImageNet ResNet-18 and **MBart-large-cc25** (large multilingual text LM).
- Pretraining: stage 1 is C2RL pretraining on the target dataset.
  - ICL: CiCo-CLCL contrastive between the visual encoder and a text encoder.
  - ECL: a CoCa-like captioning/LM loss from a text decoder.
  - The loss is α·L_con + β·L_lm with α = β = 1. SGD, 200 epochs, batch 8×8.
- Video encoder: ImageNet ResNet-18 per frame, Conv1D-BN-ReLU (k = 3), FC, and a 3-layer Transformer (512-d). Frames are downsampled at k = 25%, with random frame per clip in training and first frame at test.
- Pose encoder: none.
- Text encoder:
  - Pretraining: a 3-layer text Transformer.
  - SLRet downstream: **two separate MBart encoders**, one over the (FC-projected) sign features and one over the text, with unshared parameters.
- Temporal representation: frame-token sequence.
- Local alignment / global alignment: CiCo CLCL fine-grained sign-word similarity with softmax aggregation (the paper reuses [3]).
- Losses: InfoNCE (Eq. 1) for SLRet fine-tuning. Adam, lr 1e-4, 80 epochs, batch 8×16.
- Negative sampling: in-batch.
- Uncertainty modeling: none.
- Training-only vs inference: the ECL decoder is used only in pretraining. Inference uses the visual encoder and both MBart encoders.
- Reported results (test; Table VI, **no MedR reported**):
  - P14T: T2V **78.7 / 92.2 / 94.9**; V2T **77.6 / 91.3 / 94.2**.
  - CSL: T2V **90.3 / 96.4 / 97.7**; V2T **88.4 / 95.7 / 97.1**.
  - H2S: T2V **62.4 / 75.9 / 80.1**; V2T **57.5 / 68.4 / 73.0**.
  - OpenASL: T2V 62.2 / 81.7 / 86.8; V2T 61.6 / 79.8 / 84.6. The text says "62.6 T2V", which is inconsistent with the table.
  - The paper does **not** compare against SEDS or UPRet; its SLRet baselines are SPOT-ALIGN v1 and CiCo.
  - Ablations on P14T test (Tab. VII): ICL only gives 74.7 / 73.5 R@1, ECL only gives 74.1 / 74.5.
- Evaluation protocol: "all results are reported on the test set". All ablations are on the P14T **test** set. Checkpoint selection is not described.
- Seeds: not reported.
- Code/checkpoints: no code link in the paper; none found on GitHub (UNVERIFIED that none exists).
- Limitations: SLRet uses two MBart-large encoders, which is far heavier than CLIP-B/32 and makes the capacity comparison unfair. No MedR. Test-set ablations. No released code.
- Solves: a strong **RGB-only, no-external-sign-data** representation. Joint contrastive and generative pretraining lifts P14T and CSL R@1 substantially.
- Unsolved: reproducibility; comparison to pose-based methods; fine-grained confusability.

---

## 6. "Scaling up Multimodal Pre-training for Sign Language Understanding": Zhou, Zhao, Hu, Li, Li. IEEE TPAMI 2025 (added via forward-citation search of SEDS/CiCo)

- Sources: https://arxiv.org/pdf/2408.08544 (v1; the PDF header shows a TCSVT template). Published in IEEE TPAMI vol. 47, pp. 11753-11767, 2025, DOI 10.1109/TPAMI.2025.3599313 (Crossref).
- Datasets for SL-RT: P14T and CSL. **H2S retrieval is not reported** (H2S is used only for SLT).
- Modalities: **pose only** (79 2D keypoints from HRNet+DarkPose/MMPose, plus InterWild hands).
- Extra supervision:
  - **Large external pretraining**: SL-1.5M, about 1.55M pose-text pairs. Sources: WLASL, MSASL, NMFs-CSL and SLR500 (isolated gloss labels turned into template sentences), P14T, CSL, How2Sign, Phoenix14 (unpaired), and BOBSL.
  - Whether the P14T and CSL **test** splits were excluded from SL-1.5M is **not stated (UNVERIFIED)**.
- Pretraining: masked pose modelling plus fine-grained sign-text contrastive learning against a **frozen MBart text encoder**.
- Video encoder: none (pose only).
- Pose encoder: manual and non-manual branch Transformers over pose embeddings.
- Text encoder: MBart encoder (frozen in pretraining). The downstream SL-RT text encoder is "another text encoder" with a CLIP-style loss.
- Local alignment: fine-grained similarity (Eq. 5; ablated in Tab. XVIII).
- Global alignment: contrastive.
- Losses: pose reconstruction and sign-text contrastive loss.
- Reported results (test; Table X):
  - P14T: T2V **74.5 / 93.3 / 95.6 / 1.0**; V2T **75.1 / 92.1 / 95.3 / 1.0**.
  - CSL: T2V **87.5 / 95.2 / 97.6 / 1.0**; V2T **87.2 / 95.0 / 97.2 / 1.0**.
  - It compares only to RGB methods (CiCo, SA-COMB); SEDS is not in Table X.
- Evaluation protocol: checkpoint selection not described. Ablations on the P14T / CSL test sets.
- Seeds: not reported.
- Code: not found / not verified.
- Class: pose-only with large external sign pretraining, so it belongs to comparison class (c).

---

## 7. SAN: Lee, Hur, Choi, Cho, Gaim, Hwang, Song, Lim. "Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative Mining for Sign Language Retrieval." ACL 2026 (Long).

- Sources: https://aclanthology.org/2026.acl-long.1302.pdf (pp. 28262-28277, DOI 10.18653/v1/2026.acl-long.1302); arXiv 2607.09263. Code: https://github.com/joonmy/SAN, whose README currently says **"Coming Soon"** (checked 2026-09-23).
- Datasets: **P14T only**.
- Modalities: RGB. There are two backbones:
  - **GFSLT-VLP**: the end-to-end RGB visual encoder of Zhou et al. ICCV 2023 (arXiv 2307.14768). SAN does not restate its backbone details.
  - **CiCo**: BSL-1K I3D features plus CLIP.
- Extra supervision: gloss-free.
  - The CiCo backbone inherits the external BSL-1K I3D.
  - The hard-negative miner uses a trained GFSLT-VLP retrieval model.
  - Comparative miners are FastText, XLM-RoBERTa and GPT-4o-mini.
- Local alignment: CLCL sign-word matrix.
- Global alignment: CLCL.
- Mechanism (training only):
  1. Reliable sign-word pairs: token softmax max-prob > α = 0.7.
  2. Visually similar negative words: sign-feature cosine > β = 0.7 and a different word.
  3. Keyword substitution generates hard-negative captions (N_swap = 2, N_hard = 5).
  4. L = L_coarse + λ·L_fine, where L_fine is a V2T InfoNCE over {t_i} ∪ H_i; λ = 0.4.
  - This is **not a false-negative filter**. There is no explicit filtering of synonyms that might actually be correct.
- Training: SGD lr 1e-2, cosine schedule, 100 epochs, batch 32 (GFSLT-VLP) or 256 (CiCo).
- Uncertainty: none.
- Inference: standard CLCL (SAN is training-time only).
- Reported results (P14T test, Table 1; **MRR instead of MedR**). Coarse-grained is the standard full test set.

| model | T2V R@1/5/10/MRR | V2T R@1/5/10/MRR |
|---|---|---|
| CiCo (their run, no hard negatives) | 69.2 / 87.2 / 92.2 / 77.3 | 70.1 / 87.7 / 92.9 / 78.2 |
| CiCo + SAN | 68.1 / 87.4 / 91.7 / 76.6 | 67.8 / 87.4 / 91.7 / 76.2 |
| GFSLT-VLP (no hard negatives) | 67.9 / 88.4 / 93.8 / 77.5 | 69.4 / 88.7 / 93.3 / 77.9 |
| GFSLT-VLP + SAN | **70.2 / 89.3 / 94.4 / 78.7** | **67.4 / 85.4 / 90.5 / 75.5** |

  - Fine-grained V2T R@1 is a new stress test: pick the original caption among 40 single-word-substituted hard negatives. It rises from 17.9 to 39.4 (CiCo) and from 16.8 to 49.1 (GFSLT-VLP).
  - Text-based miners (FastText, RoBERTa, GPT-4o-mini) are lower on fine-grained and hurt coarse V2T more.
- Evaluation protocol:
  - The coarse set is the standard P14T test split.
  - The fine-grained set is built on the test split. Target words are chosen with the **same sign-embedding similarity (β)** that SAN uses in training, which may favour SAN; the paper acknowledges the proxy.
  - Checkpoint selection is not described. Hyper-parameter sweeps (α, β, λ, N_swap, N_hard) are reported on test.
- Seeds: not reported.
- Limitations (stated): P14T only; fixed β threshold; residual linguistic noise; **slight coarse-grained trade-off** (e.g. CiCo+SAN V2T R@1 falls from 70.1 to 67.8).
- Solves: shows that hard negatives should be visually confusable rather than semantically confusable, and supplies a fine-grained diagnostic.
- Unsolved: SAN does not improve the standard benchmark (a mixed ±2 R@1). There is no false-negative control, no H2S/CSL evaluation, and no released code.

---

## 8. CMCM: Yang, Wei, Li, Hu. "Causality-inspired multi-grained cross-modal sign language retrieval." Computer Vision and Image Understanding, vol. 264, art. 104631, Feb 2026.

- Source: DOI 10.1016/j.cviu.2025.104631 (https://www.sciencedirect.com/science/article/pii/S1077314225003546). **Closed access.** ScienceDirect returned 403 or a Cloudflare block, OpenAlex shows OA = closed, and there is no arXiv preprint.
- Code: https://github.com/vddong-zjut/CMCM. The repo is **partial**: module files only (`modules/CCG_Module.py`, `CSA_Module.py`, `TMCP_Module.py`, `Encoder.py`), dataset loaders for H2S, PHOENIX and CSL, and no README, training/eval script or checkpoints.
- Method (from the search-engine abstract snippet and repo module names; not from full text):
  - Augmented views with backdoor adjustment (`CausalBackdoorAdjuster`) to remove confounders.
  - A cross-modal causal-attention Gaussian network using front-door intervention with Gaussian parameterisation (a μ/σ² KL term in `GaussianAlignmentModule`) for fine-grained alignment.
  - Temporal-motion covariance pooling (TMCP, MPN-COV) for the coarse global feature.
  - The repo `VideoEncoder` wraps torchvision `r2plus1d_18` (RGB), fused with a trainable branch.
- Datasets: "three public datasets" (loaders exist for H2S, PHOENIX and CSL).
- **All numbers: UNVERIFIED.** Full text is inaccessible, so no tables could be read. Do not cite CMCM numbers until the PDF is obtained.

---

## 9. Other 2024-2026 works checked (no comparable SLRet numbers on H2S/P14T/CSL)

- **Uni-Sign** (Li et al., ICLR 2025, arXiv 2501.15187): cites SEDS but reports ISLR/CSLR/SLT only. **No retrieval results** (full-text grep).
- **Sigma** (Pu et al., arXiv 2509.21223): ISLR/CSLR/SLT. No retrieval.
- **SignDino** (arXiv 2609.06296): no retrieval results found in the text.
- **SignCLIP** (Jiang et al., EMNLP 2024, arXiv 2407.01264): isolated-sign / Spreadthesign retrieval. Does not report sentence-level H2S/P14T/CSL retrieval.
- **"A Tale of Two Languages"** (Raude et al., arXiv 2405.10266): joint CSLR and sentence retrieval on **BOBSL**, a different benchmark.
- **GTRN** (Hu et al., Neurocomputing 637:130077, 2025): *video-query* sign corpus retrieval "in the wild". A different task.
- **SignSeek** (arXiv 2609.03695): sign *dictionary* retrieval. A different task.
- **Contrastive Pretraining with Dual Visual Encoders** (IVA 2025, arXiv 2507.10306): uses a retrieval-style InfoNCE for SLT pretraining. No retrieval results.
- Search coverage: arXiv API (abs:"sign language retrieval" returned 9 hits), OpenAlex, Semantic Scholar forward citations of CiCo and SEDS, and Crossref.
  - Semantic Scholar citations of UPRet were rate-limited and not obtained. A 2025-2026 SLRet paper that appears only in a venue without an indexed abstract could have been missed.

---

## 10. Adjacent mechanisms (general video/image-text retrieval)

The ◆ items are **inference-time score normalisations**. They re-score the whole test similarity matrix, or use a query bank, and can raise R@1 by several points with no change to the model.

| # | Mechanism | Citation | One-line mechanism | Used by published SLRet papers? |
|---|---|---|---|---|
| 1 ◆ | Dual Softmax (DSL) | Cheng et al., "Improving Video-Text Retrieval by Multi-Stream Corpus Alignment and Dual Softmax Loss", arXiv 2109.04290 (2021) | Multiply the similarity by a softmax prior over the opposite axis (S ⊙ softmax_dim0(S/τ)). At inference this is done over the **whole test gallery**, which uses test-set co-occurrence. | **No.** None of SPOT-ALIGN, CiCo, UPRet, SEDS, C2RL or SAN describes it. The CiCo, SEDS and UPRet code apply softmax only over tokens within a pair (τ = 0.07) and average the two directions (`dual_mix = 0.5`); there is no gallery softmax. |
| 2 ◆ | QB-Norm / Dynamic Inverted Softmax | Bogolin et al., "Cross Modal Retrieval with Querybank Normalisation", CVPR 2022 (arXiv 2112.12777) | Normalise gallery scores using a bank of training queries to suppress hubs. | **No** (not mentioned in any SLRet paper or in the inspected code). |
| 3 ◆ | Inverted softmax / CSLS | Smith et al., ICLR 2017 (arXiv 1702.03859); Conneau et al., "Word Translation Without Parallel Data", ICLR 2018 (arXiv 1710.04087) | Hubness correction by normalising each target's scores over all sources (IS) or subtracting mean k-NN similarity (CSLS). | **No.** Relevant because CiCo frames SLRet as cross-lingual retrieval, where CSLS is standard. |
| 4 | Token-wise late interaction | Yao et al., "FILIP", ICLR 2022 (arXiv 2111.07783) | Max over tokens, then mean (token-wise max similarity). | CiCo reports that the "Max" variant is worse than softmax (Table 6). UPRet uses weighted max. |
| 5 | Multi-grained matching | Ma et al., "X-CLIP", ACM MM 2022 (arXiv 2207.07285); Wang et al., "DRL: Disentangled Representation Learning for Text-Video Retrieval" (arXiv 2203.07111) | Video-sentence, frame-word, etc. matrices with attention-over-similarity aggregation (X-CLIP); weighted token-wise interaction and channel decorrelation (DRL). | CiCo's softmax-weighted clip-word aggregation is the same family. The UPRet Eq. 24 weighted max is DRL/X-Pool-like. |
| 6 | Probabilistic embeddings | Chun et al., "PCME", CVPR 2021 (arXiv 2101.05068); Chun, "PCME++" (arXiv 2305.18171); Fang et al., "UATVR", ICCV 2023 (arXiv 2301.06309) | Gaussian embeddings with a match probability / closed-form sampled distance for one-to-many correspondence. | UPRet (training-only Gaussian feature augmentation); CMCM (Gaussian KL, UNVERIFIED). Neither uses a probabilistic score at test. |
| 7 | Hard-negative mining | Faghri et al., "VSE++", BMVC 2018 (arXiv 1707.05612); Robinson et al., "Contrastive Learning with Hard Negative Samples", ICLR 2021 (arXiv 2010.04592); Yuksekgonul et al., "NegCLIP / bags-of-words", ICLR 2023 (arXiv 2210.01936); Chen et al., "Beyond Coarse-Grained Matching in Video-Text Retrieval", ACCV 2024 (arXiv 2410.12407) | Max-violation triplet (VSE++); importance-weighted hard negatives (Robinson); caption-perturbation negatives (NegCLIP, ACCV 2024 fine-grained test). | SAN (visual-similarity keyword substitution). Other SLRet papers use in-batch negatives only. |
| 8 | False-negative handling | Chuang et al., "Debiased Contrastive Learning", NeurIPS 2020 (arXiv 2007.00224); Huynh et al., "Boosting Contrastive SSL with False Negative Cancellation", WACV 2022 (arXiv 2011.11765) | Correct the InfoNCE denominator for positive-class contamination, or detect and remove likely false negatives. | **None of the SLRet papers.** This matters for P14T and CSL, which contain near-duplicate or templated sentences; SAN substitutes words without filtering valid paraphrases. |
| 9 | Optimal transport alignment | Cuturi, "Sinkhorn Distances", NeurIPS 2013 (arXiv 1306.0895) | Entropic OT plan between token sets as a soft alignment. | UPRet (training-only S_ot / L_D). |
| 10 | Temporal alignment (DTW-style) | Dvornik et al., "Drop-DTW", NeurIPS 2021 (arXiv 2108.11996); Yang et al., "TempCLR", ICLR 2023 (arXiv 2212.13738) | Order-aware sequence alignment with outlier dropping; DTW-distance contrastive loss. | **None.** CiCo argues sign-word order is not preserved; CiCo, SEDS and UPRet are order-agnostic. SEDS's CGAF is only locally order-aware. |

**Fair-comparison note.** No published SLRet result on H2S, P14T or CSL (SPOT-ALIGN, CiCo, UPRet, SEDS, C2RL, Scaling-up, SAN) reports using DSL, QB-Norm, inverted softmax or CSLS. Any new method that uses these at inference must report numbers with and without them. Without them the comparison is not apples to apples.

The bigger protocol issue lies elsewhere. The official CiCo, SEDS and UPRet code selects the best epoch by **test** T2V R@1, and this is not stated in any of the three papers. SPOT-ALIGN is the only SLRet paper that explicitly selects on validation.
