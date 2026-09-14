Literature cutoff date: **10 September 2026**

# A. Executive Decision

**Select Overlap-Constrained Evidence Matching (OCEM) for a gated research pilot. Do not yet commit to a SOTA or A*-acceptance claim.**

The strongest verified directions are cross-lingual local matching, richer sign representations, retrieval-plus-translation learning, probabilistic alignment, and sign-aware negative supervision. Their published results belong to different resource settings. In particular, C²RL, SEDS and the pose-pretrained SL-1.5M model cannot be collapsed into a single fair leaderboard with CiCo/UPRet. CMCM's complete results remain unverified. [CiCo](https://arxiv.org/abs/2303.12793), [UPRet](https://arxiv.org/abs/2405.19689), [SEDS](https://arxiv.org/abs/2407.16394), [C²RL](https://arxiv.org/abs/2408.09949), [Scaling up](https://arxiv.org/abs/2408.08544), [CMCM](https://doi.org/10.1016/j.cviu.2025.104631).

The clearest observed problem is **fine discrimination that does not reliably transfer into better standard bidirectional ranking**. SAN's results establish the distinction; they do not establish that any particular alignment defect causes it. The leading explanation to test is that independent token matching can assign excessive support to several query tokens using strongly overlapping visual windows. This explanation remains a **HYPOTHESIS**, not an observed SLRet finding. [SAN, Table 1](https://arxiv.org/html/2607.09263v1).

OCEM is designed to retain the contextual retrieval model after successful reproduction and add a small local scoring branch. Its assignment budget is defined over the actual time support shared by local visual windows. An explicit null state accommodates unmatched text. The same optimized score is used during training and retrieval. The method requires neither pose, new gloss annotations, synthetic captions, an LLM, nor SEDS artifacts.

The novelty is deliberately narrow. Optimal transport, partial matching, null assignments, coverage, and capacity constraints are established. The proposed contribution is **a shared-support constraint for retrieval from densely overlapping sign-video measurements**, with an empirical explanation that must survive ordinary partial-OT and parameter-matched controls. No claim of a new OT solver is made.

The raw-data route is credible but not fully certified: the complete Oxford I3D checkpoint and all three How2Sign annotation files downloaded and were hashed; raw-video archive checks remain partial. How2Sign's earlier test-annotation quota block is resolved. Model loading, exact usable-video manifests and baseline reproduction remain pending. SEDS/Baidu resources remain **UNAVAILABLE / DO NOT DEPEND ON**.

**Probability assessment:** 30–50% that a correctly implemented OCEM will beat the strongest matched fair baseline on mean bidirectional R@1 across two datasets, before seeing a real pilot. This is subjective judgment, not a statistical estimate. Reconsidering the other four candidates did not justify a 70–85% band. The sensible commitment is a cheap, falsifiable pilot, with a strict stop rule if the proposed failure is absent or ordinary OT explains the gain.

**Work completed:** source review, numerical reconstruction, protocol/resource audits, five candidates, adversarial novelty search, simulated reviews, mathematical specification, tested CPU scorer, full Oxford checkpoint transfer and all H2S annotation checks. **Not completed:** full dataset download, neural baseline reproduction, GPU training, or any empirical retrieval improvement.

# B. Search Methodology

The review began on 9 September and was resumed from Research State Checkpoint 3 on 10 September 2026. The cutoff was advanced with a targeted date check; completed searches were not restarted.

**Sources searched:** arXiv abstracts/full text/PDFs; CVF proceedings and project pages; ACL Anthology and the ACL 2026 accepted-paper list; publisher/DOI/Crossref records; author institutions; official or paper-linked repositories; dataset portals and download scripts. SciSpace and Consensus supported discovery; claims below rely on primary sources. Full-text access failed for CMCM and GTRN. Their missing details remain unknown.

**Field queries:** “sign language retrieval”; “sign language video retrieval”; “text to sign video retrieval”; “cross modal sign language retrieval”; “sign video text retrieval”; “free-form sign language retrieval”; “fine-grained sign language retrieval”; “sign language representation learning retrieval”; exact seed titles; 2025/2026 variants; How2Sign/PHOENIX/CSL retrieval; title/DOI/repository combinations.

**Inclusion:** sentence-level T2V/V2T, materially relevant visual-query retrieval, and adjacent mechanisms that can invalidate or inform a candidate. Recognition, translation and sign generation papers enter the method review only when they provide relevant alignment, representation, uncertainty or negative-supervision evidence. BLEU, sign recognition accuracy and dictionary retrieval are not relabeled sentence retrieval.

**Snowballing:** backward references in SPOT-ALIGN, CiCo, UPRet, SEDS, C²RL and SAN; CMCM's Crossref bibliography; forward title/DOI searches; author/repository follow-up; recent alignment and sign representation papers. This identified CSLR², the SL-1.5M pretraining paper and the downstream retrieval experiment in *Deep Understanding of Sign Language for Sign to Subtitle Alignment*, among others.

**Mechanism searches:** sign retrieval/translation, video-text retrieval and multimodal retrieval combined with coverage, overlap, receptive field, evidence reuse, partial transport, null alignment, redundancy, capacity, false negatives, counterfactual differences and relational/Gromov–Wasserstein matching. The novelty audit was repeated after VTaMo, DualAnchor and AVIOT were found.

**Evidence labels used throughout:**

- **FACT FROM SOURCE:** explicitly supported by the linked primary paper, code or dataset release.
- **MEASURED IN THIS AUDIT:** a local calculation or access check, with its scope stated.
- **INFERENCE / HYPOTHESIS:** an interpretation or proposed research claim.
- **U = UNVERIFIED — DO NOT USE AS FACT.** A missing result or recipe is never filled from another paper's leaderboard.

The search is systematic and broad, but it cannot certify that every relevant publication has been found. An empty search result is not evidence of novelty. No external contacts were messaged.

# C. Verified Literature Map

The following tables together form the paper records. Results are in D; implementation/resource details are in F. “Target gloss-free” does not imply absence of external sign supervision.

| Paper, exact title and verified venue | Task / datasets | Inputs, encoders, pretraining and extra resources |
|---|---|---|
| **Sign Language Video Retrieval with Free-Form Textual Queries** — Duarte et al., CVPR 2022 | T2V/V2T; How2Sign, PHOENIX-2014T | RGB I3D; spotting resources; cross-modal text embedding, including GrOVLE in the main English experiments; sign-recognition outputs for fusion. [Paper](https://arxiv.org/abs/2201.02495) |
| **CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning** — Cheng et al., CVPR 2023 | T2V/V2T; H2S, P14T, CSL-Daily | RGB I3D, target pseudo-label adaptation; CLIP-initialized visual temporal/text Transformers. Paper names BSL-1K; released extractor uses `bsl5k.pth.tar`. [Paper](https://arxiv.org/abs/2303.12793), [code](https://github.com/FangyunWei/SLRT/tree/main/CiCo) |
| **A Tale of Two Languages: Large-Vocabulary Continuous Sign Language Recognition from Spoken Language Supervision** — Raude et al., 2024 arXiv; final venue U | CSLR and T2V/V2T; BOBSL | CSLR²: Video-Swin-Tiny, Kinetics initialization and isolated-sign pretraining; frozen T5-large; automatic sign labels/timestamps and synonym resources. [Paper](https://arxiv.org/abs/2405.10266) |
| **Uncertainty-aware Sign Language Video Retrieval with Probability Distribution Modeling** — Wu et al., ECCV 2024 | T2V/V2T; H2S, P14T, CSL | CiCo-derived RGB features and CLIP; Gaussian modality embeddings. [Paper](https://arxiv.org/abs/2405.19689) |
| **SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval** — Jiang et al., ACM MM 2024 | T2V/V2T; H2S, P14T, CSL | RGB I3D + RTMPose; hand/body GCNs, SignBERT hand initialization, CLIP-based encoders. Pose and its pretrained estimators are additional resources. [Paper](https://arxiv.org/abs/2407.16394), [venue](https://doi.org/10.1145/3664647.3681237) |
| **Deep Understanding of Sign Language for Sign to Subtitle Alignment** — Jang et al., 2025 arXiv; final venue U | Alignment; downstream CSLR² T2V/V2T on BOBSL | I3D/BERT alignment framework; subtitle processing, manually aligned subset and self-training. [Paper](https://arxiv.org/abs/2503.03287) |
| **Graph traverse reference network for sign language corpus retrieval in the wild** — Hu et al., Neurocomputing 637, 2025, 130077 | Visual query → video corpus; dataset names U | GTRN; frame/body-part hierarchy and reference attention; exact backbones/resources U. [Author institution](https://ro.ecu.edu.au/ecuworks2022-2026/6028/), [DOI](https://doi.org/10.1016/j.neucom.2025.130077) |
| **C²RL: Content and Context Representation Learning for Gloss-Free Sign Language Translation and Retrieval** — Chen et al., TCSVT 35(9), September 2025, 8533–8544; preprint 2024 | SLT and T2V/V2T; H2S, P14T, CSL, OpenASL | ImageNet ResNet18; temporal convolution/Transformer; target-pair representation learning; two independent mBART-large-cc25 encoders downstream. [Preprint](https://arxiv.org/abs/2408.09949), [journal](https://doi.org/10.1109/TCSVT.2025.3553052) |
| **Scaling up Multimodal Pre-training for Sign Language Understanding** — Zhou et al., TPAMI 47(12), December 2025, 11753–11767 | Multiple sign tasks; sentence retrieval on P14T/CSL | Pose representations, manual/non-manual streams; external SL-1.5M pretraining. [Paper](https://arxiv.org/abs/2408.08544), [journal](https://doi.org/10.1109/TPAMI.2025.3599313) |
| **Causality-inspired multi-grained cross-modal sign language retrieval** — Yang, Wei, Li, Hu, CVIU 264, February 2026, 104631 | Cross-modal SLRet; full evaluation settings U | CMCM; exact published backbones/pretraining U. A paper-named repository has H2S/P14T/CSL loaders, which does not verify reported experiments. [Publisher](https://doi.org/10.1016/j.cviu.2025.104631), [repository](https://github.com/vddong-zjut/CMCM) |
| **Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative Mining for Sign Language Retrieval** — Lee et al., ACL 2026 long papers | Standard T2V/V2T and synthetic-caption fine V2T; P14T | SAN on CiCo/GFSLT; pretrained GFSLT mining; local sign–word correspondence and visually confusable replacements. [Paper](https://arxiv.org/abs/2607.09263), [ACL acceptance](https://2026.aclweb.org/program/accepted_papers/) |

| Method | Global/local alignment; temporal granularity | Negative strategy / uncertainty / loss roles |
|---|---|---|
| SPOT-ALIGN | Sentence embeddings; sliding sign clips; sign-recognition/text fusion | Contrastive ranking; iterative spotting supplies sign supervision; no explicit uncertainty model |
| CiCo | Contextual clip×BPE similarities; independent row/column softmax-weighted aggregation | In-batch negatives; symmetric InfoNCE on both directional matrices; pseudo-label cross-entropy adapts I3D |
| CSLR² | Contextual sentence and sign-level spaces; global pooling | HN-NCE reweights negatives; sign-level objectives add lexical discrimination; no explicit distributional uncertainty reported |
| UPRet | Distributional local matching; contextual clips/tokens | Gaussian sampling and transport alignment; retrieval contrast. OT score is training-only in §3.6, Eq.24 |
| SEDS | Global cross-modal retrieval plus local semantic fusion and pose↔RGB matching; clips/body parts | In-batch contrast across streams; local semantic attention; no explicit probabilistic uncertainty module |
| Jang et al. | Frame/subtitle alignment followed by global temporal alignment | Selective alignment loss suppresses absent-query supervision; self-training improves subtitle timing |
| GTRN | Hierarchical visual-reference matching | Contrastive joint space; detailed losses, negative policy and uncertainty U |
| C²RL | ICL: CLCL alignment; ECL: autoregressive contextual learning; temporal convolutions/attention | Contrastive plus translation cross-entropy; downstream CLCL; no explicit uncertainty model |
| SL-1.5M model | Pose/text representation alignment | Contrastive pretraining plus masked pose modeling; external-data setting |
| CMCM | Publisher highlights: augmentation-based backdoor adjustment, Gaussian alignment, temporal-motion covariance | Complete loss, mining policy and inference score U; do not infer them from isolated code modules |
| SAN | High-confidence local sign–word mining, then caption substitutions | Visual-token similarity plus different lexical identity; coarse loss + fine V2T contrast; no explicit false-equivalence probability |

The mechanism descriptions above are **FACT FROM SOURCE**, limited to the cited versions. The following weaknesses and dependencies are **OUR ANALYSIS**:

| Method | Dependency / what it advances | Remaining question relevant to this project |
|---|---|---|
| SPOT-ALIGN | Sign spotting + joint embeddings | Does global fusion resolve near-identical alternatives? |
| CiCo | Pretrained spotting features + CLIP + CLCL | Independent local aggregation does not explicitly couple the total support used across text tokens |
| CSLR² | Weak sign labels + joint recognition/retrieval | Gains cannot be attributed to sentence-only supervision |
| UPRet | CiCo plus distributional matching | Does modeling distributions resolve correlated temporal support at inference? Unshown |
| SEDS | Multi-stream sign representation and semantic fusion | Better cues need not eliminate a matching-rule failure; pose access complicates fair reproduction |
| Jang et al. | Subtitle alignment | Better training pairs and a better retrieval scorer are different interventions |
| GTRN | Visual graph/reference matching | Different query modality; no sentence-retrieval SOTA inference |
| C²RL | Retrieval contrast + generative context | Strong representations can still admit local near misses; this must be measured |
| SL-1.5M | Scale and pose pretraining | Not evidence for a same-resource algorithmic advantage |
| CMCM | Causal/distributional/multi-grained design | Full-text uncertainty blocks a detailed novelty or SOTA verdict |
| SAN | Local visual-hard negative supervision | Substitution validity and transfer from synthetic choices to standard galleries require separate tests |

Adjacent work materially affecting the proposal:

| Primary source | Relevant mechanism / consequence |
|---|---|
| [ColBERT](https://arxiv.org/abs/2004.12832) | Multi-vector late interaction is established; merely replacing global pooling is insufficient novelty |
| [Drop-DTW](https://arxiv.org/abs/2108.11996) | Joint alignment and dropping outliers; monotonic assumptions must not be copied blindly into sign/text matching |
| [CrossCLR](https://arxiv.org/abs/2109.14910) | Uses intra-modal similarity and excludes related negatives; generic false-negative removal is crowded |
| [Verbs in Action](https://arxiv.org/abs/2304.06708) | Calibrated hard captions and verb-phrase alignment; a residual-difference candidate must distinguish itself |
| [Norton](https://arxiv.org/abs/2401.16702) | Noisy-video correspondence, OT and unmatched-content handling already coexist |
| [SignCL](https://arxiv.org/abs/2405.14312) — NeurIPS 2024 | Adjacent-frame positives/distant-frame negatives; generic temporal feature separation is not a gap |
| [SignRep](https://arxiv.org/abs/2503.08529) | Skeleton-informed RGB pretraining and style-adversarial regularization; dictionary retrieval, not the standard sentence table |
| [SHuBERT](https://arxiv.org/abs/2411.16765) | Sign representation/translation lead; not verified as a new standard SLRet result here |
| [Lost in Translation, Found in Embeddings](https://arxiv.org/abs/2512.08040) | Joint translation/subtitle alignment with keypoints, lip images and multilingual pretraining; extra-resource route |
| [SEA](https://arxiv.org/abs/2512.08094v2) — ACL 2026 | Pretrained segmentation + embeddings + dynamic programming; generic boundary correction already addressed |
| [VTaMo](https://arxiv.org/html/2607.09126v1) | Uniform-marginal OT with null token and temporal regularization; target-guided reordering during training |
| [DualAnchor](https://arxiv.org/pdf/2607.27614v1) | Partial OT with dustbins and language-prior anchoring; OT removed at inference. “Retrieval-quality” diagnostics are not standard R@K results |
| [AVIOT](https://arxiv.org/pdf/2608.20473v1) | Balanced transport from video observations into compact supports, with query-conditioned allocation; narrows redundancy/transport claims |
| [Capacity-constrained OT](https://arxiv.org/abs/1307.7774), [coverage in NMT](https://arxiv.org/abs/1601.04811), [Gromov–Wasserstein learning](https://arxiv.org/abs/1901.06003) | Mathematical constraints, coverage and relational matching are prior art, not inventions of this proposal |

# D. SOTA Reconstruction

All entries are **published results, not our reproductions**. Tuples are **R@1 / R@5 / R@10 (%)**. A common dataset name alone does not establish experimental equivalence. These tables identify verified reference points, not an exhaustive ranking that excludes unknown CMCM results.

**D1. RGB input, external sign pretraining; no pose at retrieval input.** SPOT-ALIGN's resources and fusion differ from CiCo/UPRet; it is a historical reference within this broad family.

| Method | Dataset | T2V | V2T | MedR T/V |
|---|---|---:|---:|---:|
| SPOT-ALIGN SA-COMB | H2S | 34.2 / 48.0 / 52.6 | 23.6 / 47.0 / 53.0 | 8 / 7.5 |
| SPOT-ALIGN SA-COMB | P14T | 55.8 / 79.6 / 87.2 | 53.1 / 79.4 / 86.1 | 1 / 1 |
| CiCo | H2S | 56.6 / 69.9 / 74.7 | 51.6 / 64.8 / 70.1 | 1 / 1 |
| CiCo | P14T | 69.5 / 86.6 / 92.1 | 70.2 / 88.0 / 92.8 | 1 / 1 |
| CiCo | CSL | 75.3 / 88.2 / 91.9 | 74.7 / 89.4 / 92.2 | 1 / 1 |
| UPRet | H2S | 59.1 / 71.5 / 75.7 | 53.4 / 65.4 / 70.0 | 1 / 1 |
| UPRet | P14T | 72.0 / 89.1 / 94.1 | 72.0 / 89.4 / 93.3 | 1 / 1 |
| UPRet | CSL | 78.4 / 89.1 / 92.0 | 77.0 / 89.2 / 92.7 | 1 / 1 |

Sources: [SPOT-ALIGN](https://arxiv.org/abs/2201.02495), [CiCo official results](https://github.com/FangyunWei/SLRT/tree/main/CiCo), [UPRet tables](https://arxiv.org/html/2405.19689v1). UPRet reports MnR T/V of 54.4/76.4, 4.4/4.6 and 6.7/5.5 on H2S/P14T/CSL respectively. Its reproduced CiCo H2S V2T R@1 is 50.3, versus the original 51.6: a 1.3-point discrepancy already exceeds the suggested reproduction tolerance. A secondary table transcribes SPOT P14T T2V R@10 differently; the original 87.2 is retained.

**D2. RGB + pose, extra estimator/hand-model resources.**

| Method | Dataset | T2V | V2T | MedR T/V |
|---|---|---:|---:|---:|
| SEDS | H2S | 62.5 / 75.1 / 80.1 | 57.9 / 70.4 / 74.9 | 1 / 1 |
| SEDS | P14T | 76.8 / 91.7 / 95.3 | 78.7 / 92.5 / 95.2 | 1 / 1 |
| SEDS | CSL | 85.8 / 94.4 / 95.6 | 85.4 / 93.8 / 95.8 | 1 / 1 |

[SEDS source tables](https://arxiv.org/abs/2407.16394). Reported-only comparator: unavailable Baidu features/checkpoints constrain exact reproduction.

**D3. Target-pair sign representation learning, generic ImageNet/text pretraining, different training/backbone regime.**

| Method | Dataset | T2V | V2T |
|---|---|---:|---:|
| C²RL | H2S | 62.4 / 75.9 / 80.1 | 57.5 / 68.4 / 73.0 |
| C²RL | P14T | 78.7 / 92.2 / 94.9 | 77.6 / 91.3 / 94.2 |
| C²RL | CSL | 90.3 / 96.4 / 97.7 | 88.4 / 95.7 / 97.1 |
| C²RL | OpenASL | 62.2 / 81.7 / 86.8 | 61.6 / 79.8 / 84.6 |

[Verified preprint Table VI](https://arxiv.org/html/2408.09949v1). OpenASL prose says 62.6 T2V R@1 while the table says 62.2; use the table with this discrepancy disclosed. Journal-version numerical agreement remains U. MedR/MnR not established here.

**D4. Pose input with large external sign-text pretraining.**

| Method | Dataset | T2V | V2T | MedR T/V |
|---|---|---:|---:|---:|
| Scaling up / SL-1.5M | P14T | 74.5 / 93.3 / 95.6 | 75.1 / 92.1 / 95.3 | 1 / 1 |
| Scaling up / SL-1.5M | CSL | 87.5 / 95.2 / 97.6 | 87.2 / 95.0 / 97.2 | 1 / 1 |

[Source](https://arxiv.org/abs/2408.08544). A How2Sign translation table must not be treated as retrieval evidence.

**D5. SAN's standard and fine protocols remain separate.**

| P14T model | Standard T2V R@1/5/10 | Standard V2T R@1/5/10 | Fine V2T R@1/5/10 |
|---|---:|---:|---:|
| CiCo in SAN | 69.2 / 87.2 / 92.2 | 70.1 / 87.7 / 92.9 | 17.9 / 55.3 / 79.1 |
| CiCo + SAN | 68.1 / 87.4 / 91.7 | 67.8 / 87.4 / 91.7 | 39.4 / 75.4 / 92.5 |
| GFSLT in SAN | 67.9 / 88.4 / 93.8 | 69.4 / 88.7 / 93.3 | 16.8 / 53.1 / 78.0 |
| GFSLT + SAN | 70.2 / 89.3 / 94.4 | 67.4 / 85.4 / 90.5 | 49.1 / 85.9 / 94.9 |

[SAN Table 1](https://arxiv.org/html/2607.09263v1). Standard gallery: 642 paired samples. Fine task: one true caption against 40 constructed alternatives, ten from each of four generators. Fine MRR is 35.0→54.4 for CiCo and 33.9→64.1 for GFSLT. Neither a 41-choice score nor MRR is standard full-gallery T2V recall.

**D6. Other task/resource groups.**

| Setting | Method | T2V R@1/5/10 | V2T R@1/5/10 |
|---|---|---:|---:|
| BOBSL Sent-Val, ~2K gallery | CSLR² | 51.7 / 69.9 / 75.4 | 50.2 / 69.1 / 74.7 |
| BOBSL Sent-Test, ~20K gallery | CSLR² | 29.4 / 45.2 / 51.5 | 28.1 / 44.9 / 51.0 |
| BOBSL changed subtitle supervision | CSLR² + Jang alignment | 28.59 / 43.74 / U | 26.59 / 42.51 / U |

[CSLR²](https://arxiv.org/html/2405.10266v1), [Jang et al., Table II](https://arxiv.org/html/2503.03287v1). The latter paper's own reference is 27.14/42.19 T2V and 26.25/41.99 V2T. These within-paper changes do not supersede a different CSLR² recipe automatically. GTRN's visual-query results and CMCM's numbers remain **U** and are excluded from numerical ranking.

**Experimental Equivalence Matrix**

A = directly fair; B = mostly comparable but some execution/protocol differences unresolved; C = materially different input/supervision/resources; D = cannot support a SOTA claim.

| Comparison | Class | Reason / requirement |
|---|---|---|
| OCEM vs reproduced CiCo, identical cached features, manifests, text model, negatives and schedule | A, **planned** | Must actually execute matched runs; no present empirical A claim |
| OCEM vs matched ordinary partial OT/local head | A, planned | Same parameter count, initialization, data and candidate pool |
| Published CiCo vs published UPRet | B | Related backbone; manifests, reproduced baselines and recipes differ |
| Published SPOT vs CiCo | B/C | Historical protocol relationship, but spotting/pretraining/fusion differ |
| OCEM RGB vs published SEDS | C | Pose/SignBERT/estimator resources; exact artifacts unavailable |
| OCEM CiCo route vs C²RL | C | Different representation training, text encoder and external sign resources |
| OCEM vs SL-1.5M | C | Large extra pretraining and pose |
| SAN standard vs its fine gallery | D | Different candidates/task difficulty |
| OCEM vs SAN with a faithfully regenerated matched miner | A/B, conditional | Miner resources and generated negatives must be common and documented |
| Any method vs CMCM before full results/settings are obtained | D | Unknown numerical and resource equivalence |
| Original OpenASL result vs changed/depleted gallery | D | Candidate pool changed |
| Different methods sharing only a dataset name | D until audited | Counts alone do not identify the same examples |

After this audit, the first numerical reference is **CiCo reproduction**, followed by **UPRet under the matched RGB regime**. These are not selected as the universal strongest SLRet results.

# E. Dataset & Protocol Audit

**Identity and available supervision**

| Dataset | Identity / scale / signers | Splits and pairing | Native/released supervision versus derived resources |
|---|---|---|---|
| How2Sign | ASL; US instructional recordings; continuous; ~80 h; 11 signers overall, nine green-screen signers (paper: 8/5/6 train/dev/test) | Common reported original 31,164/1,740/2,356; CiCo filtered 31,085/1,739/2,348; sentence translation ↔ realigned clip | Raw multi-view RGB, translations, boundaries, metadata; released pose is automatically estimated. Paper discusses gloss, but a usable public gloss release is U. RTMPose is later processing |
| PHOENIX-2014T | DGS; German TV weather; ~11 h, 386 broadcasts, nine interpreters | 7,096/519/642; paired continuous sentence clips and German translations | Frame images, sequence gloss, cleaned spoken transcript/translation and sample metadata; no native dense pose assumed |
| CSL-Daily | Chinese Sign Language; controlled daily-life scenarios; ~23 h, ten signers, 20,654 samples | 18,401/1,077/1,176; sentence video/Chinese translation | RGB, translation, gloss sequence; no verified sign-level time boundaries or native pose |
| OpenASL | ASL; online news and vlogs; ~288 h, ~220 signers; 98,417 pairs | C²RL: 96,476/966/975; original paper's discussion uses 967/976 before filtering | URLs, subtitle/time metadata; downloaded RGB; automatic crop metadata. Not native gloss/pose |
| BOBSL / CSLR² retrieval | BSL broadcast interpretation; ~1,467 h, 39 signers | CSLR² trains on 689K aligned subtitle pairs; Sent-Val 1,973, Sent-Test 20,870 | Broadcast video/subtitles; manually realigned evaluation; training sign timestamps/labels derived by models. Six-hour CSLR sign test is a different evaluation; raw-data access/license route not fully certified |

Primary dataset sources: [How2Sign portal](https://how2sign.github.io/), [How2Sign paper](https://arxiv.org/abs/2008.08143), [PHOENIX official release](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/), [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/), [OpenASL repository](https://github.com/chevalierNoir/OpenASL), [OpenASL paper](https://arxiv.org/abs/2205.12870), [CSLR²](https://arxiv.org/abs/2405.10266).

**Additional pretraining corpora:** SL-1.5M combines WLASL, MSASL, NMFs-CSL, SLR500, PHOENIX14, PHOENIX14-T, CSL-Daily, How2Sign and BOBSL. Isolated-sign glosses become templated text; PHOENIX14 contributes unpaired pose; pose estimators provide derived annotations. These are pretraining resources, not nine comparable sentence-retrieval benchmarks. Their aggregate checkpoint/data release is not certified here and is not required by OCEM. [Collection procedure, §III-A](https://arxiv.org/html/2408.08544v1).

**Version and preprocessing forensics**

- H2S: original downloadable sentence clips use the older timing; recut full frontal recordings using manually realigned annotations for the CiCo route. The original dataset paper's green-screen split table differs from later release counts. SPOT reports 31,075/1,739/2,348; CiCo/C²RL 31,085/1,739/2,348; SEDS 31,019/1,738/2,348. Exact IDs, crop boxes and invalid-sample reasons must therefore be versioned. [Release script](https://raw.githubusercontent.com/how2sign/how2sign.github.io/main/download_how2sign.sh), [CiCo preprocessing](https://github.com/FangyunWei/SLRT/tree/main/CiCo/data_preparation).
- P14T: original frames are 210×260 at 25 FPS. Files under a directory called `features/fullFrame` are raw frame images, not an inaccessible model-specific feature package. C²RL's prose count 7,098 conflicts with the canonical 7,096 and its own total; do not silently adopt the typo. [Official P14T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).
- CSL: raw access requires a signed institutional agreement. Do not call it anonymously downloadable or substitute an unverifiable feature mirror. [Access policy](https://ustc-slr.github.io/datasets/2021_csl_daily/).
- OpenASL: the official downloader acknowledges disappearing source videos. A newly downloaded subset cannot reproduce the original full-gallery result without an exact availability manifest. [Official repository](https://github.com/chevalierNoir/OpenASL).
- BOBSL: gallery scale and sign/word supervision differ fundamentally from H2S/P14T. Do not mix its ~20K recall values with ~642/~2,348 galleries.

**MEASURED IN THIS AUDIT: How2Sign annotation snapshot, 10 September.** Official train, validation and test TSVs were downloaded, parsed with headers and hashed. They contain 31,165 / 1,741 / 2,357 unique sample-name rows, with no blank captions or nonpositive annotated durations. These are annotation-snapshot counts, **not claims about the canonical usable-video manifest**. Test annotations were inspected for protocol integrity only; they did not drive method selection.

| Quantity | Train | Validation | Test |
|---|---:|---:|---:|
| Unique raw caption strings | 30,109 | 1,516 | 1,938 |
| Rows belonging to a repeated-caption class | 1,509 | 444 | 777 |
| Duplicate excess: rows minus unique captions | 1,056 | 225 | 419 |
| Duplicate excess after lowercasing/whitespace normalization | 1,063 | 225 | 419 |

Forty-seven validation rows (33 distinct captions) and 71 test rows (44 distinct captions) have captions occurring verbatim in training. Fourteen distinct captions occur in both validation and test. All three pairwise `VIDEO_ID` intersections are zero. This does **not** establish signer separation, perceptual video uniqueness or the exact duplicate rate after paper-specific filtering. The earlier test quota response was resolved by an ordinary retry of the same official URL. Source URLs and SHA-256 hashes are preserved in the audit artifact. [Official download script identifying the files](https://raw.githubusercontent.com/how2sign/how2sign.github.io/main/download_how2sign.sh).

The original OpenASL paper independently reports training-caption overlap for 105/967 validation and 103/976 test examples. Its associated performance analysis concerns **translation**, not retrieval; no retrieval gain is inferred from it. [OpenASL paper](https://arxiv.org/abs/2205.12870).

**Retrieval protocol to preserve**

| Item | Required treatment |
|---|---|
| Pairing | Preserve official sample IDs and their paired translation; never infer IDs from sorted filenames alone |
| Gallery | Whole released test split for standard T2V and V2T; validation used for choices; no query-specific negative subsampling |
| Positives | Standard reported ID target retained. Exact-caption-equivalence recall may be reported separately, never substituted for standard recall |
| One-to-many meaning | Sentence translations need not uniquely identify videos. Repeated strings prove representational ambiguity, not complete semantic equivalence labels |
| Training duplicates | First reproduce the legacy loss. Then mask exact-caption off-diagonal negatives identically in all matched experiments; quantify this control's own gain |
| Invalid samples | Detect decode failure, missing frames, bounds outside duration, inconsistent IDs; log exclusions before training; identical manifest for all methods |
| Crop/FPS | Record original and decoded timestamps, crop boxes, resampling, normalization, window starts/ends and selection indices |
| Tie ranking | Deterministic gallery-ID tie order, identical across methods; additionally report expected/optimistic/pessimistic tie diagnostics |
| Metrics | R@1/5/10 separately by direction; MedR/MnR; optional MRR only explicitly labeled |

**Dataset Risk Matrix** — H/M/L indicate concern to investigate, not a measured leakage finding; U means insufficient evidence.

| Dataset | Signer overlap / background | Repeated text / templates | Timing / annotation noise | Near-duplicate or source leakage | Length/lexical shortcuts | Cross-paper incompatibility |
|---|---|---|---|---|---|---|
| H2S | H: mostly overlapping signer sets, studio background | H: measured repeated validation/test captions | M: manual realignment improves timing; old clips differ | U: no shared VIDEO_ID across snapshot splits; visual duplicates unaudited | M/H | H: several filtered counts |
| P14T | H: small signer pool and weather scene; signer-disjoint guarantee U | H concern: restricted weather language; repeat frequency U | M | U: broadcast/source grouping must be checked | H | M: raw release stable, preprocessing still matters |
| CSL | H concern: ten signers/controlled recording | H concern: scripted daily sentences; frequency U | M | U | H | M: access and preprocessing versions |
| OpenASL | H: source/channel and crop variation | H: published train/eval caption overlap | H: subtitles, noisy crops, missing videos | U: URL/source grouping and near duplicates | H | H: availability changes candidate pool |
| BOBSL | H: interpreter/background/channel priors | H concern: broadcast language/repeated context | H: audio/sign offsets and pseudo-label noise | U: episode/scene grouping required | H | H: evaluation subsets and alignment supervision |

# F. Resource Availability Audit

Access observations are date-stamped checks, not blanket guarantees. **Portal** = page/script verified; **range** = binary first 1,024 bytes verified; **full** = all bytes of the specified file downloaded and locally hashed. A range check does not prove full transfer, checksum validity, loadability or baseline equivalence.

| Baseline | Code | Checkpoints/features now | Regeneration / fallback | Reproduction status |
|---|---|---|---|---|
| SPOT-ALIGN | [Official project](https://imatge-upc.github.io/sl_retrieval/) | Complete original resource chain not certified | Rebuild spotting features if the same resources are obtained | Historical reported comparator |
| CiCo | [Official repository](https://github.com/FangyunWei/SLRT/tree/main/CiCo); preprocessing/extraction scripts read | Oxford I3D fully downloaded/hashed; CLIP range passed but full transfers timed out; target-adapted checkpoint not fully verified | Regenerate target pseudo-labels and adapted I3D; self-extract all RGB features | Preferred first reproduction; not yet passed |
| UPRet | [Official repository](https://github.com/xua222/UPRet) verified | Complete trained binaries/paths U | Reimplement/retrain on the same regenerated CiCo features | Strong matched competitor after CiCo gate |
| SEDS | [Official repository](https://github.com/longtaojiang/SEDS) inspected | **INACCESSIBLE ARTIFACT: Baidu checkpoints/features; UNAVAILABLE / DO NOT DEPEND ON** | Independently regenerate RGB/RTMPose and reconstruct architecture; label independently rebuilt setting | Full original reproduction constrained; published numbers only |
| C²RL | Author-verified operational release not located | Checkpoints U | ImageNet ResNet18 + mBART route technically reconstructible, but substantial recipe work | Reimplementation, not assumed exact reproduction |
| SAN | [Paper-linked repository](https://github.com/joonmy/SAN) README says “Coming Soon” | Runnable training release/miner checkpoint not established | Implement paper recipe with documented common miner; label SAN-style if not faithful | Published fine/coarse results verified, execution pending |
| CMCM | [Paper-named repository](https://github.com/vddong-zjut/CMCM) and modules downloaded | No complete recipe/weights verified; full paper inaccessible | Resolve author linkage, complete training entrypoint and text encoder before reproduction | D for numerical SOTA claims |
| CSLR² | Paper full text verified; complete artifact chain U | U | Rebuild only after BOBSL/pseudo-sign resources are available | Optional separate resource regime |
| SL-1.5M model | Complete official code/checkpoint route U | External corpus/model not certified | Not a dependency of OCEM | Reported external-data comparator |
| GTRN | Author institution page verified; full PDF 403 | U | Visual-query task; not required | No fabricated numerical comparison |

**Critical dependencies for the selected route**

| Dependency | Required? | Publicly accessible now? | Official source / model identifier | Can regenerate ourselves? | Fallback |
|---|---|---|---|---|---|
| PHOENIX raw v3 | Yes | Range passed; ~41.7 GB archive; full integrity pending | [phoenix-2014-T.v3.tar.gz](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz) | Decode/process original frames | Pause if original data cannot be obtained |
| H2S full frontal videos | Yes for second dataset | Official script; first 30 GiB split-part range passed; other parts/full integrity pending | [How2Sign](https://how2sign.github.io/), `train_raw_videos.z01` via release script | Recut official raw data | No unofficial processed features |
| H2S realigned annotations | Yes | All three files fully downloaded/parsed/hashed; earlier test quota resolved | [Release script](https://raw.githubusercontent.com/how2sign/how2sign.github.io/main/download_how2sign.sh) | Preserve original labels and IDs | Reconcile snapshot rows with the filtered paper manifest |
| Oxford sign backbone | Yes | Full 142,594,302-byte download and local SHA-256; model load pending | [bsl5k.pth.tar](https://www.robots.ox.ac.uk/~vgg/research/bslattend/data/bsl5k.pth.tar) linked by CiCo | Target adaptation can be retrained | ImageNet route changes setting; rerun all baselines |
| CLIP | Yes | HF range passed (605,247,071 bytes); full official and HF transfers timed out in this runtime | Exact CiCo model: [OpenAI `ViT-B/32` weights](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) | Fine-tune shared encoders; no new pretraining | [Official HF checkpoint](https://huggingface.co/openai/clip-vit-base-patch32); validate conversion/parity before claiming exact reproduction |
| Signer crop boxes | Yes for original CiCo crop protocol | Script dependency verified; per-sample archive completeness U | CiCo `bounding_box_coco_{split}.pkl` | Use a pinned public detector | Shared regenerated crops form a declared reconstructed setting |
| Optional crop detector | If boxes need regeneration | Official binary range passed: 167,502,836-byte object | [`FasterRCNN_ResNet50_FPN_Weights.COCO_V1`](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) | RGB person boxes, saved once | Fixed full-frame route for all models, with separate reproduction status |
| Adapted I3D | Yes for full CiCo | Released target weights U; not necessary to download | Official CiCo pseudo-label/trainer scripts | Yes, from train raw videos + Oxford model | No-adaptation ablation is a different baseline |
| Pose / SignBERT / SEDS features | No | SEDS artifacts unavailable | Not part of selected method | Not needed | None needed |
| ResNet18 fallback | No in main setting | Official range passed | [resnet18-f37072fd.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth) | Yes | Alternative resource track, never hidden substitution |
| mBART for C²RL comparison | No in main setting | Config range passed; weight transfer U | [facebook/mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) | Learned features can be rebuilt | Defer comparison until verified |

**Follow-up resource gate, 10 September.** Oxford I3D fully downloaded (142,594,302 bytes), with local SHA-256 `6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f`. This verifies complete transfer against the response length, not a publisher-signed checksum or model loading. All H2S annotation files are now complete; the test hash is `d1799dcbf100eda822eeb1a0e6e754d82b12b2a57756bd47073c4e4b8a02ae4d`.

CiCo's loader directly uses OpenAI `ViT-B/32`, whose official expected SHA-256 is `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`. Its format should be preferred for exact reproduction; the HF model requires mapping/parity checks. Full transfers from both official routes timed out in this runtime, so neither full CLIP transfer nor loading is certified. This is an observed timeout, not proof of permanent unavailability. Recorded repository heads: SLRT `38a4f7b00da7a858d59b7fabe5093876a84db8e0`; OpenAI CLIP `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. [CiCo loader](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/module_clip.py), [OpenAI model registry](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py).

The current workspace reported approximately 29 GiB free, below the 41.7 GB PHOENIX archive alone; no full raw-data staging was attempted. Provision adequate storage and a PyTorch training environment before the reproduction run. The experiment plan's larger storage requirements remain in force.

**Verified recipe fragments; omitted fields are U, not defaults attributed to a paper.**

| Method | Sampling / trainable modules | Optimization facts / unresolved items |
|---|---|---|
| CiCo | 16-frame stride-1 features, 224 input, up to 64 clips/32 text tokens; I3D adaptation then frozen feature extraction; retrieval Transformers trainable | Adam 1e−5, cosine, batch 512; sign adaptation SGD 0.01, batch 4, 15 epochs. README uses four processes; PH evaluation mixture 0.9, H2S/CSL 0.8. Exact retrieved release epochs/seeds U |
| UPRet | Same family of features; 64/32 tokens | Adam 1e−5, batch 512, 200 epochs, four A100s; probability/transport weights must be recovered from exact configuration |
| SEDS | 24 FPS, 16-frame windows, stride 1, 64/32 token limits; RGB extractor frozen | Batch 128, 200 epochs; Adam/cosine/warmup, 1e−4 pose/fusion and 1e−5 encoders; GPU/seeds U |
| C²RL | Frame quarter-sampling; 256 resize/224 crop; representation pretraining then frozen video features | Stage 1 SGD 0.01, 200 epochs, total batch 64; retrieval Adam 1e−4, 80 epochs, total batch 128; eight RTX3090s. Journal/source agreement pending |
| SAN | Local mining thresholds 0.7/0.7; generated substitutions | Fine-loss coefficient 0.4; SGD 0.01/cosine, 100 epochs; batches 32 GFSLT/256 CiCo; full release/config/seeds U |
| CMCM | U | Isolated module code is not a verified optimization recipe |

Recipe sources: [CiCo README](https://github.com/FangyunWei/SLRT/tree/main/CiCo), [UPRet](https://arxiv.org/html/2405.19689v1), [SEDS](https://arxiv.org/html/2407.16394v1), [C²RL](https://arxiv.org/html/2408.09949v1), [SAN](https://arxiv.org/html/2607.09263v1).

**Static code observations, not empirical allegations.** SEDS's inspected `compute_metrics` collects every rank tied with the diagonal and divides by the number of collected ranks. Variable tie multiplicity can therefore change per-query weighting. Verify the executed path before asserting an effect on any published result. [Inspected file](https://raw.githubusercontent.com/longtaojiang/SEDS/master/metrics.py). CMCM's inspected modules include an undefined `DEVICE` reference and apparent tensor-shape inconsistencies; its encoder variable named I3D instantiates R(2+1)D. These establish that the inspected fragments do not independently supply a runnable reproduction, not that the paper's experiments are invalid. [Modules](https://github.com/vddong-zjut/CMCM/tree/master/modules).

# G. Failure Taxonomy

**Observed failures and risks are distinguished from their possible causes.** The existence of an unmodeled quantity does not prove that it limits recall.

| Failure family | Evidence level | What remains unresolved / measurable implication |
|---|---|---|
| Fine sign discrimination | Direct SLRet evidence: SAN fine/coarse tables | Determine which real-gallery near misses share the diagnostic weakness; synthetic choices alone are insufficient |
| Dense/ambiguous local representations | Adjacent evidence from SignCL; direct representation quality gains in SLRet | Frozen RGB features may merge distinctions. A better scorer cannot recover handshape detail that was discarded by spatial/temporal sampling. [SignCL](https://arxiv.org/abs/2405.14312) |
| Temporal annotation mismatch | Direct timing-related retrieval evidence in SPOT and CSLR²/Jang experiments | Remaining headroom may be smaller on manually realigned H2S and trimmed P14T |
| Correlated local support | Architectural inference from overlapping clips and independent matching; effect unmeasured | Measure whether erroneous high-scoring pairs concentrate more support on overlapping intervals than correct pairs |
| Ambiguous positives | Exact duplicate captions measured in H2S; reported overlap in OpenASL | Semantic equivalence beyond exact strings is unknown. ID-only loss/metrics can conflict with a free-form semantic task |
| Negative distribution mismatch | Direct SAN evidence | Visually similar tokens are not necessarily incorrect semantic alternatives; spelling difference does not prove a negative label |
| Signer/background shortcuts | Dataset-design risk, not a measured current-model failure | Measure foreground/background and signer-conditioned errors before claiming causal disentanglement is needed |
| Relation/composition errors | Plausible; current SLRet-specific magnitude unverified | Need real-gallery cases with similar lexical content and different relations; random word reversal is not valid linguistic evidence |
| Reproduction artifacts | Direct manifest/access/code audit | A recall gain can arise from a changed gallery, timing, crop or tie policy; isolate these before algorithmic claims |

**Representation questions.** Handshape, movement, orientation, location and non-manual signals are distinct information sources. Pose can remove useful texture/finger detail or fail under occlusion; RGB can encode clothing/background as well as signing. These are representational trade-offs, not evidence that either modality always wins. The selected method changes evidence aggregation; it makes no claim to reconstruct lost finger detail, isolate phonemes, or produce linguistically complete sign units.

**Temporal questions.** A 16-frame feature is a measurement over an interval, not a detected sign. Adjacent windows overlap and boundaries can contain coarticulation. Spoken text order is not generally a valid monotonic sign alignment target. OCEM therefore uses known measurement supports, permits many-to-many soft allocation, and leaves language order modeling in the contextual encoders. It does not infer gold sign boundaries.

**Semantic questions.** One sign can express multiple textual tokens; a sentence can be paraphrased; different signing sequences can express similar meanings. Conversely, similar movements can carry different meanings. A hard one-word/one-sign assignment is unjustified. A capacity prior must be tested for harm on compressed translations and simultaneous cues, not defended as a conservation law.

**Uncertainty must not be collapsed into one Gaussian:**

| Type | Meaning | Existing coverage | Proposed scope |
|---|---|---|---|
| Visual | Occlusion, blur, hand/face ambiguity | Better representations; UPRet distributional modeling | Not directly solved by OCEM |
| Temporal | Clip/sign boundary or sampling uncertainty | Alignment work, window encoders | Supports are known; linguistic boundaries remain unknown |
| Linguistic | Paraphrase, morphology, multiple valid translations | Text/context learning; probabilistic approaches partly address ambiguity | Null mass avoids forcing every text token to a unique sign |
| Correspondence | Which observed region supports a query unit | Local matching/OT already established | Joint support load is the narrow intervention |
| Annotation | Wrong/misaligned/missing translation | Selective alignment and self-training | Preserve data; diagnose separately, no invented correction labels |

Generalization must be checked across signer, scene, camera, language, domain, length and video quality. Improvements on DGS weather alone do not establish broad sign-language understanding.

# H. Research Gap Matrix

G1 = observed failure; G2 = not already solved in the proposed form; G3 = relevance to recall; G4 = observable on existing data; G5 = algorithmically tractable. **Conditional G1** means the broad failure is observed but the proposed causal mechanism still needs Stage A. None of the new mechanisms is empirically confirmed by this desk study.

S/M/L below mean small/moderate/potentially large **hypothesized** upside, not predicted percentage points.

| Rank / gap | Evidence and exposing work | Partial solutions / unresolved part | G1–G5 | Existing data | T2V / V2T / fine upside | Novelty / experiment / resource risk |
|---|---|---|---|---|---|---|
| 1. Correlated overlapping support inflates near-miss scores | SAN establishes local failure; CiCo reveals independent aggregation | UPRet/VTaMo/DualAnchor already align tokens; shared raw-support budgeting not located | G1 conditional; G2 provisional; G3–5 yes | P14T, H2S | M / M / M–L if causal | M–H / M–H / L–M |
| 2. Fine discrimination damages coarse bidirectional ranking | SAN's standard/fine contrast | Better negative labels versus optimization interference unresolved | G1 yes for trade-off; G2 provisional; G3–5 yes | P14T, H2S | M / M / M | H / H / L–M |
| 3. Exact-boundary scoring is fragile | SPOT timing; CSLR²/Jang downstream alignment results | SAT/SEA already improve alignment; score-time uncertainty under fixed sample boundaries less clear | G1 yes in weak timing; G2 crowded; G3–5 yes | H2S, optional BOBSL/OpenASL | S–M / S–M / S | H / M / M |
| 4. Relations disappear despite matching local units | Indirect local ambiguity evidence only | CMCM/contextual encoders/GW already address structure broadly | G1 conditional; G2 high risk; G3–5 conditional | P14T/H2S real-neighbor diagnostics | M / M / M–L | H / H / L–M |
| 5. Ambiguous positives receive harmful negative gradients | Measured H2S duplicates; OpenASL source analysis | CrossCLR/Norton/UPRet overlap; reliable equivalence beyond exact duplicates unresolved | G1 duplication yes, harm unmeasured; G2 crowded; G3–5 yes | H2S/OpenASL; check P14T/CSL | S–M / S–M / S | H / M / L |
| 6. Missing hand/non-manual cues | SEDS; sign representation literature | Generic pose/RGB fusion already addressed | **Reject: G2 fails** for generic fusion | Existing RGB/pose-generation routes | Unscored | High collision / extra resources |
| 7. Signer/background invariance | Dataset risks; SignRep adjacent evidence | No measured current SLRet error magnitude; adversarial style learning exists | **Defer: G1 not established**, G2 crowded | Metadata-dependent | Unscored | High evidence risk |
| 8. Global-only alignment | Historical global model limitation | CiCo explicitly introduced local cross-lingual matching | **Reject: G2 fails** | — | — | Prior-art collision |
| 9. Semantically hard but visually easy negatives | SAN | SAN directly addresses this with local sign mining | **Reject: G2 fails** | — | — | Prior-art collision |
| 10. Generic distributional uncertainty | UPRet/CMCM | Already a contribution of existing methods | **Reject: G2 fails** | — | — | Prior-art collision |
| 11. Retrieval plus generative context / bigger encoder | C²RL | Already studied; scale alone prohibited | **Reject** | — | — | Forbidden/incremental |
| 12. Split, duplicate and metric inconsistencies | This protocol audit | Must be corrected experimentally | **Reject as primary contribution**: protocol obligation, not algorithm | All | — | Outside requested contribution |

The surviving gaps authorize **tests**, not a declaration that a major unsolved failure has been proven. If Stage A rejects rank 1's explanation, its high position in this matrix has no force.

# I. Five Candidate Methods

Exactly five candidates were considered. They share the public RGB/CLIP route unless explicitly stated. No candidate requires SEDS features, SEDS weights, pose, new human labels or test-set information.

**Candidate 1 — Overlap-Constrained Evidence Matching (OCEM).**

**Falsifiable hypothesis:** Existing local scores overrate some near misses because several query tokens draw support from overlapping measurements of the same short interval; constraining their aggregate support should improve R@1 when erroneous pairs are more concentrated than correct pairs.

**Core mechanism:** a support-aware assignment problem, with unmatched text mass, used as an actual retrieval score. Retain CiCo's contextual score; add a linear projection of pre-contextual I3D clips. Build a token-to-window similarity matrix, allocate text mass under shared-time inequalities, and contrast the resulting mixed score against real pairs. Temporal structure is the measured receptive field, not an assumed sign sequence. Full equations appear in N.

**Three closest papers and exact distinction:** CiCo independently aggregates local similarities; OCEM couples assignments. UPRet introduces distributional/transport learning; OCEM targets overlapping support and explicitly retains the optimized score at inference. VTaMo uses null-augmented OT and temporal alignment for translation; OCEM does not reorder tokens or decode language and adds constraints on overlapping measurements. These distinctions do not prove sufficient novelty; DualAnchor/AVIOT further limit the claim.

**Expected pattern:** moderate T2V/V2T R@1 upside only if the proposed failure is common; smaller R@5/10 changes; potentially larger fine-choice effects. No numerical improvement is predicted. It can fail because temporal concentration is legitimate, the local features lack discriminative information, or ordinary OT already supplies the benefit.

**Cost:** medium conceptual difficulty; medium engineering difficulty; 524,800 extra projection parameters in the I3D case; pairwise matrices and a convex dual solve. Training is materially cheaper than new visual pretraining but more expensive than plain scoring. Candidate status: selected conditionally after K/L.

**Candidate 2 — Rival-Conditional Evidence Scoring (RCES).**

**Hypothesis:** Shared sentence content masks the decisive difference between confusing candidates; explicitly learning residual evidence for that difference should improve R@1 when the differing content is visibly expressed.

**Mechanism/architecture:** contextual RGB and text tokens; softly align a true training caption to a real rival caption and compute unmatched token weights. Compare their residual visual support and train a pairwise preference predictor. A residual score can rerank V2T candidates, whose texts are available. For T2V, candidate-video captions are unavailable at test: a video/text student must learn the residual predictor from training pairs and operate without those captions. A margin objective plus the baseline contrastive objective trains it. Candidate-rival scoring uses a fixed top-K shortlist at inference, with the shortlist ceiling reported.

**Nearest differences:** SAN synthesizes caption substitutions; RCES uses actual paired captions. CSLR² adds sign-level pseudo-label supervision; RCES uses within-pair semantic differences without sign labels. Verbs in Action adds generated negatives and phrase objectives; RCES makes rival-dependent residual scoring central. The teacher/student asymmetry and pairwise comparisons may reduce it to known contrastive calibration.

**Expected pattern:** moderate V2T R@1, small-to-moderate T2V, small R@5/10, moderate fine-choice upside. The asymmetry is a major weakness: access to paired test captions would be prohibited, and a weak student may erase the benefit. Real rivals can also be paraphrases rather than negatives.

**Cost:** high engineering difficulty, medium-to-high conceptual complexity; extra preference/student heads; K-dependent scoring/memory and training. It is resource-feasible but less cleanly isolatable than C1.

**Candidate 3 — Boundary-Marginalized Retrieval (BMR).**

**Hypothesis:** Treating imperfect clip boundaries as exact contaminates retrieval with transitions or irrelevant content; marginalizing plausible boundaries should help when timing uncertainty is substantial.

**Mechanism/architecture:** encode five deterministic spans entirely inside the existing sample: full clip, centered 90%, first 90%, last 90%, centered 80%. Let baseline span scores be s_r and a fixed prior place half the mass on the full clip. Use `tau_b log sum_r pi_r exp(s_r/tau_b)` plus the full-clip score. This induces a query-conditioned latent posterior without boundary labels. Train symmetric contrast; preserve full-span ranking through the baseline loss. No candidate-specific ground-truth caption is used at test.

**Nearest differences:** SPOT-ALIGN repairs timing upstream; BMR integrates uncertainty in the retrieval score. SEA predicts alignment from segmentation and embeddings; BMR uses no new aligner or annotations. Norton addresses noisy video correspondence with richer alignment; BMR restricts the intervention to within-sample boundaries. This remains close to established multiple-instance learning and test-time ensembles.

**Expected pattern:** small-to-moderate T2V/V2T R@1 on noisy timing, small on carefully realigned clips; modest R@5/10; uncertain fine-choice benefit. It fails if boundaries are already good, interior spans omit decisive signs, or five-view ensembling explains the result.

**Cost:** low-to-medium conceptual/engineering complexity, up to five encoder views or reusable local features; inference/storage increase. No extra frames beyond the official sample are allowed. It ranks second by decision utility but has less defensible novelty.

**Candidate 4 — Relation-Consistent Late Interaction (RCLI).**

**Hypothesis:** Matching the presence of units cannot distinguish their relationships; jointly matching local affinities and learned within-modality relations should improve compositional R@1 if relation errors are substantial.

**Mechanism/architecture:** RGB/text tokens produce low-rank relation matrices, for example `R^v_ik=u_i^T A_v u_k+psi(delta_time)` and `R^t_jl=y_j^T A_t y_l`. Optimize unary affinity plus a relational penalty `sum P_ij P_kl (R^v_ik−R^t_jl)^2`. Learn the relation heads with contrastive pair supervision and use the optimized score at retrieval. Text order is not forced to match signed order; no dependency parser is introduced.

**Nearest differences:** CiCo lacks an explicit pair-of-correspondences term. CMCM models multiple granularities and temporal motion, but its full relation formulation is U. Gromov–Wasserstein learning already aligns relational graphs; the proposed application needs a sign-specific relation argument, not a renamed solver.

**Expected pattern:** moderate R@1 and potentially larger compositional diagnostic effects if relations are identifiable; small recall-tail effects. It may learn arbitrary relation matrices, collapse to unary similarity or be unstable. Without relation annotations, causal interpretation is particularly weak.

**Cost:** high conceptual/engineering complexity; nonconvex inner optimization; dense relational terms can scale as O(M²m²) per pair. Accessible inputs do not make this a reliable research bet.

**Candidate 5 — Reliability-Bounded Positive Sets (RBPS).**

**Hypothesis:** ID-only negatives suppress genuine alternatives; conservative, uncertainty-bounded positive sets should improve retrieval when such conflicts are a substantial source of errors.

**Mechanism/architecture:** preserve the same RGB/text encoders and independent score. Exact-caption masking is a shared baseline correction. Beyond it, use out-of-fold bidirectional agreement from models trained only on training folds to estimate candidate positive confidence. Cap each added weight and total extra positive mass; optimize a weighted multi-positive contrastive loss. At inference use ordinary embeddings with no new labels or cross-fitting ensemble.

**Nearest differences:** UPRet models embedding uncertainty; RBPS addresses label/positive-set uncertainty. CrossCLR excludes related negatives; RBPS would require a demonstrably calibrated positive-risk constraint. Norton already rectifies noisy cross-modal targets, making a general soft-label claim weak.

**Expected pattern:** small-to-moderate standard R@1, small recall-tail changes, uncertain or negative fine discrimination if near synonyms collapse. Exact-ID evaluation may conflict with the semantic objective. Agreement can reinforce mistakes, especially on repeated templates.

**Cost:** moderate complexity, ordinary inference, extra training folds/teachers. The simplest duplicate fix is useful but is not an A*-level primary contribution.

**Per-candidate dependency audit** — shared rows are expanded explicitly to avoid hiding a candidate-specific resource.

| Candidate | Dependency | Required? | Accessible now? | Official source | Regenerable? / fallback |
|---|---|---|---|---|---|
| C1 | Raw P14T/H2S, annotations | Yes | P14T/H2S raw range; all H2S annotations fully downloaded | Dataset links in F | Yes preprocessing; exact filtered manifest pending |
| C1 | Oxford I3D, CLIP, crop route | Yes | Oxford backbone full; CLIP full transfer unresolved; crop completeness pending | CiCo/Oxford/OpenAI/HF in F | Yes target adaptation/features; altered backbone means new comparison group |
| C1 | Supports and assignment solver | Yes | Generated locally; reference scorer runs | Our code; NumPy/SciPy | Yes; GPU port is engineering work |
| C2 | Same raw/encoder stack | Yes | Same bounded checks | F | Yes; no SEDS |
| C2 | Rival captions and residual teacher | Yes, training only | Existing train pairs | Original annotations | Train ourselves; test-caption dependence is disallowed |
| C3 | Same raw/encoder stack | Yes | Same bounded checks | F | Yes; no external alignment checkpoint |
| C3 | Within-clip span variants | Yes | Computable from raw sample | Original timestamps | Deterministic; full-span fallback loses candidate mechanism |
| C4 | Same raw/encoder stack | Yes | Same bounded checks | F | Yes; no pose/parser |
| C4 | Relation heads/GW optimizer | Yes | Algorithms public; training unvalidated | GW source in C | Implement ourselves; unary fallback is the baseline |
| C5 | Same raw/encoder stack | Yes | Same bounded checks | F | Yes; no LLM paraphrases |
| C5 | Out-of-fold confidence models | Yes | Trainable from existing pairs | No external checkpoint | Exact-duplicate-only control, with novelty abandoned |
| C1–C5 | SEDS/Baidu checkpoints/features | **No** | **UNAVAILABLE / DO NOT DEPEND ON** | Not required | Self-generated RGB throughout |

No candidate requires an inaccessible checkpoint as a hard method dependency. Full data availability and reproduction still gate any empirical claim.

# J. Candidate Scorecard

Scores are **subjective 1–10 planning judgments**, not measurements. The score deliberately values reproducibility and isolation, not just conceptual novelty.

| Criterion | C1 OCEM | C2 RCES | C3 BMR | C4 RCLI | C5 RBPS |
|---|---:|---:|---:|---:|---:|
| Scientific novelty | 7 | 7 | 5 | 6 | 4 |
| Novelty defensibility | 6 | 5 | 4 | 4 | 3 |
| SLRet-specific motivation | 8 | 8 | 7 | 8 | 7 |
| Evidence strength | 5 | 7 | 8 | 4 | 7 |
| Expected retrieval improvement | 7 | 7 | 5 | 7 | 5 |
| R@1 potential | 8 | 8 | 5 | 7 | 5 |
| Cross-dataset potential | 7 | 6 | 7 | 6 | 7 |
| Fair-comparison feasibility | 9 | 7 | 8 | 8 | 9 |
| Implementation reliability | 7 | 5 | 8 | 4 | 8 |
| Resource accessibility | 8 | 8 | 8 | 8 | 8 |
| Ablation clarity | 9 | 7 | 9 | 6 | 9 |
| Reviewer appeal | 7 | 7 | 5 | 7 | 5 |
| Compute efficiency | 6 | 5 | 5 | 3 | 9 |

Use the geometric mean to penalize weak dimensions:

\[
U_c=100\left(\prod_{r=1}^{13}\frac{s_{cr}}{10}\right)^{1/13},\qquad
R_c=\max(0,U_c-\textstyle\sum_p p_{cp}).
\]

| Candidate | Unpenalized U | Applied risk penalties | Adjusted R | Rank |
|---|---:|---|---:|---:|
| C1 | 71.4 | Prior-art collision −8; solver engineering −3 | **60.4** | **1** |
| C3 | 62.6 | Prior-art collision −13; view-ensemble confound −3 | 46.6 | 2 |
| C2 | 66.0 | Collision −8; inference asymmetry −8; isolation −4 | 46.0 | 3 |
| C5 | 62.9 | Collision −16; confidence feedback −4 | 42.9 | 4 |
| C4 | 57.4 | Collision −10; unstable training −10; isolation −4 | 33.4 | 5 |

The requested fatal-risk screen is separate from these nonzero penalties:

| Fatal risk | Outcome |
|---|---|
| Unavailable annotations | None permitted; C2 test captions would be an immediate rejection |
| Inaccessible checkpoints/features / SEDS Baidu dependency | Zero accepted dependencies; any later introduction is a no-go |
| Too many modules | C2/C4 penalized; do not rescue them by adding more machinery |
| Fragile training | C4 heavily penalized; C1's convex score still needs a performant GPU implementation |
| Unfair SOTA comparison | No score can compensate; comparison is excluded or relabeled |
| Gain from larger backbone | All matched controls use the same backbone; a violation invalidates attribution |
| Cannot isolate contribution | C2/C4 penalized; C1 must beat ordinary partial OT and the same local head |

C1 remains first when revisiting the alternatives after reviewer criticism. Its 60.4 utility is **not** a 60.4% probability of success.

# K. Adversarial Novelty Check

Assume C1 is not novel. The following attempts constrain or invalidate broad versions of it.

| Attack / searched concepts | Closest primary evidence | Decision |
|---|---|---|
| “sign retrieval + optimal transport / uncertainty” | [UPRet](https://arxiv.org/abs/2405.19689) | Reject “first OT for SLRet” and generic probabilistic matching |
| “sign translation + partial OT / null / lexical fidelity” | [VTaMo](https://arxiv.org/html/2607.09126v1), [DualAnchor](https://arxiv.org/pdf/2607.27614v1) | Reject partial/null matching as the contribution |
| “video + redundancy / transport / support allocation” | [AVIOT](https://arxiv.org/pdf/2608.20473v1) | Reject generic transport-based evidence compression or query-conditioned capacity claims |
| “multimodal retrieval + unmatched content / noisy OT” | [Norton](https://arxiv.org/abs/2401.16702) | Reject generic missing-content handling |
| “coverage / repeated attention” | [Coverage NMT](https://arxiv.org/abs/1601.04811) | Reject the claim that preventing repeated attention is new |
| “capacity-constrained optimal transport” | [Korman et al.](https://arxiv.org/abs/1307.7774) | No claim to new convex duality, capacity OT or a new solver |
| “sign representations + local ambiguity / density” | [SignCL](https://arxiv.org/abs/2405.14312) | Reject simple temporal repulsion or feature separation |
| “overlap-constrained / shared evidence / receptive-field transport” across SLRet, SLT and video retrieval | No exact shared-time inequality located in the inspected retrieval formulations | Narrow hypothesis survives; absence from searches does not establish novelty |

**Exact remaining distinction:** ordinary token OT constrains token marginal masses. OCEM additionally couples several visual columns through a known matrix mapping each local feature to its underlying observed time support. If two columns overlap, their combined load on shared atoms is constrained even when each individual column has unused capacity. This is not equivalent in general to assigning independent upper bounds to each column.

**Important narrowing:** this is a modeling prior about concentration, not a proof that two words cannot share a sign. Contextual Transformer outputs can contain information from the entire video; assigning them a 16-frame support would be false. OCEM therefore applies the support constraint only to the pre-contextual local branch.

The exact-duplicate invariance demonstrated by the reference scorer is useful for implementation validation. It is not independently novel: ordinary OT can also be invariant when duplicate reference masses are split correctly. OCEM does not claim invariance to arbitrary frame-rate changes, new resampled features or altered signing speed.

**Unresolved:** CMCM's complete paper and additional unpublished/unindexed work may still collide. A publication novelty claim remains conditional on obtaining CMCM's full method and a focused final pre-submission refresh. Do not present a negative search as certification.

# L. Reviewer Simulation

These are three separately reasoned simulated perspectives, not external reviews. Scores refer to a hypothetical submission with the evidence currently available.

| Reviewer | Score / confidence | Main rejection reasons | Required changes |
|---|---|---|---|
| A — Sign-language expert | 4/10; 4/5 | Duration is not semantic capacity; morphology and simultaneous non-manual information permit shared support; no empirical linguistic grounding | Avoid one-word/one-sign assumptions; constrain only local measurements; retain context; test compressed translations and harms; do not call transport cells sign boundaries |
| B — Retrieval expert | 4/10; 4/5 | OT/coverage are old; extra local head may explain gains; full-gallery transport may be slow; score/loss mismatch can obscure causality | Parameter-matched head, ordinary partial OT, independent caps and shuffled-support controls; exact same score in training/inference; latency and both directions |
| C — Hostile A* reviewer | 3/10; 5/5 | Failure mechanism unmeasured, baseline unreproduced, CMCM unknown, only synthetic checks; no evidence of cross-dataset or statistically reliable gains | Reproduction gate; validation error enrichment; two datasets/three seeds; paired uncertainty; resource manifests; kill if simple controls explain gain |

**Revisions made:**

1. Restrict physical support to local frozen I3D features; retain the contextual branch for semantics.
2. Use soft text mass and a null state, with adjustable capacity; explicitly treat capacity as a prior.
3. Center the score by its zero-affinity value to remove geometry-only offsets.
4. Preserve the same mixed score in the training loss and at inference.
5. Make ordinary partial OT, the local projection alone, duplicate masking and independent caps mandatory controls.
6. Use full-gallery evaluation for primary claims; approximate shortlisting is a separately labeled deployment experiment.
7. Require validation-only failure confirmation before a full training campaign; no fabricated SOTA target.

A hostile reviewer would still reject a results paper today. Better specification does not supply missing empirical evidence. The revised project is worth a **bounded pilot**, with acceptance contingent on the observations in W/X.

# M. Final Proposed Method

**Selected method: Overlap-Constrained Evidence Matching (OCEM).** All other candidates are alternatives rejected for this research commitment, not parallel components of the final architecture.

**Prospective paper thesis, conditional on experiments:** “We show that correlated local support inflates near-miss scores in sign-language retrieval, and introduce OCEM to address it by coupling token assignments through the shared time support of visual measurements.” The phrase “we show” is an intended paper claim; it is **not established by this report**. If the evidence does not support it, the thesis must be abandoned.

**Why this candidate:** it turns a specific possible error in scoring into a small, falsifiable intervention; it can preserve the strongest accessible baseline's representations; its inner optimization is convex; no new linguistic labels or unavailable features are needed. Its novelty and empirical evidence remain moderate risks. The attraction is testability and controllable implementation, not a guarantee of large gains.

**Architecture and tensors** — batch dimension B is omitted where a module is pair-specific. Let M≤64, m≤32 valid text tokens, local dimension p=1024 for the chosen I3D features, and retrieval dimension d=512.

| Module | Input → output | Operation / trainability | Purpose |
|---|---|---|---|
| Raw preprocessing | Frames/timestamps → cropped 16-frame windows and `[M,2]` supports | Original dataset/CiCo pipeline; frozen deterministic evaluation transform | Preserve reproducible temporal support |
| Domain-agnostic + adapted I3D | `[M,3,16,224,224]` → `H:[M,1024]` | Self-extracted, weighted feature combination; frozen during retrieval | Same visual evidence as matched baseline |
| Baseline visual context F | H → `Z:[M,512]` | Reproduced CiCo visual Transformer; internal width follows its configuration | Long-range semantic context |
| Baseline text context G | Token IDs → `Y:[m,512]` | Reproduced CLIP-initialized text encoder; padding/special tokens masked locally | Contextual query representation |
| New local projection | H → `U:[M,512]` | `normalize(H W + bias)`; trainable; **524,800 parameters** | Match local features without falsely assigning a local support to globally contextual tokens |
| Baseline scorer | Z,Y → two scalars `s_b^T,s_b^V` | Original masked directional aggregation | Retain global/contextual performance |
| Support geometry | Window intervals → A:`[L,M]`, w:`[L]`, q:`[M]` | Deterministic, no learned parameters; L≤2M−1 | Couple windows sharing observed time |
| OCEM scorer | U,Y,A,w,q → scalar φ and assignment P:`[m,M+1]` | Convex entropy-regularized assignment; no learned solver | Correct local support concentration |
| Final score | `s_b^d,φ` → `S^d` | Fixed validation-selected blend | Actual training and inference decision |

The local branch is **not** passed through the global visual Transformer. Otherwise the claimed raw-time support would be invalid. The text token can be contextual: the support constraint concerns what video measurements are used, not an assertion that textual tokens are independent linguistic units.

**No additional supervision:** baseline BSL-derived pretraining, generic CLIP pretraining and detector resources are explicitly inherited and shared with controls. Target pseudo-label adaptation uses training videos only. OCEM adds no pose, gloss, translation decoder, LLM, external sign corpus or generated negatives.

**Temporal interpretation:** the model knows the input windows' sampling geometry. It does not know sign boundaries. Text mass can be divided across several windows; several text tokens can use the same interval within a loose total budget; mass can be unmatched. There is no monotonic text/sign ordering constraint and no claim to represent all simultaneous phonological channels separately.

# N. Mathematical Formulation

The following equations specify **the proposed method**, not equations attributed to prior papers.

**N1. Problem and representations.** Let the training data be

\[
\mathcal D=\{(V_n,T_n,\mathrm{id}_n)\}_{n=1}^{N}.
\]

A video is a time-indexed frame sequence. Feature i is computed from an actual interval \(I_i=[s_i,e_i)\), with local frozen feature \(h_i\in\mathbb R^{1024}\). Sampling/padding metadata must record which original times contributed; padded copies are not extra independent observations.

\[
u_i=\frac{W_vh_i+b_v}{\|W_vh_i+b_v\|_2+\delta},\qquad
 y_j=\frac{G(T)_j}{\|G(T)_j\|_2+\delta},\quad \delta=10^{-8},
\]

\[
C_{ji}=y_j^\top u_i\in[-1,1],\qquad C\in\mathbb R^{m\times M}.
\]

Use valid non-special BPE tokens, not a new POS tagger. Padding/special positions have no local mass. An empty-token query uses the baseline score and is logged; do not silently change evaluation membership.

Let \(s_b^T(V,T)\) and \(s_b^V(V,T)\) be the **successfully reproduced baseline's** T2V/V2T aggregation scores. For orientation, CiCo's rule averages softmax-weighted token similarities along the appropriate axes, rather than imposing joint assignments. Exact normalization, scale and masks are frozen from the reproduction configuration; they are not guessed from a later reimplementation.

**N2. Disjoint time atoms.** Form the union \(\Omega=\cup_i I_i\), split it at every distinct endpoint, and retain nonempty covered atoms \(J_a\), \(a=1,\ldots,L\). Define

\[
A_{ai}=\frac{|J_a\cap I_i|}{|I_i|},\qquad
w_a=\frac{|J_a|}{|\Omega|}.
\]

Thus \(A_{ai}\ge0\), \(\sum_aA_{ai}=1\), and \(\sum_aw_a=1\). Gaps outside the observed union receive no mass; report the unobserved fraction rather than pretending those frames were encoded. Time is measured in seconds or exact frame-index units consistently.

**N3. Reference visual mass, including exact copies.** Group exactly equal intervals into canonical supports \(\bar I_g\), with multiplicity \(n_g\). For the coverage count over unique supports,

\[
c(t)=\sum_g\mathbf1[t\in\bar I_g],\qquad
\bar q_g=\frac{1}{|\Omega|}\int_{\bar I_g}\frac{1}{c(t)}\,dt,
\qquad q_i=\bar q_{g(i)}/n_{g(i)}.
\]

The piecewise-constant integral is computed exactly on the atoms. \(\sum_iq_i=1\). This reference avoids assigning additional prior mass merely because a feature is copied. It is a quadrature convention, not a semantic importance estimate.

For text, set \(b_j=1/m\). With fixed null prior \(0<\pi<1\), define

\[
R_{ji}=b_j(1-\pi)q_i\;(i\ge1),\qquad R_{j0}=b_j\pi.
\]

**N4. Shared-support feasible set.** Let \(P_{ji}\) allocate text mass j to window i and \(P_{j0}\) to null. With \(t_i=\sum_jP_{ji}\),

\[
\mathcal P_\kappa=\left\{P\ge0:
P_{j0}+\sum_{i=1}^MP_{ji}=b_j\ \forall j,
\quad A t\le\kappa w\right\}.
\]

\(\kappa>0\) controls permitted concentration. Start with \(\kappa=1.5\), then test a small validation grid. Null makes the problem feasible even when all matching mass must be rejected. No individual token or sign must map one-to-one.

The inequality is the core hypothesis: it penalizes excessive concentration relative to observed time. It does **not** assert that linguistic information is uniformly distributed in time. A failure on legitimate compact expressions would be evidence against the prior.

**N5. Optimized, centered score.** Null similarity is zero. For \(\epsilon>0\),

\[
F_\kappa(C)=\max_{P\in\mathcal P_\kappa}
\left\{\sum_{j,i\ge1}P_{ji}C_{ji}
-\epsilon\sum_{j,i\ge0}P_{ji}\log\frac{P_{ji}}{R_{ji}}\right\}.
\]

Because both P and R have total mass one, the omitted generalized-KL linear terms cancel. Use

\[
\boxed{\phi(V,T)=F_\kappa(C)-F_\kappa(0)}.
\]

Subtracting \(F_\kappa(0)\) prevents a video from receiving a geometry-only score offset when its reference distribution is infeasible. Geometry may still affect its response to real content, intentionally. With uniform b, the zero-affinity term can be cached by video geometry and scorer hyperparameters.

**N6. Convex dual and solution.** For nonnegative multipliers \(\mu\in\mathbb R_+^L\),

\[
z_j(\mu)=\pi+(1-\pi)\sum_iq_i
\exp\left[\frac{C_{ji}-(A^\top\mu)_i}{\epsilon}\right],
\]

\[
F_\kappa(C)=\min_{\mu\ge0}
\left\{g(\mu)=\epsilon\sum_j b_j\log z_j(\mu)
+\kappa w^\top\mu\right\}.
\]

Compute log z with log-sum-exp, including the null logit. Given μ,

\[
P_{ji}=\frac{b_j(1-\pi)q_i\exp[(C_{ji}-(A^\top\mu)_i)/\epsilon]}{z_j(\mu)},
\quad P_{j0}=\frac{b_j\pi}{z_j(\mu)},
\]

\[
\nabla_\mu g=\kappa w-A t.
\]

The reference implementation uses bound-constrained L-BFGS-B. A GPU implementation may use projected gradient or a batched constrained optimizer, but must match the reference optimum. A conservative projected-gradient step can use \(\eta\le\epsilon/(\|A\|_2^2+\delta)\); a practical accelerated/preconditioned version needs its own convergence checks. Ordinary balanced Sinkhorn is **not** a drop-in solver for these additional overlapping constraints.

For a numerical certificate, let \(r=At\), and scale real mass by

\[
\zeta=\min\left(1,\min_a\frac{\kappa w_a}{\max(r_a,\delta)}\right).
\]

Set \(\tilde P_{ji}=\zeta P_{ji}\) for real columns and transfer removed row mass to null. This gives a feasible primal lower bound; g(μ) is a dual upper bound. Require absolute capacity residual ≤10⁻⁵ and duality gap ≤10⁻⁴ during training, tightened to 10⁻⁶ for primary evaluation, in normalized score units. Increase iterations or flag failure; do not silently score different examples with inconsistent solver accuracy.

**N7. Gradients.** The entropy term yields a unique optimum for positive reference masses. At the optimum, the envelope theorem gives

\[
\frac{\partial F_\kappa}{\partial C_{ji}}=P^*_{ji}.
\]

A custom backward can therefore use the converged assignment, without retaining every solver iteration. This saves memory; it does not excuse inaccurate solutions. \(F_\kappa(0)\) has no encoder parameters. Do not use the gradient of the raw dot product alone while claiming to optimize the entropy-regularized value.

**N8. Inference score.** For direction \(d\in\{T,V\}\),

\[
\boxed{S^d(V,T)=(1-\gamma)s_b^d(V,T)+\gamma\phi(V,T)}.
\]

Start with \(\gamma=0.25\). It is fixed within a run and chosen only on validation. The local φ is shared across directions; the contextual baseline's directional scores remain distinct.

**N9. Complete training loss.** Let \(\ell_{\rm sym}(Z;\mathcal C)\) be symmetric InfoNCE over a candidate set/mask:

\[
\ell_{\rm sym}=-\frac1{2B}\sum_n
\log\frac{e^{Z_{nn}/\tau}}{\sum_{k\in\mathcal C_n^r}e^{Z_{nk}/\tau}}
-\frac1{2B}\sum_n
\log\frac{e^{Z_{nn}/\tau}}{\sum_{k\in\mathcal C_n^c}e^{Z_{kn}/\tau}}.
\]

Indices use video rows/text columns. Every candidate set contains the positive. First reproduce CiCo's original CLCL. For the matched corrected baseline and OCEM, the exact-caption exclusion mask is common. Define

\[
\mathcal L_{\rm base}=\tfrac12\ell_{\rm sym}(Z_b^T;\mathcal B)
+\tfrac12\ell_{\rm sym}(Z_b^V;\mathcal B),
\]

\[
\mathcal L_{\rm mix}=\tfrac12\ell_{\rm sym}(S^T;\mathcal C)
+\tfrac12\ell_{\rm sym}(S^V;\mathcal C),
\qquad
\boxed{\mathcal L=\mathcal L_{\rm base}+\lambda\mathcal L_{\rm mix}}.
\]

Use the reproduced temperature treatment; start with \(\lambda=1\). The base term preserves pressure for broad ranking. The mixed term trains the score actually used at test. There are no auxiliary translation, pose, causal or synthetic-negative losses.

**N10. Positive and negative construction.**

- Positive: only the original paired ID in each loss numerator. Exact duplicate-caption off-diagonals are excluded from negatives; they are not newly annotated positives. Apply the same rule to every matched control.
- Base pool: all valid in-batch pairs. No test examples or validation-mined training negatives.
- Mixed pool: positive plus up to eight high-scoring real negatives and eight uniform random real negatives per row and column; union the required pairs. Use a frozen reproduced baseline miner and fixed sampling seeds for critical ablations, so candidate selection itself does not explain gains.
- Semantic hardness diagnostic: cosine between pooled text embeddings. Visual hardness diagnostic: cosine between pooled frozen sign features. These diagnostics are not claims to recover sign phonology.
- False negatives beyond exact matches remain possible. Do not label paraphrases negative solely because a token differs. Report sensitivity to removing high semantic-agreement negatives; such removal must be shared across controls.
- Curriculum: projection-only warmup with uniform negatives; enable the fixed hard pool after the local head is stable. No adaptive sixth module is added.

**Comparison with SAN:** SAN changes training supervision using mined local visual confusability and word substitutions. OCEM changes the pairwise inference score, uses real paired candidates and does not assert that different words are necessarily incorrect translations. Random, semantic, visual/SAN-style and proposed scoring comparisons are required; mining alone is not the novelty claim.

**N11. Mathematical properties versus empirical claims.** Exact duplicate visual features with equal support leave φ unchanged when their prior mass is split. Permuting already computed text-token rows leaves φ unchanged; **permuting the input sentence need not**, because G is contextual. As \(\kappa\to\infty\), the score reduces to independent reference-weighted soft matching with null. These properties are implementation checks, not predictions of improved recall.

The supplied CPU checks produce zero duplicate-invariance error and capacity violation on a small constructed example, and a finite-difference gradient error below 2×10⁻¹⁰. A distributed-affinity example scores above a concentrated-affinity example under the chosen support geometry. These numbers are **synthetic mathematical checks only**; no sign-language examples were scored.

**N12. Complexity.** Relative to a fixed baseline:

| Quantity | Added cost |
|---|---|
| Parameters | 1024×512+512 = 524,800, independent of gallery size |
| Local projection | O(Mpd) per video, precomputable at inference |
| Pair affinities | O(Mmd); at 64×32×512, ~2.1 million multiply/add FLOPs per pair |
| Solver | O(I(Mm+LM)) with I iterations and L≤2M−1; exponentials/logs and convergence overhead must be profiled |
| Working memory | O(Mm+LM+L) per pair/block, plus cached features; no unrolled-iteration graph with envelope backward |
| Full-gallery ranking | O(N_query N_gallery [Mmd+I(Mm+LM)]), tiled; not an ANN-compatible single dot product |

Do not claim negligible latency from the parameter count. Report GPU seconds/query, batch throughput, peak memory, and preprocessing amortization. A top-K reranker may be useful later, but its candidate ceiling and changed evaluation path must be explicit.

# O. Training Algorithm

```text
INPUT: official raw datasets and annotations; verified baseline configuration
       public Oxford I3D and CLIP artifacts; fixed train/val/test ID manifests

0. Freeze resource manifest: source URLs, checksums, licenses, versions, transforms.
1. Recut/crop/decode raw data with the audited baseline pipeline.
   Save actual window support and sample-selection indices alongside RGB features.
2. Generate target pseudo-labels using TRAIN data only; adapt I3D as baseline requires.
   Freeze both I3D models; independently extract train/val/test features.
3. Reproduce the original CiCo loss/protocol before modifying duplicate treatment.
   If either direction misses the stated tolerance, diagnose; do not claim SOTA.
4. Build the common exact-caption negative-exclusion control and re-evaluate it.
5. Run Stage A diagnostics on validation errors; STOP if the proposed failure is absent.
6. Initialize identical local projections for all scorer controls.
   Warm up projection-only variants with frozen contextual encoders and uniform negatives.
7. For each training batch:
      compute baseline contextual video/text tokens and directional score matrices
      compute L_base using all allowed in-batch candidates
      select fixed-miner hard + random real pairs, excluding exact duplicates
      for each required video/text pair, in memory-bounded blocks:
          project pre-contextual local video features; compute C
          load/build A,w,q from actual support metadata
          solve dual, certify feasibility and primal/dual gap
          compute phi = F(C) - F(0)
          form the SAME directional mixed scores used at inference
      compute L_mix; loss = L_base + lambda * L_mix
      backpropagate using certified assignment/envelope gradients
      update local projection and, after warmup, contextual retrieval encoders
      keep I3D frozen
8. Select checkpoint only by prespecified validation mean bidirectional R@1.
   Stop early or reject if direction-specific harm exceeds the guardrail.
9. Freeze all choices, then run whole-gallery test evaluation across fixed seeds.
```

No training or adaptation uses test labels, test-generated negatives, or the proposed diagnostic strata from the test set. Existing annotations may be inspected for protocol integrity; they must not drive method selection.

# P. Inference Algorithm

```text
PRECOMPUTE per video:
    local I3D features H, local projected vectors U, contextual vectors Z
    exact time geometry A,w,q and F(0), keyed by scorer hyperparameters
PRECOMPUTE gallery text encodings for V2T; encode incoming queries for T2V

FOR each query:
    FOR every candidate in the fixed official gallery, tiled:
        compute reproduced baseline directional score
        compute local token affinities C from U and Y
        solve OCEM dual to the fixed evaluation tolerance
        obtain certified centered score phi
        S = (1-gamma)*baseline_score + gamma*phi
    sort candidates by S, using fixed gallery-ID tie order
    return ranked IDs

EVALUATION:
    compare ranks against original paired IDs
    report both directions' R@1/5/10, MedR, MnR
    separately report exact-caption-equivalence and tie diagnostics
```

The query text is available when scoring a candidate video. The candidate video's ground-truth caption is **not** used for T2V inference. V2T uses the ordinary candidate text gallery. No translation generation, hidden SEDS representation or training-only score is required.

# Q. Why It Should Beat SOTA

This is a **mechanistic case to test**, not an empirical conclusion. OCEM can improve a strong encoder only if a significant fraction of its mistakes are scoring errors caused by concentrated support. If the information is absent from the features, the method has no general reason to help.

| Existing method | Main strength | Remaining issue relevant to the hypothesis | OCEM's specific intervention / limit |
|---|---|---|---|
| SPOT-ALIGN | Strong spotting-derived representations | Sentence-level ranking may miss a local contradiction | Score distributed local support; cannot recover lost visual detail |
| CiCo | Fine cross-lingual matching | Independent aggregation permits joint overuse of correlated windows | Couple allocation across all query tokens through shared support |
| UPRet | Distributional ambiguity modeling | Published transport score is not its inference scorer; shared support is not the stated target | Use a support-constrained optimized value at retrieval, not only a training regularizer |
| SEDS | Richer manual/pose cues | Modality richness alone does not establish independent supporting evidence | Test the scoring hypothesis separately; RGB comparison is not a same-resource superiority claim |
| C²RL | Content and context learning | Strong semantics may still overrate a local near miss | Test a narrow score-level correction; do not promise to bridge its large gains with a small head |
| SAN | Exposes and trains visual confusability | Synthetic fine improvement need not transfer to real-gallery bidirectional ranking | Change the score on real pairs; retain coarse pressure; compare matched mining controls |
| CMCM | Multi-grained causal/distributional direction | Full method/results unknown | No claim it lacks the same mechanism or that OCEM will beat it until verified |
| CSLR² | Joint lexical and sentence supervision | Extra sign resources make direct comparison different | Same-resource scoring study only; no claim to replace its supervision |
| SL-1.5M model | Broad pose pretraining | Scale/resource advantages are not a matched algorithm test | Optional transfer test, not primary comparator |
| VTaMo/DualAnchor, adjacent recent work | Explicit partial/null alignment | Their inspected objectives are translation-oriented | Distinction must survive direct ordinary-OT/null baselines, not task relabeling |

The strongest fair target is determined by the **actual reproduced comparison group**, including UPRet and any newly verified compatible CMCM result. Beating a weaker CiCo-only baseline is useful Stage B evidence, not sufficient SOTA evidence.

# R. Experiment Plan

**Primary datasets:** P14T and How2Sign. They cover two sign languages and substantially different domains and appear in the core RGB retrieval literature. CSL-Daily is the next cross-dataset confirmation after legitimate access is granted. OpenASL and BOBSL are optional, with separate resource/gallery protocols; neither is required to rescue a failed core hypothesis.

**No new dataset or benchmark.** Use existing splits and the standard full candidate galleries. Diagnostic slices are analyses of those same examples. SAN's fixed fine protocol is optional only if its official candidate files or an unambiguous reproducible release become available. Do not construct a new synthetic test set and label it the official SAN result.

**Baseline tiers**

| Tier | Required experiments |
|---|---|
| Reproduction | Original CiCo on both primary datasets, with exact manifest/configuration provenance |
| Matched controls | CiCo with common duplicate mask; same local projection with independent matching; ordinary partial OT; independent clip-cap OT; OCEM |
| Strong same-resource comparison | UPRet reimplemented/retrained on identical regenerated features; reproduce original recipe separately before claiming published equivalence |
| Fine supervision | Random, semantic-hard, pooled-visual-hard, and faithfully implemented SAN/SAN-style local mining; same resources and candidate labels where comparable |
| Broader reported references | SPOT-ALIGN, SEDS, C²RL, SAN, CMCM, SL-1.5M, CSLR² with explicit A/B/C/D status |
| Cross-representation confirmation | After core success, add OCEM to an independently rebuilt second local-feature pipeline; include its own unchanged and partial-OT baselines |

SEDS's published values remain marked **reported only**. Independently extracted RGB/pose features can support a clearly labeled SEDS reimplementation, but they do not reproduce the inaccessible archive merely because the architecture name matches.

**Reproduction gate.** For each primary dataset and direction,

\[
|R@1_{\rm reproduced}-R@1_{\rm reported}|\le1.0\ \text{percentage point},
\]

or provide official variance evidence explaining a different tolerance. The comparison uses the same actual evaluation pool and resources. A lucky score match on a different manifest is not a pass. If it fails, diagnose timing, crops, sample IDs, mask/normalization, features, checkpoint, data adaptation, tokenizer, batch denominator, schedules and versions. Do not compensate with unverified archives. No new SLRet neural method has been trained in this study, so this gate is still open.

**Recommended starting configuration — HYPOTHESES, not published OCEM facts**

| Item | Starting choice / rationale |
|---|---|
| Baseline initialization | Successfully reproduced CiCo checkpoint using Oxford I3D and CLIP ViT-B/32 |
| Features | Frozen, self-extracted 1024-D local windows; same 64-clip cap/selection as baseline; retain exact supports |
| Text | Baseline tokenizer/32-token limit and contextual 512-D vectors; record truncation rate |
| New parameters | One 1024→512 linear projection with bias; no new Transformer |
| Warmup | 5–10 projection-only epochs with frozen contextual encoders; compare all local scorer variants from identical initialization |
| Full optimization | Begin with Adam and the reproduced schedule; contextual LR 1e−5, new projection 1e−4; consider 5e−6–2e−5 and 5e−5–2e−4 only after a viable pilot |
| Mixed loss | λ=1; γ=0.25; ε=0.05; π=0.15; κ=1.5 initially |
| Minimal validation sweep | κ∈{1,1.5,2,4}; include κ=∞ as the no-budget control. Keep ε/π fixed initially; sweep γ∈{0.1,0.25,0.5} only if core evidence survives |
| Batch | 32–128 for cheap projection pilots; matched effective contrastive batch 128–512 for full training according to reproduced baseline |
| Epoch budget | 20–40 matched fine-tuning epochs for an initial full candidate; extend only if validation curves justify it and controls receive equal updates |
| Warmup/schedule | ~5% LR warmup and cosine after the reproduction stage, shared across candidate/control experiments |
| Precision | BF16/FP16 encoders where supported; FP32 solver/log-sum-exp; FP64 CPU reference for solver certification |
| Hardware | One 24–48 GB GPU for pilot; plan 2–4 GPUs of 48–80 GB for full matching/batch experiments; actual throughput must be measured |
| Seeds | Three independent training seeds for primary claims; one-seed screening only for early ablations |
| Selection | Validation mean of T2V/V2T R@1, with no-direction-harm guardrail; test opened after choices are frozen |

Gradient accumulation alone does **not** enlarge an InfoNCE denominator. To match a 512-example negative pool on smaller devices, use cross-device gathering or a verified gradient-cache implementation, and compare all methods identically.

**Raw-to-feature procedure**

1. Download original data and annotations; store SHA-256, release date, source URL and license/access conditions. Join by explicit IDs. Check frame bounds before any model work.
2. H2S: recut full frontal recordings at realigned boundaries. Prefer verified official crop boxes. If rebuilding them, use the pinned public COCO detector, store all boxes, and rerun every baseline on the same crops. Declare the changed preprocessing.
3. Preserve actual frame timestamps/FPS according to the baseline. Do not import SEDS's 24-FPS choice into a CiCo reproduction without checking. The inspected CiCo loader converts OpenCV BGR to RGB, scales image tensors to [0,1], uses mean `(0.5,0.5,0.5)` and std `(1,1,1)`, and resizes/crops with the configured 256/224 sizes. Confirm the transform helper and execution path when locking the environment. [Loader](https://raw.githubusercontent.com/FangyunWei/SLRT/main/CiCo/I3D_feature_extractor/datasets/videodataset.py).
4. Generate train-only target pseudo-labels and adapted I3D as required. Pin model/class vocabulary and feature layer. The paper's BSL-1K wording and released `bsl5k` artifact must be reconciled in the manifest.
5. Extract features with the exact actual window starts; save padding/duplicate supports. Do not regenerate different stochastic feature caches for each ablation.
6. Train retrieval modules on identical feature files. Decode failures and skipped clips are recorded once, not silently handled differently by each model.

**Compute and storage estimates.** These are planning arithmetic, not measured runtimes. At 80 hours and an assumed 24 decoded frames/s, dense stride-1 extraction produces about 6.9 million windows. One FP16 1024-D feature stream is about 14.2 GB; two I3D streams about 28.3 GB before metadata. A fused 64-window cache for roughly 31K clips is about 4.1 GB; contextual 512-D caches add about 2.0 GB. Padding/sampling and actual durations alter these totals.

If measured sustained throughput is 100–500 windows/s/GPU, two frozen passes over 6.9 million windows require about 7.7–38.4 GPU-hours, **excluding** decoding/cropping, pseudo-label generation and sign-encoder training. Benchmark 1,000 clips first and extrapolate with that measured throughput. Raw H2S frontal archives alone total roughly 329 GB from the portal; allow 0.5–1 TB working space if materializing frames/caches. Network transfer and video decoding may dominate initial work. [H2S download sizes](https://how2sign.github.io/).

**Statistical reliability.** Report mean±SD over three seeds, plus paired confidence intervals for recall differences. Resample source-video groups where available; repeat sensitivity analysis grouping exact-caption duplicates rather than treating every repeated query as independent. Use at least 10,000 paired bootstrap resamples after the protocol is frozen. Keep the four primary dataset×direction comparisons visible; do not select the best direction or seed. On P14T, one query is approximately 0.156 percentage point, so a one-point gain represents only about six or seven queries. Fine-grained slices require their own counts and uncertainty.

**Implementation roadmap**

| Milestone | Concrete deliverable / exit condition |
|---|---|
| Resource lock | Exact source/checksum/config manifest; full raw annotations and unchanged galleries |
| Baseline reproduction | Repeatable CiCo run within gate; saved logits, IDs and error list; UPRet route established |
| Frozen-error study | Prespecified support-concentration analysis on validation, with confound controls |
| Reference parity | Batched GPU score/gradient matches CPU reference and satisfies numerical tolerances |
| Minimal matched pilot | Same local head, same updates, ordinary OT versus OCEM; two seeds, validation only |
| Core confirmation | Three seeds on both primary datasets; full-gallery latency and statistical analysis |
| Submission preparation | Main ablations, robustness, current CMCM/full-text resolution and final novelty refresh |

An experienced researcher might budget several weeks for baseline/resource work and another several weeks after a positive pilot. These are workload expectations, not promised calendar dates.

# S. Ablation Matrix

Every row tests an explanation. Adding/removing a named block without a causal prediction is insufficient.

| Experiment | Controlled question | Observation that falsifies the explanation |
|---|---|---|
| Original CiCo → common duplicate-mask baseline | Is an evaluation/training ambiguity fix responsible? | Most apparent OCEM gain disappears after the shared correction |
| Baseline → parameter-matched local linear head + independent score | Is extra local representation sufficient? | Local head matches OCEM under equal updates/parameters |
| Local head → ordinary balanced OT | Does generic joint matching explain the effect? | Balanced OT gives the same gain and error pattern |
| Local head → ordinary partial OT/null, uniform masses | Is partial matching the entire story? | No advantage of shared-support inequalities |
| Partial OT with duration/coverage-weighted marginals | Is nonuniform weighting sufficient? | Same performance without overlapping constraints |
| Independent clip caps vs shared-time caps | Does coupling overlapping columns matter? | Independent caps match OCEM, including overlap-stratified errors |
| Real A vs shuffled window-to-support association | Does true measurement geometry cause the gain? | Shuffled supports retain the gain within uncertainty |
| κ finite vs κ=∞ | Does constrained concentration matter? | Best configuration always removes the constraint |
| Null prior variations / forced matching where feasible | Does allowance for unmatched text prevent harm? | Gain is unrelated to null behavior or requires near-total abstention |
| Centered F(C)−F(0) vs raw F(C) | Is geometry-only bias affecting ranking? | Uncentered gain is explained by length/support offsets |
| Contextual-only / local-only / mixed | Is local evidence complementary to context? | Mixed score adds no value or gains solely from a larger local weight |
| Frozen local head rescoring vs learned OCEM | Is scoring or representation adaptation responsible? | Only extra training, not support-aware scoring, explains the effect |
| Random / semantic / visual / SAN-style negatives | Is mining, rather than the new score, responsible? | Gains vanish when the negative pool is matched |
| Exact duplicate-token copies with split q | Does implementation respect claimed invariance? | Score changes beyond numerical tolerance |
| BPE uniform mass vs deterministic word-group aggregation where applicable | Is tokenizer fragmentation the real driver? | Advantage disappears after identical tokenization controls |
| Same update budget / same wall-time comparison | Is additional optimization sufficient? | Longer-trained baseline closes the gap |

A no-null configuration may be infeasible under strict support budgets. Check feasibility first; report the limitation rather than dropping failing samples. For that ablation, use feasible κ values and document the changed feasible set.

The final method has no pose branch. RGB/pose/shuffled-pose ablations are therefore not relevant to its primary claim. If a later SEDS reimplementation is included, those controls belong to that separate resource track.

# T. Failure / Error Analysis Plan

Analyze the unchanged standard evaluation examples; this is not a new benchmark. Define bins and hypotheses on training/validation and freeze them before test reporting.

| Factor | Operational diagnostic | Interpretation limits |
|---|---|---|
| Sentence/video length | Token count, seconds, sampled windows, truncation | Length can correlate with content difficulty; report conditional comparisons |
| Lexical overlap | Training-defined normalized token overlap between true and retrieved captions, used only for analysis | T2V inference never receives candidate-video captions |
| Semantic hardness | Frozen text-embedding similarity | Proxy for meaning, not an equivalence label |
| Visual hardness | Frozen local/global feature similarity | May reflect signer/background rather than sign confusability |
| Support concentration | Maximum/upper-quantile atom load from alignment, null mass, active-constraint count | Descriptive until controlled intervention confirms relevance |
| Temporal complexity | Motion-feature variation, duration, number of high-change regions | Not a linguistically annotated sign count |
| Frequent/rare concepts | Training-only text frequency bins; existing gloss labels only as a disclosed diagnostic if available | Never use test gloss to train the gloss-free method |
| Signer/source | Existing metadata; within/across-source error rates | Standard splits are not automatically signer-disjoint |
| Visually confusable signs | Existing sign/gloss metadata or documented qualitative cases | Do not infer phonological categories from cosine distance alone |
| Repeated captions | Exact duplicate class size, paired-ID and equivalence-class outcomes | Semantically equivalent paraphrases remain unannotated |
| Natural paraphrases | Only existing alternate/repeated descriptions supported by released data | No LLM-generated test “benchmark” |

For every corrected error, ask whether the false candidate loses support on a small shared interval while the correct candidate retains support elsewhere. Compare against ordinary partial OT on the same pairs. Report newly introduced errors, particularly compact translations, long non-manual spans, fingerspelling, short clips and rare subwords.

**Shortcut tests:**

- Length-only and metadata-only diagnostic baselines; correlate score changes with duration ratios.
- Preserve foreground while masking background, and vice versa, using the same fixed crop masks for both methods. Treat these as distribution-shift diagnostics, not replacement leaderboards.
- Stratify by signer/source; where metadata is absent, state that limitation rather than invent identities.
- Report duplicate-caption strata and exact-input ties. Remove duplicate-negative conflicts in all compared training losses.
- Analyze weather templates separately from low-overlap P14T queries; confirm on instructional H2S.
- Perturb feature-copy density without altering underlying measurements. Separately test modest resampling while recomputing true supports; do not assume physical speed changes preserve every sign meaning.
- Shuffle supports as a negative control. If arbitrary supports help equally, the intended explanation is unsupported.

# U. Reproducibility Checklist

This checklist describes concrete release requirements; unchecked items remain work, not implied completion.

| Item | Current status / required artifact |
|---|---|
| Raw public data | P14T/H2S endpoints verified to the stated level; full transfer/integrity pending; CSL access conditional |
| Exact model IDs | Oxford `bsl5k.pth.tar`; CiCo/OpenAI `ViT-B/32` (`ViT-B-32.pt`); official HF checkpoint as conversion-validated fallback; optional `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` |
| Immutable model versions | Resolve downloaded hashes and repository commits before training. Model names/main-branch URLs alone are not an immutable lock |
| Preprocessing | Official crop/realignment/extraction code inspected; save crops, timestamps, actual frame indices, normalization and invalid-sample reasons |
| Self-generated features | Required; include scripts and feature-layer/shape assertions; no SEDS/Baidu input |
| Target adaptation | Train-only pseudo-label generation; thresholds, vocabulary and adapted-checkpoint hash |
| Split manifest | Original IDs, caption/tokenizer hashes and source group; same files for all methods |
| Baseline gate | Not yet executed; publish original and corrected-loss results separately |
| Dependencies | Original CiCo README specifies Python 3.7/PyTorch 1.7.1/CUDA 11.0; any modern port requires numerical parity. Full package/environment lock pending |
| Mathematical scorer | Supplied `ocem_reference.py`, using NumPy/SciPy; synthetic checks passed |
| GPU scorer | Not implemented or benchmarked; compare values/gradients and record solver iterations/gaps |
| Training provenance | Seeds, optimizer states, schedules, batch denominators, precision, device model, time and peak memory |
| Inference | Whole-gallery scores, candidate order, tie policy and ID ranks; release block-scoring implementation |
| Statistical analysis | Three seeds, paired group bootstrap, all four primary R@1 cells |
| External supervision inventory | BSL backbone and generic CLIP/detector resources declared; no target gloss, pose or generated captions |
| Unavailable resources | SEDS Baidu **UNAVAILABLE / DO NOT DEPEND ON**; do not silently replace with unofficial archives |
| Evidence bundle | Access logs, measured H2S annotation audit, candidate scores and mathematical-check JSON accompany the report |

The full neural pipeline is not claimed to be implemented merely because the reference score runs. The code artifact implements the proposed inner problem and mathematical checks, providing a precise starting point for the PyTorch integration.

# V. Risk Register

Probabilities are qualitative planning judgments: L <~25%, M ~25–50%, H >~50%. They are not calibrated event-frequency estimates and should be updated after each gate.

| Risk | Probability | Impact | Early warning | Mitigation / stop rule |
|---|---|---|---|---|
| Proposed concentration failure is rare or noncausal | H | Fatal to thesis | No enrichment in validation errors | Kill OCEM; do not reinterpret a null result as success |
| Ordinary partial OT already explains gain | M–H | Fatal to novelty | Equal matched scores/error corrections | Reject support-specific contribution |
| Capacity harms valid compact/simultaneous expressions | M | High | More null mass and errors on short/complex clips | Looser κ/shared-context score; kill if only κ=∞ works |
| Pre-contextual features lack necessary detail | M–H | High | Neither local head nor scorer separates near misses | Do not add arbitrary modules; reassess representation premise |
| Local head/extra training explains gain | M | Fatal to attribution | Parameter/update-matched baseline closes gap | Require matched controls; discard claimed mechanism |
| Novelty collision, including unresolved CMCM | M | High | Same support constraint in full prior method | Reformulate substantially or reject; no renaming |
| Dataset access incomplete | M | High | Missing raw archive parts or manifest discrepancies persist | Official recovery only; no cross-dataset claim until complete |
| Baseline cannot be reproduced | M | High | >1-point discrepancy or changed manifest | Diagnose recipe/resources; no SOTA claim |
| Duplicate/tie/crop artifact explains improvement | M | High | Gain changes under common protocol correction | Report correction separately; common controls |
| GPU inner solver is too slow or inaccurate | M | High | Large gap, nonconvergence, impractical latency | Reference parity and batch profiling before scaling; reject if budget fails |
| Stronger comparator eliminates effect | M–H | High | No gain over matched UPRet/verified later model | Limit claim or stop; don't compare only to CiCo |
| Results do not transfer beyond P14T | M | High | H2S direction degrades or effect disappears | Cross-dataset gate; no general SLRet claim |
| Seed/query variance overwhelms small gain | M | High | Paired CI includes zero | More seeds only when a meaningful effect remains plausible; no best-seed reporting |
| Scope drifts into fusion/LLM/benchmark creation | L if enforced | Fatal to requested scope | Rescue requires more modules/labels | Keep one intervention or reformulate explicitly |
| Hidden dependence on SEDS artifacts | L with manifest checks | Fatal | Loader references Baidu/precomputed SEDS paths | Remove dependency before any run |

# W. Go/No-Go Plan

Thresholds below are **prespecified proposed decision rules**, not observed results. They can be agreed before seeing pilot outcomes; do not adjust them retrospectively to keep the idea alive.

| Stage | Experiment | Go criterion | No-go action |
|---|---|---|---|
| 0 — Resource/reproduction | Full files/manifests; CiCo on primary datasets | Exact pool/resources; both directional R@1 values within 1 point or justified official variance | Resolve access/recipe; no new-method SOTA campaign |
| A — Verify failure | Frozen baseline validation errors; support-concentration proxies with same-query correct/incorrect pairs | Incorrect top candidates show higher concentration with paired 95% CI above zero; effect survives length/lexical controls; at least 10% of baseline R@1 errors fall in the prespecified affected stratum | Kill the concentration explanation |
| A2 — Validate local interpretation | Train only the common unconstrained local projection on training pairs; repeat validation analysis | Same qualitative association exists for actual local supports, not only contextual token indices | Reject physical-support interpretation |
| B — Minimal intervention | Frozen common head, score with ordinary partial OT vs OCEM; then short matched head training; two seeds | ≥1.0-point mean bidirectional validation R@1 gain over the strongest simple control, no direction worse by >0.5 point; real A better than shuffled A | Diagnose one concrete issue or stop; do not scale |
| C — Core method | Joint contextual/local training with identical budgets | Gain survives parameter/update/negative controls and numerical certificates; measured latency within the declared deployment/research budget | Simplify once if justified; otherwise stop |
| D — Cross-dataset | P14T and H2S, three seeds | Positive mean R@1 improvement on both datasets, no direction worse by >0.5 point; paired aggregate 95% CI excludes zero | No broad SLRet claim; reconsider whether a narrow domain claim is meaningful |
| E — Full study | Strongest fair baseline, ablations, robustness | At least 1-point aggregate R@1 gain across four cells, support-specific controls fail to explain it, no major shortcut or fairness issue | No SOTA/A* submission claim |

The Stage A concentration statistic can be computed from baseline local affinity maps, but their contextual tokens make it a **proxy**. Stage A2 and the intervention are necessary before a claim about actual local evidence. High concentration alone can be legitimate and is not an error label.

Ordinary partial OT is not an optional ablation: if it matches OCEM, the proposed contribution is too weak. Similarly, a high SAN-style synthetic score cannot rescue a failure on standard real galleries.

**Probability update policy.** Present estimate is 30–50%. Passing A/B might justify a 50–70% research-success band, depending on effect size and controls. A 70–85% “strong research bet” requires replicated cross-dataset evidence against the strongest fair baseline. These bands are judgments, not Bayesian posteriors. Reconsideration of all five candidates did not produce evidence warranting a higher current estimate.

# X. Expected Outcome

The paper thesis would be supported by the following **pattern**, not by any fabricated future number:

1. Erroneous top candidates concentrate local support more than the paired correct candidate after confound controls.
2. OCEM lowers those particular false scores while preserving valid distributed support; ordinary partial OT and independent caps do not fully reproduce the effect.
3. Improvements concentrate at R@1, with smaller positive or neutral R@5/10 changes, consistent with reranking close alternatives rather than just recovering distant matches.
4. Both directions improve under the standard galleries and the effect replicates on P14T and H2S.
5. The result persists after duplicate masking, exact manifest matching, parameter/update controls and three seeds.
6. Legitimate short/compact translations are not systematically harmed; null mass does not collapse toward one.
7. The GPU implementation meets declared numerical and latency limits without inaccessible artifacts.

Evidence against the thesis includes gains only on constructed captions; gains explained by the local head, larger training budget or ordinary OT; unchanged performance with shuffled support; a best κ at infinity; harm tied to sentence compression; or disappearance after protocol correction.

**Paper-level contribution test:** novelty is plausible only in the narrow formulation; importance is conditional on Stage A; technical substance is an explicit constrained score with a tractable dual; evidence can be isolated but has not yet been collected; generality requires two datasets; the insight fits one figure; reproduction is architecturally independent of SEDS but still needs full downloads and a passed neural baseline gate. None of these statements should be converted into an empirical success claim.

# Y. Paper Skeleton

**Working title:** *When Local Matches Reuse the Same Evidence: Support-Constrained Sign Language Retrieval*.

**Abstract thesis, prospectively worded:** Current local SLRet scoring may overrate near matches when correlated visual windows supply repeated support to several query tokens. We investigate this failure under matched training resources and propose OCEM, which couples local assignments through their shared observed time support while retaining a contextual retrieval model. The method uses the same score during learning and retrieval and requires no extra sign annotations or SEDS artifacts. The abstract's final claims and numerical results must be filled only after the experiments succeed.

**Three intended contributions:**

1. A controlled empirical explanation of a real-gallery local-scoring failure, distinguished from representation quality, negative supervision and duplicate/protocol artifacts.
2. A support-constrained retrieval score with explicit unmatched mass, a convex dual and tested numerical properties; no claim to invent optimal transport itself.
3. A reproducible matched-resource evaluation across existing datasets, isolating the mechanism against ordinary partial OT and strong retrieval baselines.

If contribution 1 fails, contribution 2 alone is unlikely to carry the paper.

**Section structure:** Introduction; related work and resource equivalence; diagnosis; OCEM formulation; reproducible implementation; standard retrieval results; hypothesis-driven ablations; error/robustness analysis; limitations.

**Figure 1 concept:** a true query and a real-gallery near miss with comparable baseline scores. Show the underlying timeline and overlapping feature windows. Several words appear supported by the same short visual interval in the false candidate; the correct candidate has additional distributed evidence. Use real examples only after the pattern is verified, with consent/licensing respected.

**Figure 2 method:** raw video → independently extracted local features. One branch retains contextual CiCo scoring; the other produces local affinities and their known support matrix. A small inset depicts shared atom budgets and null mass; the final score blends contextual and constrained evidence. The figure should not depict cells as linguistically annotated signs.

**Key tables:** resource-equivalence matrix; reproduced baseline gate; full-gallery R@1/5/10 by direction and dataset; ordinary-OT/independent-cap controls; negative-policy controls; error-stratum and shortcut analyses; latency/memory and solver accuracy. Keep reported-only SEDS/C²RL references visibly separated from reproduced matched results.

# Z. Final Verdict

**GO WITH CONDITIONS**

Proceed with OCEM's resource lock, baseline reproduction and cheap failure-mode/intervention tests. The method is explicit, independently reconstructible in design, and avoids every forbidden SEDS/Baidu dependency. It also has a clear way to fail scientifically.

Do not yet commit to a full A* submission campaign or claim a high probability of SOTA. The proposed causal failure is unmeasured, the strongest matched neural baseline has not been reproduced, and CMCM remains incompletely verified. Scale only if the gates in W succeed. If ordinary partial OT, a matched local head or duplicate/protocol correction explains the gain, abandon the support-specific thesis rather than adding modules.
