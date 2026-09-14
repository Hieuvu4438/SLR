# RESEARCH STATE CHECKPOINT

**Literature cutoff: 10 September 2026. Checkpoint 8 — resource gate advanced; baseline reproduction pending.**

This supersedes checkpoint 7. The original A–Z design is preserved; no candidate was regenerated. The companion `slret_research_report.md` contains the full A–Z report, numerical tables, mathematical specification and experimental plan. `slret_evidence_audit.json` preserves access checks and annotation calculations; `ocem_reference.py` and `ocem_mathematical_checks.json` contain a CPU scorer and synthetic mathematical checks. They contain no trained SLRet model or empirical retrieval improvement.

### 1. Current research objective

Develop one defensible methodological/algorithmic contribution to **Sign Language Retrieval**, primarily sentence-level T2V/V2T, using established datasets. No new dataset/benchmark contribution, invented results, scale-only contribution or unlabelled supervision advantage. Fair SOTA comparison is required. The pipeline must have **no dependency on inaccessible SEDS pretrained checkpoints or SEDS/Baidu-hosted precomputed features**. Prefer raw public videos, public preprocessing and independently regenerated representations. Distinguish facts, measured audit findings and hypotheses.

### 2. Current phase

Completed: resumed checkpoint 3 without restarting; literature/SOTA reconstruction; dataset/protocol/resource audit; failure taxonomy and gap ranking; exactly five candidates; scoring; adversarial novelty searches; three simulated reviews; revisions; selection of exactly one method; full mathematics, pseudocode, experiments, ablations, risks, compute estimates and stop rules; CPU mathematical reference checks; A–Z report.

Current decision: **GO WITH CONDITIONS** for an OCEM pilot. Subjective probability of beating the strongest matched baseline across two datasets: **30–50%**, not a measured probability or a 70–85% strong bet. Reconsideration of all five candidates did not justify inflation.

Newly completed: full Oxford I3D transfer/hash, all H2S annotations including test, exact CiCo CLIP source inspection and repository commit recording. Not completed: full raw-video downloads, complete CLIP transfer/parity and model-load verification; complete CMCM/GTRN full-text audit; neural baseline reproduction; GPU OCEM implementation; SLRet training/evaluation. Next phase is the resource/reproduction gate, followed by measuring the hypothesized failure. Do not claim that mathematical scorer checks passed the reproduction gate.

### 3. Verified papers

U means **UNVERIFIED — DO NOT USE AS FACT**. Reported numbers are published results, never our reproductions. Exact recall values are retained in section 4.

| Paper / venue | Verified mechanism and datasets | Primary source; code/resource state |
|---|---|---|
| **Sign Language Video Retrieval with Free-Form Textual Queries**, Duarte et al., CVPR 2022 | SPOT-ALIGN: I3D spotting, joint text/video embeddings and sign-recognition fusion; H2S/P14T | [Paper](https://arxiv.org/abs/2201.02495), [project](https://imatge-upc.github.io/sl_retrieval/). Code/project route known; full artifact chain not certified |
| **CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning**, CVPR 2023 | Target-adapted + domain-agnostic I3D; CLIP contextual encoders; independent row/column softmax-weighted token aggregation and contrast; H2S/P14T/CSL | [Paper](https://arxiv.org/abs/2303.12793), [official code](https://github.com/FangyunWei/SLRT/tree/main/CiCo). Oxford I3D full transfer/hash passes; CLIP range passes but full transfers time out; target adaptation planned |
| **A Tale of Two Languages: Large-Vocabulary Continuous Sign Language Recognition from Spoken Language Supervision**, Raude et al., 2024 arXiv; final venue U | CSLR²: Video-Swin-Tiny/Kinetics + isolated-sign pretraining, frozen T5-large, sign and sentence objectives/HN-NCE; BOBSL | [Full paper](https://arxiv.org/abs/2405.10266). Full code/checkpoint/raw-data chain U |
| **Uncertainty-aware Sign Language Video Retrieval with Probability Distribution Modeling**, ECCV 2024 | UPRet: CiCo-derived RGB/CLIP features, Gaussian embeddings, sampling and OT; H2S/P14T/CSL. OT score is training-only in §3.6 Eq.24 | [Paper](https://arxiv.org/abs/2405.19689), [code](https://github.com/xua222/UPRet). Code exists; trained binaries/features not fully verified |
| **SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval**, ACM MM 2024 | I3D + RTMPose; SignBERT-initialized hand GCN, body GCN, local semantic fusion and CLIP; H2S/P14T/CSL | [Paper](https://arxiv.org/abs/2407.16394), [code](https://github.com/longtaojiang/SEDS). **Baidu checkpoints/features UNAVAILABLE / DO NOT DEPEND ON** |
| **Deep Understanding of Sign Language for Sign to Subtitle Alignment**, Jang et al., 2025 arXiv; final venue U | I3D/BERT selective alignment, subtitle processing, manual alignment subset and self-training; downstream CSLR² on BOBSL | [Paper](https://arxiv.org/abs/2503.03287), [paper-linked code](https://github.com/art-jang/sign-to-subtitle). Artifact chain U; extra timing supervision |
| **Graph traverse reference network for sign language corpus retrieval in the wild**, Hu et al., Neurocomputing 637 (2025), 130077 | GTRN: visual query/corpus retrieval, hierarchical frame/body-part reference attention. Exact backbones/datasets/numbers U | [Author institution](https://ro.ecu.edu.au/ecuworks2022-2026/6028/), [DOI](https://doi.org/10.1016/j.neucom.2025.130077). PDF inaccessible; code/checkpoints U. Do not place in a sentence T2V table |
| **C²RL: Content and Context Representation Learning for Gloss-Free Sign Language Translation and Retrieval**, TCSVT 35(9), Sept 2025, 8533–8544; preprint 2024 | ImageNet ResNet18, temporal convolution, ICL/CLCL + autoregressive ECL; frozen representation features and two independent mBART-large-cc25 encoders downstream; H2S/P14T/CSL/OpenASL | [Preprint](https://arxiv.org/abs/2408.09949), [journal](https://doi.org/10.1109/TCSVT.2025.3553052). No verified author code; mBART config only checked. **Not TPAMI** |
| **Scaling up Multimodal Pre-training for Sign Language Understanding**, Zhou et al., TPAMI 47(12), Dec 2025, 11753–11767 | Manual/non-manual pose, contrast + masked pose pretraining on SL-1.5M; sentence retrieval P14T/CSL | [Paper](https://arxiv.org/abs/2408.08544), [journal](https://doi.org/10.1109/TPAMI.2025.3599313). Complete public corpus/model chain U; extra-resource comparator |
| **Causality-inspired multi-grained cross-modal sign language retrieval**, Yang/Wei/Li/Hu, CVIU 264 (Feb 2026), 104631 | CMCM publisher highlights: augmentation/backdoor adjustment, Gaussian alignment, temporal-motion covariance. Full published recipe/results U | [Publisher](https://doi.org/10.1016/j.cviu.2025.104631), [paper-named repository](https://github.com/vddong-zjut/CMCM). Repository has isolated modules/loaders, not a certified runnable reproduction; author linkage incomplete |
| **Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative Mining for Sign Language Retrieval**, ACL 2026 long papers | SAN on CiCo/GFSLT; local sign–word confidence, visually similar tokens with different words, synthetic substitutions; standard P14T and fine V2T | [Paper](https://arxiv.org/abs/2607.09263), [ACL acceptance](https://2026.aclweb.org/program/accepted_papers/), [repository](https://github.com/joonmy/SAN). README “Coming Soon”; runnable release not established |

Adjacent verified work retained for novelty, not a common SLRet leaderboard:

- [ColBERT](https://arxiv.org/abs/2004.12832): established multi-vector late interaction.
- [Drop-DTW](https://arxiv.org/abs/2108.11996): alignment with dropping/outliers; monotonicity is not automatically valid for sign/spoken word order.
- [CrossCLR](https://arxiv.org/abs/2109.14910): related-negative exclusion/intra-modal similarity.
- [Verbs in Action](https://arxiv.org/abs/2304.06708): hard captions/verb alignment; limits residual-caption novelty.
- [Norton: Multi-granularity Correspondence Learning from Long-term Noisy Videos](https://arxiv.org/abs/2401.16702): noisy correspondence/OT/unmatched content.
- [SignCL: Improving Gloss-free Sign Language Translation by Reducing Representation Density](https://arxiv.org/abs/2405.14312), NeurIPS 2024: adjacent positives/distant temporal negatives. Generic temporal separation is not novel.
- [SignRep: Enhancing Self-Supervised Sign Representations](https://arxiv.org/abs/2503.08529): skeleton-informed RGB/style-adversarial pretraining; dictionary retrieval, not the standard sentence protocol.
- [SHuBERT](https://arxiv.org/abs/2411.16765): representation/translation lead, abstract-level only; no verified new standard retrieval result here.
- [Lost in Translation, Found in Embeddings: Sign Language Translation and Alignment](https://arxiv.org/abs/2512.08040): pose/lip, Sliding Perceiver, SLT/SSA, external pretraining; not an equivalent sentence-retrieval result.
- [SEA: Segment, Embed, and Align](https://arxiv.org/abs/2512.08094v2), ACL 2026: pretrained segmentation, embeddings and dynamic programming; extra segmentation supervision.
- [VTaMo: Video-Text Alignment Model for Sign Language Translation](https://arxiv.org/html/2607.09126v1): balanced uniform OT, null token, temporal regularization, training-time target-guided reordering. Full text read.
- [DualAnchor: Preserving Language Priors and Improving Lexical Fidelity in Gloss-Free Sign Language Translation](https://arxiv.org/pdf/2607.27614v1): partial OT/dustbins, frozen-LM anchoring; OT removed at inference. Full PDF read. Retrieval-quality strata are not full-gallery R@K.
- [AVIOT: Aggregating Visual Information with Optimal Transport for VideoLM Token Compression](https://arxiv.org/pdf/2608.20473v1): balanced source-to-compact-support OT/query-conditioned temporal allocation. Full PDF read; no shared raw-receptive-field constraint identified.
- [Capacity-constrained OT](https://arxiv.org/abs/1307.7774), [NMT coverage](https://arxiv.org/abs/1601.04811), [Gromov–Wasserstein graph matching](https://arxiv.org/abs/1901.06003): mathematical/coverage/relational prior art. Do not claim these concepts as inventions.

### 4. Current SOTA understanding

All tuples are **T2V R@1/5/10 | V2T R@1/5/10 (%)**. These are references, not one fair leaderboard.

| Resource group / method | H2S | P14T | CSL-Daily |
|---|---|---|---|
| RGB + sign pretraining: SPOT-ALIGN SA-COMB | 34.2/48.0/52.6 \| 23.6/47.0/53.0 | 55.8/79.6/87.2 \| 53.1/79.4/86.1 | — |
| RGB + sign pretraining: CiCo | 56.6/69.9/74.7 \| 51.6/64.8/70.1 | 69.5/86.6/92.1 \| 70.2/88.0/92.8 | 75.3/88.2/91.9 \| 74.7/89.4/92.2 |
| RGB + sign pretraining: UPRet | 59.1/71.5/75.7 \| 53.4/65.4/70.0 | 72.0/89.1/94.1 \| 72.0/89.4/93.3 | 78.4/89.1/92.0 \| 77.0/89.2/92.7 |
| RGB + pose: SEDS | 62.5/75.1/80.1 \| 57.9/70.4/74.9 | 76.8/91.7/95.3 \| 78.7/92.5/95.2 | 85.8/94.4/95.6 \| 85.4/93.8/95.8 |
| Target representation/translation pretraining + mBART: C²RL | 62.4/75.9/80.1 \| 57.5/68.4/73.0 | 78.7/92.2/94.9 \| 77.6/91.3/94.2 | 90.3/96.4/97.7 \| 88.4/95.7/97.1 |
| Pose + external SL-1.5M | — | 74.5/93.3/95.6 \| 75.1/92.1/95.3 | 87.5/95.2/97.6 \| 87.2/95.0/97.2 |

MedR is 1/1 where reported for these modern rows; SPOT H2S is 8/7.5. UPRet MnR T/V: H2S 54.4/76.4, P14T 4.4/4.6, CSL 6.7/5.5. UPRet's own CiCo H2S V2T R@1 = 50.3 differs from the original 51.6; do not erase the discrepancy.

C²RL OpenASL table: **62.2/81.7/86.8 | 61.6/79.8/84.6**; prose says 62.6 for T2V R@1. Preserve table/prose conflict; preprint/journal numerical agreement U.

CSLR² BOBSL Sent-Val (1,973): **51.7/69.9/75.4 | 50.2/69.1/74.7**. Sent-Test (20,870): **29.4/45.2/51.5 | 28.1/44.9/51.0**. Prose reverses subset sizes; table supports 2K validation/20K test. Jang alignment experiment's own CSLR² reference → improved: T2V R1/5 **27.14/42.19→28.59/43.74**, V2T **26.25/41.99→26.59/42.51**; extra timing supervision and distinct setup.

SAN P14T standard (full 642) versus fine V2T (1 positive + 40 synthetic negatives):

| Model | Standard T2V | Standard V2T | Fine V2T R1/5/10; MRR |
|---|---|---|---|
| CiCo in SAN | 69.2/87.2/92.2 | 70.1/87.7/92.9 | 17.9/55.3/79.1; 35.0 |
| CiCo+SAN | 68.1/87.4/91.7 | 67.8/87.4/91.7 | 39.4/75.4/92.5; 54.4 |
| GFSLT in SAN | 67.9/88.4/93.8 | 69.4/88.7/93.3 | 16.8/53.1/78.0; 33.9 |
| GFSLT+SAN | 70.2/89.3/94.4 | 67.4/85.4/90.5 | 49.1/85.9/94.9; 64.1 |

Fine negatives: ten each SAN/FastText/RoBERTa/GPT-4o-mini; not a new benchmark to construct here. CMCM/GTRN numbers remain U.

Equivalence: **A** only actually matched reproduced runs; **B** related published RGB settings pending manifests/configs; **C** pose, translation/large-language-model or external-corpus supervision differences; **D** unverified results/protocols or incompatible query/gallery tasks. Current proposal-vs-CiCo A status is planned, not achieved. Strongest fair target must include matched UPRet and any subsequently verified compatible CMCM; a CiCo-only win is insufficient SOTA evidence.

### 5. Dataset/protocol state

| Dataset | Identity, splits and annotations | Protocol and risks |
|---|---|---|
| [How2Sign](https://how2sign.github.io/) | ASL instructional; ~80 h; 11 signers overall, nine green-screen; paper 8/5/6 train/dev/test. Common release 31,164/1,740/2,356; CiCo 31,085/1,739/2,348. RGB, translations, boundaries, metadata, automatic pose; usable gloss release U | Recut full frontal recordings with realigned annotations; old sentence clips differ. SPOT training count 31,075; SEDS 31,019/dev 1,738. Whole 2,348-test gallery in cited methods. Exact manifest pending; signer, background, template, duplicate and timing risks |
| [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) | DGS weather; ~11 h, 386 broadcasts, nine interpreters; 7,096/519/642; 25 FPS, 210×260 raw frames, translation and gloss | `features/fullFrame` means raw images. Small, template-heavy gallery; signer-disjoint guarantee U. C²RL's 7,098 prose typo is not adopted |
| [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/) | CSL daily life; ~23 h, ten signers, 20,654 samples; 18,401/1,077/1,176; RGB, gloss, translation | Institutional agreement required; conditional third dataset. Template/signer risks; no unverifiable mirror |
| [OpenASL](https://github.com/chevalierNoir/OpenASL) | ASL online news/vlogs; ~288 h, ~220 signers, 98,417 pairs; C²RL 96,476/966/975 versus original dev/test 967/976; URLs, subtitle times, automatic crops | Disappearing source videos change the gallery. Original caption overlap 105/967 dev and 103/976 test concerns translation, not a measured retrieval effect |
| BOBSL / CSLR² | BSL broadcast; 1,467 h, 39 signers; 689K aligned training subtitles; Sent-Val 1,973/Sent-Test 20,870; derived sign labels/timestamps | Separate large-gallery/extra-supervision group; raw license/download route not certified; timing, channel, signer and episode risks |

SL-1.5M sources verified in §III-A: WLASL, MSASL, NMFs-CSL, SLR500, PHOENIX14, PHOENIX14-T, CSL-Daily, How2Sign and BOBSL. Isolated glosses become templated text; PHOENIX14 contributes unpaired pose. These are pretraining resources, not nine equivalent sentence-retrieval protocols. None is an additional OCEM dependency.

**Measured H2S annotation snapshot:** downloaded official train/dev files contain 31,165/1,741 rows with unique sample IDs. Train: 30,109 raw captions; 1,509 rows in repeated-caption classes; 1,056 duplicate excess (1,063 after case/whitespace normalization). Dev: 1,516 raw captions; 444 rows in repeated classes; 225 excess. Forty-seven dev rows (33 distinct captions) occur verbatim in train. The test retry succeeded: 2,357 unique sample IDs; 1,938 raw captions; 777 rows in repeated classes; duplicate excess 419. Seventy-one test rows (44 captions) occur in train; 14 distinct captions occur in both dev/test. All three pairwise `VIDEO_ID` intersections are zero. No blank captions or nonpositive durations. These are annotation counts, not the filtered usable-video manifest or proof against visual leakage. Test inspection was for protocol integrity only; the earlier quota block is resolved.

Common evaluation: preserve paired IDs and whole official galleries; report T2V/V2T R@1/5/10, MedR and MnR; retain the standard ID target and separately diagnose equivalent captions; deterministic gallery-ID tie order plus tie bounds. Reproduce the legacy loss first, then apply exact-caption negative masks to all matched controls. No new semantic labels, revised benchmark or test-driven choices.

### 6. Confirmed findings

**F1:** Fine discrimination and standard retrieval can move in opposite directions. **Evidence:** SAN table above. **Implication:** synthetic-choice gains alone cannot support standard bidirectional SOTA.

**F2:** The frontier uses materially different resources. **Evidence:** SEDS pose/SignBERT; C²RL target translation/mBART; SL-1.5M external corpus versus CiCo/UPRet RGB. **Implication:** separate A/B/C/D comparison groups.

**F3:** Local alignment, visual-hard mining, uncertainty/OT, pose fusion, translation pretraining and null matching already exist. **Evidence:** CiCo, SAN, UPRet, SEDS, C²RL, VTaMo, DualAnchor. **Implication:** reject broad novelty claims.

**F4:** Split and caption ambiguity are real audit issues. **Evidence:** conflicting H2S counts, measured duplicate captions and OpenASL source loss/overlap. **Implication:** manifests, tie handling and matched duplicate controls are mandatory, not the contribution.

**F5:** SEDS `metrics.py` collects every diagonal-score tie, giving queries variable numbers of ranks. **Evidence:** [code](https://raw.githubusercontent.com/longtaojiang/SEDS/master/metrics.py); synthetic 2×2 example gives legacy 66.67% versus per-query expected 75%. **Implication:** audit the evaluator. Impact on published results is unmeasured; do not allege inflated results.

**F6:** A SEDS-independent route has stronger resource evidence. **Evidence:** full Oxford I3D transfer/hash, all H2S annotation files, raw P14T/H2S and CLIP/detector ranges. **Implication:** annotation access is resolved; full raw-video integrity, CLIP parity, model loading and exact manifests remain gates.

**F7:** OCEM's convex formulation/reference gradients pass constructed checks. **Evidence:** duplicate error 0, capacity violation 0, duality gap 0, finite-difference error 1.67e−10. **Implication:** mathematical sanity only; not retrieval evidence.

### 7. Open hypotheses

**H1 — PROMISING, not empirically verified.** False near-miss pairs disproportionately reuse shared temporal support; OCEM reduces these errors. Support: CiCo permits reuse and SAN documents fine errors. Contrary: valid semantics can be compact/simultaneous; reuse may be irrelevant. Test validation-error concentration with length/lexical controls, then a pre-contextual local head with real versus shuffled support. Kill if absent or ordinary OT explains the gain.

**H2 — OPEN.** Support constraints improve standard R@1 without harming broad recall. Support: the intervention affects inference. Contrary: null/capacity may discard valid evidence. Test both directions and datasets with fixed guardrails.

**H3 — OPEN.** Real rival-caption residuals isolate distinctions better than synthetic negatives. Contrary: T2V cannot access candidate-video captions; student transfer may fail. C2 not selected.

**H4 — WEAK for this commitment.** Within-clip boundary marginalization helps established aligned datasets. Timing literature supports it, but SEA/prior alignment and span ensembling limit novelty/headroom. C3 not selected.

**H5 — WEAK.** Cross-modal relations identify compositional errors without extra labels. Identifiability and GW prior art are substantial objections. C4 not selected.

**H6 — WEAK as an A* method.** Cross-fitted reliability improves ambiguous-positive training. Duplication supports the motivation; false-negative prior art, confirmation bias and ID-metric ambiguity limit it. C5 not selected.

### 8. Research gaps under consideration

| Rank / gap | Evidence and unresolved part | Closest work, risks and resources |
|---|---|---|
| 1. Shared support counted repeatedly | Structural possibility verified; causal SLRet error magnitude unmeasured. G1 remains conditional | CiCo, UPRet, VTaMo, DualAnchor, AVIOT, coverage/OT. Moderate-high novelty risk; moderate possible R@1 upside; existing RGB/timestamps |
| 2. Fine/coarse ranking trade-off | SAN evidence; cause and same-resource remedy open | SAN, CSLR², Verbs in Action. Moderate upside, crowded mining; existing paired data |
| 3. Uncertain clip boundaries | SPOT/CSLR²/Jang evidence; headroom on cleaned H2S/P14T unknown | SEA, Norton. Moderate/low upside, high incrementality risk; within-clip spans |
| 4. Relational correspondence | Local ambiguity supported; relation-specific retrieval failure unmeasured | SignCL, CMCM, GW. High novelty/optimization risk; existing pairs |
| 5. Ambiguous positives | Duplication verified; retrieval harm unmeasured | CrossCLR, UPRet, Norton. Modest upside, high prior-art risk; train captions |

Other assessed gaps: generic hand/non-manual omission addressed by SEDS; semantic versus visual hardness by SAN; uncertainty by UPRet/CMCM; global-only matching by CiCo. Signer/background shortcuts are unmeasured in current retrieval and face SignRep prior art. Evaluation inconsistency is protocol work, not the primary algorithmic contribution. Conditional gaps are not established empirical findings.

### 9. Rejected ideas

Do not regenerate without new evidence: CiCo+SEDS; UPRet+generic hard negatives; C²RL+pose; SAN+another loss; RGB/pose concatenation; generic partial/null OT; word substitutions presented as new mining; larger CLIP/LLM/data; merely adding attention/Transformer blocks; generic causal/multi-grained stacks; dataset/benchmark creation; any SEDS/Baidu dependency. Reasons: prior-art collision, forbidden scope, unfair resources or no isolated mechanism.

C2–C5 are documented alternatives, **not selected for implementation**. Do not redefine them silently or import their modules into OCEM. If ordinary OT, a local head or protocol correction explains OCEM's gains, abandon its specific thesis instead of adding modules.

### 10. Candidate methods

Exactly five were generated. All can use the common self-extracted RGB/I3D/CLIP stack. None requires SEDS, pose, new manual labels or inaccessible intermediate features. Full resource validation remains pending.

| Candidate | Fixed mechanism / objective / inference | Closest three; score and risk |
|---|---|---|
| **C1 OCEM** | Local projection; entropy-regularized assignment with null and shared-time constraints; optimized score blended with CiCo at train/test | CiCo, UPRet, VTaMo; **60.4/100**, rank 1. Unmeasured cause, OT collision, solver latency |
| **C2 Rival-Conditional Evidence Scoring (RCES)** | Align real paired/rival captions, remove shared content, learn residual preferences. V2T uses gallery text; T2V requires a student using only query/video at test | SAN, CSLR², Verbs in Action; **46.0**, rank 3. Direction asymmetry, privileged-caption leakage, student engineering |
| **C3 Boundary-Marginalized Retrieval (BMR)** | Five spans: full, center 90%, first 90%, last 90%, center 80%; half prior on full; prior-weighted log-sum-exp plus base/full-span preservation. No outside context | SPOT, SEA, Norton; **46.6**, rank 2. Ensemble/boundary prior art, weak clean-data headroom |
| **C4 Relation-Consistent Late Interaction (RCLI)** | Low-rank video/text relations; unary matching plus fused-GW penalty ΣPijPkl(Rv_ik−Rt_jl)²; optimized test score; no sign/spoken-order identity assumption | CiCo, CMCM, GW; **33.4**, rank 5. Nonconvexity, identifiability, compute |
| **C5 Reliability-Bounded Positive Sets (RBPS)** | Train-only cross-fitted bidirectional confidence, capped soft-positive weights; exact-duplicate mask shared with controls; independent inference | UPRet, CrossCLR, Norton; **42.9**, rank 4. Confirmation bias, old false-negative ideas, ID-metric ambiguity |

Criteria, in order: scientific novelty, novelty defensibility, SLRet motivation, evidence, retrieval upside, R@1 upside, generalization, fairness, implementation, accessibility, ablation, reviewer appeal, compute.

- C1: [7,6,8,5,7,8,7,9,7,8,9,7,6].
- C2: [7,5,8,7,7,8,6,7,5,8,7,7,5].
- C3: [5,4,7,8,5,5,7,8,8,8,9,5,5].
- C4: [6,4,8,4,7,7,6,8,4,8,6,7,3].
- C5: [4,3,7,7,5,5,7,9,8,8,9,5,9].

Utility = 100 × geometric mean(score/10); subtract penalties 11,20,16,24,20 respectively. These are subjective decision utilities, not probabilities. No precise future recall gains were predicted.

### 11. Current leading method

**OCEM — Overlap-Constrained Evidence Matching.** Hypothesis: shared support inflates near-miss matching; couple assignments by actual observed time and retain this correction at inference. This is not a claim that linguistic information is uniformly distributed or one word equals one sign.

Architecture: frozen self-generated I3D features h_i∈R1024 from actual 16-frame windows I_i; retain the reproduced CiCo contextual branch and CLIP text tokens y_j∈R512. New local linear projection 1024→512 plus bias, L2 normalization: **524,800 parameters**. The local branch bypasses the global visual Transformer. M≤64, m≤32 non-special BPE tokens. No pose, gloss, LLM or translation decoder is added.

Specification (full equations in report N):

- Split Ω=∪I_i into covered endpoint atoms J_a, L≤2M−1. Set A_ai=|J_a∩I_i|/|I_i| and w_a=|J_a|/|Ω|.
- Group identical intervals into canonical supports g with multiplicity n_g. Let c(t) count unique supports covering t. Define qbar_g=|Ω|⁻¹∫_(I_g)1/c(t)dt and q_i=qbar_g/n_g. Exact copies share prior mass; arbitrary frame-rate invariance is not claimed.
- b_j=1/m; R_ji=b_j(1−π)q_i, R_j0=b_jπ; start π=.15. C_ji=y_jᵀnormalize(W_vh_i+b_v).
- P≥0; each real-plus-null row sums to b_j. Set t_i=Σ_jPji and require **At≤κw**, starting κ=1.5. Null score is zero. No monotonic alignment.
- Fκ(C)=max_P〈P_real,C〉−εKL(P||R), starting ε=.05. **φ=Fκ(C)−Fκ(0)** removes geometry-only offsets.
- Dual: minimize over μ≥0 the value εΣ_jb_j log[π+(1−π)Σ_iq_i exp((Cji−(Aᵀμ)i)/ε)]+κwᵀμ. Gradient = κw−At. P is the corresponding weighted row-softmax with null. Envelope gradient dF/dC=P_real at the optimum.
- CPU reference uses L-BFGS-B. For a feasible lower bound, scale real mass by ζ=min(1,min_a κw_a/max((At)a,δ)) and transfer removed mass to null. Primal≤F≤dual. Training residual≤1e−5/gap≤1e−4; evaluation gap≤1e−6. Balanced Sinkhorn is not a drop-in solver.
- Final directional score: **S^d=(1−γ)s_CiCo^d+γφ**, starting γ=.25. Same score at train/test.
- **L=Lbase+λLmix**, starting λ=1. Each term averages symmetric InfoNCE on both directional matrices, faithful to CiCo's two scores. Base uses all batch pairs. Mixed candidates: positive plus up to eight hard and eight random negatives per row/column; union required pairs. Critical ablations share a frozen baseline miner and sampling seeds.
- Positive = original paired ID. Reproduce legacy training first; then use exact-caption off-diagonal negative exclusion in every matched control. No inferred paraphrase positives. Projection-only warmup with uniform negatives for 5–10 epochs; contextual encoders may then train while I3D stays frozen.
- Full-gallery inference in both directions, tiled by pairs. No candidate-video captions in T2V. Shortlisting is a separate deployment experiment. Added pair cost O(Mmd+I(Mm+LM)), memory O(Mm+LM); ~2.1M affinity FLOPs at 64×32×512 plus the solver. GPU throughput is unknown.

Optimization starting hypotheses: Adam; contextual LR 1e−5/new head 1e−4; effective batch 128–512 matched to the reproduced baseline, pilot batch 32–128; mixed-precision encoders, float32 solver; nominal 2–4 GPUs depending on batching/tiling. These are recommendations, not measured runtime claims.

OCEM ranks first for its small, isolatable mechanism, convex inner problem, matched RGB resources and clear ordinary-OT/shuffled-support controls. Its SOTA upside is not empirically established.

### 12. Novelty audit state

Completed searches combined sign retrieval/translation, video-text and multimodal retrieval with coverage, evidence reuse, overlap, receptive field, partial transport, null assignment, capacity, redundancy, false negatives, residual rivals and relational/GW matching. Backward/forward searches used CiCo, UPRet, SEDS, C²RL and SAN; 2026 title/DOI checks were completed. Do not repeat generic searches without a new lead.

Closest collisions inspected: VTaMo, DualAnchor, AVIOT, CiCo, UPRet, Norton, ColBERT, coverage and capacity-constrained OT. **Generic OT, null/partial matching, coverage and solver novelty are rejected.** The surviving claim is narrowly an aggregate assignment constraint over actual overlapping local video supports, connected to an isolated SLRet failure and retained at inference. No exact match was identified; search absence is not proof of novelty.

Open: CMCM full paper and complete author-linked code; potentially unindexed recent receptive-field-constrained retrieval. Its paper-named repository has isolated modules, an undefined `DEVICE` and shape/configuration questions. Static issues do not establish failure of the published system.

Simulated reviews: A (sign expert) 4/10, confidence 4/5; B (retrieval expert) 4/10, confidence 4/5; C (hostile) 3/10, confidence 5/5. A attacks linguistic capacity assumptions, simultaneous cues and contextual support. B attacks OT novelty, head/training confounds and latency. C attacks absent empirical evidence/reproduction and unknown CMCM. Revisions: local pre-contextual branch, loose/null budget, ordinary OT/independent caps/shuffled support, full-gallery profiling and hard gates. Scores were not raised just because prose improved. These are simulated perspectives, not external reviews.

### 13. Resource availability

A successful byte range is **not** full download, hash or model-load verification. Oxford I3D is now fully downloaded/hashed, but no model was loaded. Raw datasets retain full integrity gates.

| Resource | Status / source | Needed? / alternative |
|---|---|---|
| P14T raw v3 | HTTP206; 41,699,758,035 total bytes: [official archive](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz) | First dataset; full hash/extraction pending |
| H2S full frontal | [Official script](https://raw.githubusercontent.com/how2sign/how2sign.github.io/main/download_how2sign.sh). Public confirmation then range206; first part 32,212,254,720 bytes. Train ~290GB/dev16GB/test23GB | Planned second dataset; no full archive audit |
| H2S train/dev annotations | Official Drive IDs `1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok`, `1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI`; fully downloaded/parsed | Required; hashes below |
| H2S test annotations | Same official ID `1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu` now returned a full 423,682-byte TSV; parsed/hashed | Access block resolved; exact 2,348 usable-video manifest still pending |
| Oxford I3D `bsl5k.pth.tar` | Full HTTP200 transfer, 142,594,302 bytes; locally hashed: [Oxford](https://www.robots.ox.ac.uk/~vgg/research/bslattend/data/bsl5k.pth.tar) | Required initialization; model load still pending. A different I3D changes comparability |
| CLIP | CiCo uses [OpenAI `ViT-B/32`](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt); official source/checksum verified. HF range passed, but both full-transfer attempts timed out | Required; prefer exact CiCo format. Official HF `openai/clip-vit-base-patch32` fallback needs conversion/parity checks; no model load yet |
| Target-domain I3D | Published adapted checkpoint not certified | Regenerate using official CiCo and training-only pseudo-labels; share with controls |
| Crop detector | Torchvision `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`; [binary](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) range167,502,836 bytes | Regenerate boxes if needed; original CiCo bbox completeness U. New boxes are not asserted identical |
| CiCo feature extraction | Scripts read: 16 frames, stride1, RGB0..1, mean.5/std1, resize256/crop224; record actual supports | Required, self-extracted; no SEDS features |
| SEDS checkpoints/features | **UNAVAILABLE / DO NOT DEPEND ON**, Baidu-hosted | **No**; published values or labelled independent reimplementation only |
| CSL-Daily | Institutional agreement | Optional third dataset, not anonymous access |
| OpenASL / BOBSL | Source availability varies / license-download chain uncertified | Optional separate groups |
| ResNet18 | Official range46,830,571 bytes; `resnet18-f37072fd.pth` | C²RL comparator only |
| mBART-large-cc25 | Config retrieved; full weights U | C²RL comparator only |
| SAN code | README “Coming Soon” | Faithful baseline when released; otherwise explicitly SAN-style |
| CPU reference | Python3.12.14, NumPy2.3.5, SciPy1.17.0 executed; Torch absent | Mathematical reference only; GPU trainer not implemented |

Additional complete-transfer hash: Oxford I3D `6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f`. The response length matches 142,594,302 bytes; model loading is unverified. H2S test hash: `d1799dcbf100eda822eeb1a0e6e754d82b12b2a57756bd47073c4e4b8a02ae4d`.

Exact OpenAI CLIP expected SHA256: `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`. Recorded heads: SLRT `38a4f7b00da7a858d59b7fabe5093876a84db8e0`; OpenAI CLIP `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. [CiCo loader](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/module_clip.py) uses that exact OpenAI artifact. Full official/HF CLIP downloads timed out; do not mark them complete. The current workspace had ~29 GiB free, below the PHOENIX archive alone; full raw staging needs a larger data volume. PyTorch/model loading and neural training remain unperformed.

H2S train SHA256: `67a8f4fb6dc066f38006b89ac84c27a4b05d7fa07d5e6952f997042724d7a847`.

H2S dev SHA256: `1bf5d57ce90d61571776d06a179335d99648d8c94fc22b327d2abb328d26137c`.

CiCo paper BSL-1K versus released extractor bsl5k must remain visible. Preserve original FPS for CiCo; do not inherit SEDS's 24 FPS silently. Domain-mixture weights/configurations and crop selection need versioned manifests. Features are regenerable; exact published reproduction remains conditional.

### 14. Decisions already made

D1. No SEDS/Baidu hard dependency.

D2. Dataset/benchmark creation, scale-only changes and module stacking are outside the contribution scope.

D3. Select exactly **OCEM**, conditional on falsification gates. The other four remain documented alternatives.

D4. Start with official CiCo rebuilt from raw resources. Matched UPRet is needed for a stronger fair claim. Primary full galleries: H2S and P14T; CSL conditional; OpenASL/BOBSL separate.

D5. Keep uncertainty explicit. Do not invent CMCM/GTRN results. C²RL is TCSVT; Scaling up is TPAMI; SAN fine retrieval is not standard retrieval.

D6. Reproduce the neural baseline before neural OCEM training. Legacy protocol first; common duplicate/tie controls second. No SOTA claim presently.

D7. Local projection precedes visual context. Capacity is a falsifiable prior, not a linguistic law. Use the same full-gallery scoring rule at train/test; no T2V candidate captions.

D8. Kill the support-specific thesis if ordinary OT, independent caps, a matched local head or protocol correction explains the gain. Do not rescue it by adding modules.

### 15. Outstanding questions

**CRITICAL:** Can raw P14T/H2S integrity and CLIP parity/loading be established? Can the downloaded Oxford model be loaded? Can the annotation snapshot be reconciled with the exact 2,348 usable-video manifest? Can CiCo reproduce both R@1 directions within 1pp on matching IDs? Are baseline false matches more concentrated after length/lexical controls? Does this persist in the pre-contextual local head? Does real A beat ordinary OT/shuffled A? Does CMCM invalidate novelty or change the strongest fair target?

**IMPORTANT:** GPU solver accuracy/latency; null-mass calibration; harm on compact translations/simultaneous cues; clip boundaries; duplicate/tie impact; parameter/update/negative confounds; three-seed cross-dataset stability; UPRet resource provenance.

**OPTIONAL:** CSL access; SAN runnable release/fixed fine candidates; C²RL author code; complete OpenASL manifest; GTRN full paper; BOBSL transfer. No external contacts or permission-dependent downloads are assumed.

### 16. Exact next actions

1. Use the completed report and evidence package. Resolve only outstanding CMCM full text or concrete new competing-method leads; do not restart broad searches or regenerate rejected ideas.
2. **Gate 0:** provision sufficient raw-data storage and a PyTorch environment, then complete official raw-data integrity and CLIP transfer/parity checks; load the already downloaded Oxford checkpoint. Freeze environment, recorded repository commits, crops, timestamps, tokenizer and IDs. Reconcile the downloaded H2S annotations with the exact filtered manifest before claiming cross-dataset reproducibility. No unofficial feature archives.
3. Regenerate training-only domain adaptation/features using CiCo, then reproduce legacy retrieval. Require each direction `|R1_reproduced−R1_reported|≤1.0pp`, unless official variance supports another tolerance. If it fails, diagnose features, manifests, crops, normalization, batch denominators, temperature, schedule and versions. No SOTA claim.
4. Establish common exact-caption negative and tie controls. **Stage A:** compare false near-miss concentration with matched true pairs on validation, controlling length and lexical similarity. Require a paired 95% CI above zero and ≥10% of baseline errors affected. If absent, kill OCEM. Contextual maps are only a proxy.
5. **Stage A2:** train the matched unconstrained local head on training pairs; repeat the diagnostic with actual pre-contextual supports. If absent here, kill the shared-support interpretation.
6. Port the CPU dual/envelope gradient to batched GPU and validate certificates. **Stage B:** two seeds; minimal OCEM versus a parameter-matched local head, ordinary partial OT, independent window caps and true/shuffled A. Hold negative pool and updates fixed. Require ≥1pp mean bidirectional validation R@1 over the strongest simple control, no direction below −0.5pp, and true A better than shuffled A. Otherwise diagnose once or stop.
7. **Stage C:** full same-score training; matched UPRet and negative-policy controls; whole-gallery latency/memory. Hypothesis ablations: global/local/mixed, κ→∞, null, centering, q, ordinary OT, independent caps, shuffled support and parameter-matched heads.
8. **Stage D:** P14T+H2S with three fixed seeds; positive effect on each dataset, no direction below −0.5pp, aggregate paired 95% CI above zero. **Stage E:** target ≥1pp average gain over four R@1 cells versus the strongest fair baseline, with mechanism/shortcut/length/signer/caption/template analyses. Existing data only. Reject the specific thesis if ordinary OT matches or κ→∞ wins.
9. Update probability only after real evidence. A/B success may support 50–70%; 70–85% requires replicated cross-dataset evidence. Report all seeds and failures. Save a new checkpoint after each gate.

### 17. Continuation instruction

> RESUME RULE:
> When this checkpoint is supplied in a future session, treat it as the authoritative current state of the research. Do not restart the literature review, regenerate already rejected ideas, or change established facts without new evidence. Continue directly from the "Exact next actions" section while still verifying any new factual claims through primary sources.
