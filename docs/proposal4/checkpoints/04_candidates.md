# RESEARCH STATE CHECKPOINT

Literature cutoff date: 10 September 2026. Checkpoint 4: exactly five candidate methods.

### 1. Current research objective

Develop one defensible methodological contribution to sentence-level Sign Language Retrieval (T2V/V2T), with fine-discrimination diagnostics on existing datasets. No new dataset/benchmark contribution, no invented results, fair supervision/input comparisons, no dependency on inaccessible SEDS checkpoints or SEDS/Baidu precomputed features. Prefer raw public data, public preprocessing and independently extracted representations. Research proposal, not an empirical claim that training has been completed.

### 2. Current phase

Completed: multi-query seed search; primary full-text reading of SPOT-ALIGN, CiCo, UPRet, SEDS, C²RL, SAN, CSLR² and Scaling up Multimodal Pre-training; publisher/author verification of CMCM and GTRN. Literature review is bounded, not an assertion of exhaustive coverage. CMCM full text and GTRN full PDF remain inaccessible. Dataset/protocol/resource audit completed with explicit access limitations. Completed: failure taxonomy and gap ranking. Completed: exactly five candidates and provisional scoring. Current: adversarial novelty audit of candidate 1. No final architecture accepted.

### 3. Verified papers

Reported recall is percent; pairs below are T2V/V2T R@1. Full R@5/10 values retained in local source tables and will be included in the final report.

| Paper | Venue / source | Method and evaluation | Published R@1 | Code/resources |
|---|---|---|---|---|
| Sign Language Video Retrieval with Free-Form Textual Queries | CVPR 2022; https://arxiv.org/abs/2201.02495 | SPOT-ALIGN: iterative spotting/alignment, I3D, cross-modal and sign-recognition fusion; H2S/P14T | H2S 34.2/23.6; P14T 55.8/53.1 | https://imatge-upc.github.io/sl_retrieval/; full reproduction resource chain still being audited |
| CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning | CVPR 2023; https://arxiv.org/abs/2303.12793 | BSL-pretrained I3D + target pseudo-label adaptation; CLIP-initialized temporal/text encoders; independent row/column softmax token aggregation | H2S 56.6/51.6; P14T 69.5/70.2; CSL 75.3/74.7 | https://github.com/FangyunWei/SLRT/tree/main/CiCo; regeneration scripts; Oxford backbone byte probe succeeds |
| A Tale of Two Languages: Large-Vocabulary Continuous Sign Language Recognition from Spoken Language Supervision | 2024 arXiv version verified; https://arxiv.org/abs/2405.10266 | CSLR²: Video-Swin-Tiny, frozen T5, HN-NCE sentence/sign retrieval, BOBSL pseudo-sign labels | BOBSL Sent-Test 29.4/28.1; R@5 45.2/44.9; R@10 51.5/51.0 | Final venue/code/checkpoints not yet verified |
| Uncertainty-aware Sign Language Video Retrieval with Probability Distribution Modeling | ECCV 2024; https://arxiv.org/abs/2405.19689 | UPRet: CiCo backbone, Gaussian sampling, OT; paper explicitly says OT score is training-only | H2S 59.1/53.4; P14T 72.0/72.0; CSL 78.4/77.0 | https://github.com/xua222/UPRet; repository verified; trained binary/complete recipe not verified |
| SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval | ACM MM 2024; https://arxiv.org/abs/2407.16394; DOI 10.1145/3664647.3681237 | Frozen I3D, RTMPose, SignBERT-initialized hand GCN, local CGAF and pose/RGB matching | H2S 62.5/57.9; P14T 76.8/78.7; CSL 85.8/85.4 | https://github.com/longtaojiang/SEDS (master); Baidu artifacts UNAVAILABLE / DO NOT DEPEND ON |
| C²RL: Content and Context Representation Learning for Gloss-Free Sign Language Translation and Retrieval | TCSVT 35(9), 2025, 8533–8544; DOI 10.1109/TCSVT.2025.3553052; full text https://arxiv.org/abs/2408.09949 | ResNet18/ImageNet + temporal conv; ICL=CLCL, ECL=autoregressive translation; frozen learned features + two independent mBART encoders | H2S 62.4/57.5; P14T 78.7/77.6; CSL 90.3/88.4; OpenASL table 62.2/61.6 | No author-verified working code/checkpoint located. Journal metadata verified with Crossref. Do NOT mislabel as TPAMI. |
| Scaling up Multimodal Pre-training for Sign Language Understanding | TPAMI 47(12), 2025, 11753–11767; DOI 10.1109/TPAMI.2025.3599313; https://arxiv.org/abs/2408.08544 | Pose-based sign/text contrast + masked pose modeling; external SL-1.5M corpus | P14T 74.5/75.1; CSL 87.5/87.2 | Code/checkpoint not verified; separate external-data setting |
| Graph traverse reference network for sign language corpus retrieval in the wild | Neurocomputing 637, 7 July 2025, 130077; DOI 10.1016/j.neucom.2025.130077 | GTRN: visual query/document retrieval, hierarchical frame/body-part reference attention | UNVERIFIED — DO NOT USE AS FACT | Author institution https://ro.ecu.edu.au/ecuworks2022-2026/6028/ verified; PDF fetch 403; dataset/backbone numbers unverified |
| Causality-inspired multi-grained cross-modal sign language retrieval | CVIU 264, February 2026, 104631; DOI 10.1016/j.cviu.2025.104631 | CMCM: publisher highlights backdoor adjustment, causal Gaussian alignment and temporal-motion covariance pooling | UNVERIFIED — DO NOT USE AS FACT | Publisher abstract/metadata verified; full text 403; a paper-associated repository is now located at https://github.com/vddong-zjut/CMCM; complete author linkage, recipe and trained resources remain unverified |
| Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative Mining for Sign Language Retrieval | ACL 2026 long papers verified by ACL author/accepted-paper pages; https://arxiv.org/abs/2607.09263 | SAN: confident local sign–word pairs, visually similar different-word substitutions; extra V2T contrastive objective; P14T only | CiCo fine V2T 17.9→39.4; standard T2V 69.2→68.1, V2T 70.1→67.8. GFSLT fine 16.8→49.1; standard 67.9→70.2 / 69.4→67.4 | https://github.com/joonmy/SAN; raw README says Coming Soon, no reproducible release verified |

Adjacent leads verified at abstract level, not fully audited: SignRep (arXiv:2503.08529), SHuBERT (2411.16765), SEA (2512.08094), Lost in Translation, Found in Embeddings (2512.08040). Other index-only leads are not established facts.

Additional direct downstream study: Jang, Choi, Ahn, Chung, “Deep Understanding of Sign Language for Sign to Subtitle Alignment” (arXiv:2503.03287, 2025; final venue unverified), https://arxiv.org/html/2503.03287v1 ; official code linked by paper https://github.com/art-jang/sign-to-subtitle . Selective alignment loss + language-specific subtitle processing + self-training; uses limited manually aligned training data. Its CSLR² downstream comparison reports T2V R@1/5 27.14/42.19→28.59/43.74 and V2T 26.25/41.99→26.59/42.51. Different subtitle supervision; not a new fair H2S/P14T SOTA row.

### 4. Current SOTA understanding

No single fair leaderboard. Separate (i) BSL-sign-pretrained RGB+CLIP CiCo/UPRet, (ii) RGB+pose SEDS, (iii) target-pair-only sign learning + ImageNet/mBART C²RL, (iv) SL-1.5M pose pretraining, (v) SAN’s synthetic-caption diagnostic, (vi) BOBSL 20K retrieval, (vii) visual-query corpus retrieval. CMCM unknown numbers prevent an exhaustive current-SOTA assertion. C²RL and SEDS are cross-resource references, not automatically A-fair comparisons. UPRet reproduces CiCo H2S V2T at 50.3 vs original 51.6, already outside the suggested 1-point reproduction gate.

### 5. Dataset/protocol state

H2S: original 31,164/1,740/2,356; CiCo and C²RL say 31,085/1,739/2,348; SPOT-ALIGN says 31,075/1,739/2,348; SEDS says 31,019/1,738/2,348. Use exact sample manifests, not counts. Original downloadable sentence clips use old timing; CiCo requires full frontal videos recut with manually realigned timestamps. P14T: canonical 7,096/519/642; C²RL prose says 7,098 but stated total 8,257 contradicts that. CSL: 18,401/1,077/1,176; access agreement required. OpenASL: C²RL says 96,476/966/975; original paper discusses 967/976 before filtering and reports repeated train/eval captions. BOBSL: Sent-Val 1,973 and Sent-Test 20,870 (paper prose reverses count order; tables make 2K/20K clear). How2Sign publication reports 11 total signers, nine green-screen signers, eight/five/six in train/val/test and mostly signer-overlapping evaluation. Paper describes gloss but public gloss release is not verified. Native released keypoints are automatically estimated, not manual pose labels. P14T has 9 interpreters, 25 FPS, 210×260 frames, sequence gloss and cleaned German transcript. CSL-Daily access requires a signed institutional agreement. OpenASL is about 288 hours and ~220 signers; official downloader warns of missing videos. Its original paper reports 10.9% dev and 10.6% test captions repeated in training; these rates are not assumed identical in C²RL’s filtered manifest. BOBSL is about 1,467 hours/39 signers; CSLR² trains on 689K automatically signing-aligned subtitle pairs. Original-gallery ID recall must remain separate from equivalence-class diagnostics. Static SEDS metrics.py inspection finds its legacy compute_metrics collects every score tied with the diagonal, changing the rank-array denominator; dataset-level impact has NOT been measured. Code source: https://raw.githubusercontent.com/longtaojiang/SEDS/master/metrics.py .

New audit results on 10 September: official H2S raw-video first part returns HTTP 206 binary bytes through the public download confirmation; full file is 32,212,254,720 bytes. Train/dev TSVs downloaded in full. Parsed snapshot has 31,165/1,741 unique sample-name rows, no blank captions or nonpositive durations; these are NOT the filtered standard retrieval manifests. Exact caption duplicates affect 1,509 train rows and 444 dev rows (duplicate excess 1,056/225); 47 dev captions occur in train; train/dev VIDEO_ID intersection is zero. This does not establish signer or near-duplicate leakage. Test TSV returns Google Drive quota error; no bypass or unofficial replacement. CMCM repository modules downloaded: missing DEVICE definition in one module and inconsistent encoder tensor shapes mean a complete runnable release is not established; static observations do not invalidate published experiments.

### 6. Confirmed findings

F1: Fine-grained success is distinct from standard retrieval success. Evidence: SAN Table 1, above. Implication: optimize both with separate outcome gates.

F2: Published preprocessing manifests differ. Evidence: explicit source counts above. Implication: matching R@1 alone cannot certify fairness.

F3: SEDS’s released artifacts are a hard practical constraint. Evidence: official README points only to Baidu features/checkpoints; retrieval failed; user explicitly forbids dependence. Implication: regenerate features independently.

F4: UPRet already imports Gaussian uncertainty and OT into SLRet. Evidence: full text §3. Implication: generic probabilistic or OT additions are not novel.

F5: C²RL already combines retrieval alignment and generative context learning. Evidence: full text §III. Implication: generic contrast+translation is not a contribution.

### 7. Open hypotheses

H1 (OPEN): Independently aggregated local alignments may reuse overlapping visual evidence for distinct text units; importance in real retrieval errors has NOT been measured. Test on frozen baseline and validation-only diagnostics.

H2 (OPEN): Hard-negative supervision may improve local discrimination while damaging coarse ranking because substitution labels and alignment confidence are imperfect. Evidence: SAN trade-off; alternative cause is optimization. Test real-gallery negatives and matched control objectives.

H3 (OPEN): Weak timing and repeated captions may materially distort cross-paper gains. Evidence: dataset protocols and original OpenASL duplicate analysis. Quantify from official annotations without changing test pool.

### 8. Research gaps under consideration

G1 refers to evidence for the *failure*, not proof of the proposed causal explanation. “Conditional” requires a frozen-baseline measurement before method training. No fully verified new causal gap has yet been established.

| Rank | Gap | Evidence / partial solutions | G1–G5 outcome | Resources / risk |
|---|---|---|---|---|
| 1 | Local similarity overcounts correlated overlapping clip evidence in near-miss retrieval | SAN proves fine discrimination failure; CiCo independent token aggregation provides a possible cause. UPRet already uses OT. The overcounting explanation is unmeasured. | Conditional G1; G2 open; G3–G5 pass for a cheap test | Same raw RGB and timestamps; low access risk, medium-high novelty risk |
| 2 | Fine discriminative training fails to preserve standard bidirectional ranking | SAN’s own fine/coarse results show a trade-off; optimization versus label noise unresolved | Failure G1 verified; causal G2 open; G3–G5 pass | Same pairs; no synthetic labels required; hard-negative prior art crowded |
| 3 | Retrieval treats uncertain clip boundaries as exact | SPOT timing ablation; CSLR² alignment ablation; Jang et al. 2025 downstream retrieval gains | G1 verified in weakly aligned corpora; G2 crowded by SAT/SEA; G3–G5 pass within fixed clips | Raw videos accessible routes; headroom on realigned H2S/P14T unknown |
| 4 | Short local relations survive neither bag-like matching nor global sentence pooling | SAN/SignCL support local ambiguity, not an order-specific SLRet error rate | Conditional G1; G2 crowded by relational/GW models; G3–G5 technically pass | Existing RGB; nonconvex engineering risk; no monotonic spoken/sign-order assumption |
| 5 | Positive-ID losses penalize exact duplicate or genuinely equivalent captions | OpenASL original duplicate analysis; semantic many-to-many is plausible | G1 verified duplication, unmeasured retrieval harm; G2 heavy false-negative prior art; G3–G5 pass | Train captions only; no LLM paraphrase labels; modest novelty |
| 6 | RGB misses handshape/non-manual details | SEDS modality ablations and sign-language structure | G2 fails for generic pose/RGB fusion | Reject generic fusion; pose is extra resource |
| 7 | Signer/background shortcuts dominate learned embeddings | Dataset design exposes risk; SignRep already uses style-adversarial pretraining | G1 not established for current retrieval; G2 collision | Defer; no claim of confirmed signer leakage |
| 8 | Global embeddings ignore fine sign-word structure | CiCo already directly addresses it | G2 fails | Reject |
| 9 | Semantic negatives are visually easy | SAN directly addresses local visual confusability | G2 fails | Reject |
| 10 | Generic embedding uncertainty | UPRet and CMCM | G2 fails | Reject |
| 11 | Larger language model or translation objective solves semantics | C²RL already uses both | G2 fails / forbidden scale-only contribution | Reject |
| 12 | Split/tie/duplicate evaluation inconsistencies create spurious SOTA | Published counts, inspected legacy metric code | G1 yes; algorithmic-contribution test fails | Protocol audit obligation, not proposed paper contribution |

Newly verified adjacent work: SignCL (NeurIPS 2024, arXiv:2405.14312) samples adjacent positives and temporally distant negatives; SEA (ACL 2026, arXiv:2512.08094v2) pretrained sign segmentation + embedding + dynamic-program alignment; VTaMo (arXiv:2607.09126) uses uniform-marginal OT, a null token, temporal variation, and target-guided feature reordering during training. DualAnchor (2607.27614) partial-OT lead; AVIOT (2608.20473) dense-to-compact transport lead, full PDFs being inspected. ColBERT (2004.12832), Norton (2401.16702), Drop-DTW (2108.11996) further limit generic late interaction / null-alignment novelty.

### 9. Rejected ideas

Reject generic RGB+pose fusion (SEDS); generic Gaussian+OT (UPRet); visual-hard negative word substitution (SAN); contrastive+autoregressive pretraining (C²RL); generic causal multi-grained network (CMCM); larger backbone/data-only contribution; any unavailable SEDS dependency. Do not regenerate these without a substantially distinct mechanism.

### 10. Candidate methods

**C1: Overlap-Constrained Evidence Matching (OCEM)**

Hypothesis: Near-miss video/text pairs obtain excessive local scores by reusing overlapping clip evidence; limiting aggregate alignment load on actual observed time should improve R@1 when erroneous pairs are more concentrated than correct pairs.

Mechanism: Retain the reproduced CiCo score. Project pre-contextual I3D window features into the text space. Solve an entropy-regularized partial assignment with null text mass and linear capacity constraints on shared raw-time support; use its centered optimal value during both training and inference.

Constraints: No monotonic word/sign ordering, no one-word/one-sign assumption, no pose, no synthetic captions, no SEDS artifacts. Physical supports apply only to local pre-contextual features.

Closest prior work: CiCo, UPRet, VTaMo. Risk-adjusted decision score: 60.4/100 (subjective utility, NOT probability). Status: LEADING HYPOTHESIS; adversarial novelty search pending.

**C2: Rival-Conditional Evidence Scoring (RCES)**

Hypothesis: Shared caption content hides the local distinction between visually confusable gallery candidates; explicitly contrasting unmatched content from real paired captions should improve R@1 when those differences are visually expressed.

Mechanism: For a query and real training-caption rival, align text tokens, softly remove shared content, then compare residual text-video support. Train a pairwise preference head with original video/text IDs. At inference average residual margins against a fixed top-K rival shortlist, preserving an independent base score.

Constraints: No synthetic negative sentences; rival captions are ordinary gallery text only for V2T. T2V cannot use candidate-video ground-truth captions; requires a video-side residual head trained with paired data and used without captions at test. This modality asymmetry is a major risk.

Closest prior work: SAN, CSLR2, Verbs in Action. Risk-adjusted decision score: 46.0/100 (subjective utility, NOT probability). Status: SURVIVES AS CANDIDATE; high risk.

**C3: Boundary-Marginalized Retrieval (BMR)**

Hypothesis: A single exact clip boundary overweights transitions or neighboring content; integrating retrieval evidence across plausible within-clip boundaries should improve retrieval when annotation boundary uncertainty is substantial.

Mechanism: Encode five deterministic interior spans of each official clip, define a text-conditioned latent-span posterior and a prior-weighted log-sum-exp retrieval score, and train with symmetric contrast plus full-span preservation. No extra context outside the official sample.

Constraints: No new boundary annotations; raw-time jitter is within the existing sample. All baselines receive identical preprocessing and optional span ensemble controls.

Closest prior work: SPOT-ALIGN, SEA, Norton. Risk-adjusted decision score: 46.6/100 (subjective utility, NOT probability). Status: SURVIVES; likely incremental on carefully realigned datasets.

**C4: Relation-Consistent Late Interaction (RCLI)**

Hypothesis: Local token similarity cannot distinguish sequences with similar units but different relations; matching learned intra-video motion relations to intra-text semantic relations should improve compositional R@1 where relation errors are present.

Mechanism: Learn low-rank video and text relation matrices, jointly optimize unary token similarity and relational consistency through a fused Gromov-Wasserstein-type assignment, and use the same optimized score at test. Do not equate signed order with spoken word order.

Constraints: No parser, gloss, pose or additional language model. The relation semantics must emerge from paired retrieval data, making identifiability weak.

Closest prior work: CiCo, CMCM, Gromov-Wasserstein Learning for Graph Matching and Node Embedding. Risk-adjusted decision score: 33.4/100 (subjective utility, NOT probability). Status: SURVIVES; nonconvexity and limited relation supervision penalized.

**C5: Reliability-Bounded Positive Sets (RBPS)**

Hypothesis: ID-only contrastive targets suppress valid semantic alternatives; training against conservative positive equivalence sets should improve recall without semantic collapse when duplicate/ambiguous captions account for substantial errors.

Mechanism: Make exact-caption duplicates a shared control, then estimate additional positive confidence from out-of-fold bidirectional agreement and bound each added pair's weight. Optimize a robust multi-positive objective and score candidates with the learned independent embeddings.

Constraints: No test labels or LLM paraphrases. Additional confidence must be cross-fitted from training pairs. Semantic equivalence is not established by embedding agreement alone.

Closest prior work: UPRet, CrossCLR, Norton. Risk-adjusted decision score: 42.9/100 (subjective utility, NOT probability). Status: SURVIVES as alternative; duplicate handling itself is a control, not a novelty claim.

### 11. Current leading method

C1 OCEM is the leading hypothesis. It ranks above alternatives because a local scoring intervention with a convex inner problem can be tested without new labels or pretrained resources. Its putative causal failure has not been measured, and generic transport/null alignment is already prior art. The candidate is not accepted until novelty and reviewer checks.

### 12. Novelty audit state

Completed field queries: all eight requested retrieval variants; exact seed titles; 2025/2026 How2Sign/SLRet searches; SciSpace and Consensus discovery; backward references in seeds, CMCM Crossref bibliography; forward title/DOI/repository searches. Search failures or missing indexed results are NOT novelty evidence. Next search mechanism-specific prior art after gap ranking.

New resource checks: official H2S realigned training annotation responds with binary TSV bytes (5,607,385-byte object); first large full-video part returns the normal Google Drive virus-scan confirmation page, not video bytes. Full public confirmation/download remains to be checked. No new access authorization is assumed.

### 13. Resource availability

| Resource | Status | Source | Needed? | Alternative |
|---|---|---|---|---|
| Oxford BSL5K I3D | HTTP 206: first 1,024 bytes of 142,594,302-byte object verified; full transfer/load pending | https://www.robots.ox.ac.uk/~vgg/research/bslattend/data/bsl5k.pth.tar | Candidate baseline | ImageNet ResNet18 route, separate setting |
| P14T raw v3 | HTTP 206 first bytes verified; object 41,699,758,035 bytes | https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz | Primary dataset | No unofficial mirror |
| CLIP ViT-B/32 | Official HF weight range response verified (605,247,071 bytes) | https://huggingface.co/openai/clip-vit-base-patch32 | Candidate baseline | Official OpenAI CLIP implementation |
| ImageNet ResNet18 | Official range response verified (46,830,571 bytes) | https://download.pytorch.org/models/resnet18-f37072fd.pth | Fallback | Regenerate supervised baseline with same resources |
| mBART-large-cc25 | Config bytes verified; model binaries pending | https://huggingface.co/facebook/mbart-large-cc25 | C²RL reproduction option | Not needed for CiCo route |
| H2S full frontal videos/realigned annotations | Official portal/script accessible; full download pending | https://how2sign.github.io/ | Cross-dataset primary | Author/institutional release only |
| CSL-Daily | Access agreement, not anonymous download | https://ustc-slr.github.io/datasets/2021_csl_daily/ | Extension after access | H2S/P14T core |
| OpenASL | Official script warns videos disappear | https://github.com/chevalierNoir/OpenASL | Optional | Do not claim unchanged-pool SOTA on a partial gallery |
| SEDS Baidu checkpoints/features | UNAVAILABLE / DO NOT DEPEND ON | https://github.com/longtaojiang/SEDS | NO | Self-extracted RGB/pose if comparison needed |

### 14. Decisions already made

D1: No SEDS artifacts. D2: No new dataset/benchmark. D3: No unsupported SOTA number. D4: SAN diagnostic cannot be relabeled a standard retrieval result. D5: C²RL venue is TCSVT, not TPAMI. D6: A successful portal or range request is not proof of full archive integrity or reproducibility.

### 15. Outstanding questions

CRITICAL: CMCM full methodology/results and exact resources; inspect newly located repository modules; establish runnable strong baseline; full raw-data transfer/manifests; real magnitude of any proposed failure. IMPORTANT: C²RL official code and journal-version numerical agreement; actual duplicate/tie behavior in retrieval implementations; confirm final venues and code for CSLR². OPTIONAL: further dictionary retrieval transfer and very large corpora.

### 16. Exact next actions

1. Adversarially test OCEM against partial/null OT, capacity-constrained OT, redundancy-aware retrieval and support/provenance transport.
2. Keep the newly discovered metrics tie issue diagnostic, not the paper contribution.
3. Preserve the ranked gaps above; kill mechanism 1 if its proposed evidence concentration does not characterize real validation errors.
4. Search adjacent alignment, temporal segmentation, transport, false-negative and invariance literature.
5. Retain exactly C1–C5 definitions above; do not introduce a sixth candidate or silently alter their claims.
6. Attempt to falsify the leading candidate’s novelty; simulate three reviewers; only then accept one conditional specification.

### 17. Continuation instruction

> RESUME RULE:
> When this checkpoint is supplied in a future session, treat it as the authoritative current state of the research. Do not restart the literature review, regenerate already rejected ideas, or change established facts without new evidence. Continue directly from the "Exact next actions" section while still verifying any new factual claims through primary sources.
