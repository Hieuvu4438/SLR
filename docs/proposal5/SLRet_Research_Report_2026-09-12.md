# Sign Language Retrieval: Evidence, Research Gaps, and Implementable Methods

## 1. Executive Summary

**The strongest defensible opportunity is to improve the visual evidence used to distinguish confusing language without damaging ordinary sentence retrieval.** It is not enough to add another global embedding, local alignment module, pose stream, or hard-negative miner: all are established directions. Current systems already contextualize clip and text tokens; their remaining weaknesses concern what is aligned, which differences receive supervision, and whether improvements survive changes in data processing and pretraining.

**FACT FROM SOURCE:** SAN, ACL 2026, substantially improves its PHOENIX-2014T fine-grained stress test. However, its CiCo variant changes ordinary T2V/V2T R@1 from 69.2/70.1 to 68.1/67.8. Fine-grained discrimination and full-pool retrieval therefore remain a measurable tradeoff in at least one strong published experiment. This does not prove that representation drift is the cause. [SAN, Table 1 and §5](https://aclanthology.org/2026.acl-long.1302/)

**Current frontier, with qualifications:** the largest verified How2Sign R@1 values in the inspected tables are SEDS 62.5 T2V and 57.9 V2T; C²RL reports 62.4/57.5 and competitive higher-rank recalls. On PHOENIX-2014T, C²RL reports T2V 78.7, while SEDS reports V2T 78.7. C²RL reports CSL-Daily 90.3/88.4 and OpenASL 62.2/61.6. These are not a single controlled leaderboard: resources differ, C²RL figures were verified in its 2024 arXiv version rather than the final journal tables, and numerical results for the relevant 2026 CMCM paper remain **UNVERIFIED**. An exhaustive “true SOTA” certification is therefore not justified. [SEDS](https://arxiv.org/abs/2407.16394), [C²RL](https://arxiv.org/abs/2408.09949), [CMCM](https://doi.org/10.1016/j.cviu.2025.104631)

**Method 1 — Shared-support sign contrast:** start from an independently trained UPRet baseline using the published CiCo feature regime. Mine sign-aware alternatives from training pairs, then compare the changed text spans against exactly the same reference-selected visual support. Retain the original retrieval objective; add a bounded-margin visual discrimination loss. The proposed contribution is **holding the evidence constant when contrasting alternatives**, not local hard negatives themselves. The mandatory adversarial control is FSC-CLIP-style local negatives with the identical SAN-style miner. No SEDS-trained parameters are used.

**Method 2 — Sampling-consistent partial alignment:** replace independent local matching with a partial transport score whose capacities derive from physical video coverage and word-level mass, then stabilize correspondences across two ordered samplings of the same video using an EMA teacher. Public SuperGlue and ALBEF implementations supply the numerical solver and teacher update. Neither OT, null matching, nor teacher regularization is claimed as new: VTaMo and DualAnchor already make generic OT-for-sign-alignment claims obsolete by July 2026. The narrower hypothesis is that retrieval should depend on signing evidence, not the number of overlapping observations used to represent it. [VTaMo](https://arxiv.org/abs/2607.09126), [DualAnchor](https://arxiv.org/abs/2607.27614)

**Primary recommendation:** Method 1. It targets stronger experimental evidence, changes training rather than deployment, and has a clean falsification experiment. Confidence of improving the strongest *reproduced, resource-matched* control is **Moderate**. Confidence of exceeding every published system across all datasets is **Low**. Both proposals are implementation-ready research hypotheses, not experimentally validated or guaranteed publication-ready methods.

Throughout, **FACT FROM SOURCE** denotes paper, repository, or annotation evidence; **AUTHOR CLAIM** denotes an interpretation made by the authors; **YOUR INFERENCE** denotes independent analysis; **YOUR HYPOTHESIS** denotes an untested explanation or proposed mechanism. Unavailable information is marked **UNVERIFIED**. Repository findings are static inspections, not reproduced training results.

## 2. What Exactly Is Sign Language Retrieval?

### Mathematical definition

A sign video is \(V=(I_1,\ldots,I_F)\), where \(I_t\in\mathbb R^{H\times W\times3}\). Optional extracted observations include pose \(P\in\mathbb R^{F\times J\times c}\), with \(J\) landmarks and \(c\) coordinate/confidence channels. Text is \(T=(w_1,\ldots,w_M)\), usually a written-language translation of the signed utterance, not a transcription of signs in written-word order.

A video encoder produces contextual features
\[
X=f_\theta(V,P)\in\mathbb R^{L\times d},
\]
and a text encoder produces
\[
Y=g_\phi(T)\in\mathbb R^{M'\times d}.
\]
\(L\) may count clips rather than frames; \(M'\) may count subwords rather than words. Global embeddings \(v=\operatorname{norm}(\operatorname{pool}(X))\) and \(t=\operatorname{norm}(\operatorname{pool}(Y))\) permit cosine similarity \(s_{\rm global}=v^\top t\). A late-interaction system instead computes \(A=XY^\top\), after row-wise feature normalization, and aggregates token–clip affinities.

For paired examples \(\{(V_i,T_i)\}_{i=1}^{B}\), a conventional symmetric contrastive objective is
\[
\mathcal L_{\rm ret}=-\frac1{2B}\sum_i
\left[
\log\frac{\exp(s(V_i,T_i)/\tau)}
{\sum_j\exp(s(V_i,T_j)/\tau)}
+
\log\frac{\exp(s(V_i,T_i)/\tau)}
{\sum_j\exp(s(V_j,T_i)/\tau)}
\right].
\]
\(\tau>0\) is the retrieval temperature. This formula assumes one positive index per row; it is not automatically correct for repeated captions or multiple relevant videos. A relevance set \(\mathcal P(i)\) can replace the single numerator with a sum over known positives. Unknown semantic equivalence must not be silently converted into positive labels.

**T2V:** rank candidates in a fixed video pool \(\mathcal C_V\) by \(s(V,T_q)\). **V2T:** rank candidates in a fixed text pool \(\mathcal C_T\) by \(s(V_q,T)\). In group-based implementations, a video candidate is a predefined set of recordings, and its score may be the maximum over its members. The pool and relevance relation are part of the mathematical task, not incidental loader details. [Original task formulation](https://arxiv.org/abs/2201.02495), [CiCo](https://arxiv.org/abs/2303.12793)

For query \(q\), let \(r_q\) be the first rank of an annotated relevant candidate. Then
\[
R@K=\frac{100}{Q}\sum_q\mathbf1[r_q\le K],\qquad
\operatorname{MedR}=\operatorname{median}_q r_q,\qquad
\operatorname{MRR}=\frac1Q\sum_qr_q^{-1}.
\]
Mean rank is \(\frac1Q\sum_qr_q\). MedR becomes nearly uninformative once more than half of queries rank a positive first. R@1/5/10 should remain the established headline metrics; MRR and mean rank help diagnose tails where already supported.

### Why ordinary action retrieval is an inadequate mental model

Sign languages are natural languages with their own lexicons, grammars, variation, and discourse structure. Their visual-manual realization carries linguistic distinctions that may be much smaller than the object/action changes separating ordinary video captions. Neither ASL, DGS, BSL, nor Chinese Sign Language is a universal visual rendering of its surrounding spoken language. [PHOENIX-2014T introduction](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html), [How2Sign](https://how2sign.github.io/), [SignCLIP](https://aclanthology.org/2024.emnlp-main.518/)

| Information | Linguistic significance and retrieval implication |
|---|---|
| Hand shape | Finger configuration can distinguish lexical items. Generic full-frame features and sparse hand estimates may collapse these differences. |
| Orientation | Palm/finger orientation matters; 2D landmarks alone do not uniquely determine 3D orientation. A pose stream is not a complete hand representation. |
| Location | Body-relative articulation and signing-space locations can encode lexical distinctions and reference. Removing all location information in the name of invariance is wrong. |
| Motion | Direction, trajectory, repetition, speed, and movement combinations can matter. Clip averaging can preserve a rough action while obscuring its linguistic realization. |
| Body posture | Posture and body shifts may encode role, discourse, or grammatical information as well as nuisance variation. |
| Facial/non-manual markers | Eye gaze, brows, mouth, head, and torso cues may mark grammar, prosody, lexical content, or discourse. They can co-occur with several manual units and span different durations. |
| Temporal order | The arrangement and scope of signs matter. A permutation-invariant final aggregator does not erase all order if its inputs are contextualized, but it does not itself enforce compositional correctness. |
| Signer variation | Identity, clothing, camera, body proportions, handedness, and signing style vary. Invariance must preserve linguistically meaningful body-relative geometry and motion. |
| Semantic equivalence | Different sign realizations and different written translations may express similar content; index-based supervision observes only part of this relation. |
| Visual confusability | Similar-looking signs may have different meanings. Conversely, semantic neighbors in a language model need not look alike when signed. |
| Granularity and ordering | One sign can correspond to several written words; several signs can realize a written expression; function words can be implicit; non-manual information can be simultaneous. Word-to-sign matching need not be monotonic or one-to-one. |

The articulatory distinctions above motivate feature design, but not every distinction has been isolated as a retrieval bottleneck on each benchmark. SEDS supplies modality ablations, SAN supplies visual-confusion examples, and CSLR² supplies qualitative examples of lexical/granularity mismatch. These are stronger evidence than treating generic linguistic plausibility as a demonstrated model failure. [SEDS](https://arxiv.org/abs/2407.16394), [SAN](https://aclanthology.org/2026.acl-long.1302/), [CSLR²](https://arxiv.org/abs/2405.10266)

**YOUR INFERENCE:** monotonic alignment may be appropriate between consecutive subtitle sentences in a broadcast, or between two time-resampled views of one video. It is not thereby appropriate between the written words and signs inside an utterance. Likewise, point embeddings are not logically incapable of representing ambiguity: probabilistic models need evidence of better ranking or calibration, rather than an argument based only on their parameterization.

## 3. Dataset and Evaluation Audit

### Established data and available supervision

Scales below are approximate where corpus editions and preprocessing differ. “Gloss available” does not mean that a gloss-free retrieval system used it.

| Dataset | Language/domain and scale | Videos, signers, text | Gloss and pose | Standard split / relevant edition |
|---|---|---|---|---|
| [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) | DGS; German weather broadcasts; about 11 h | 9 signers; short, repetitive-domain utterances; German translations; 25 FPS | Gloss available. Pose must be extracted for a pose method. | 7,096 train / 519 dev / 642 test. Do not substitute the different PHOENIX recognition split. |
| [How2Sign](https://how2sign.github.io/) | ASL; instructional material; about 80 h | 11 signers; English text; multiple camera views, RGB/depth; sentences come from interpreted instructional content | No standard dense sentence gloss supervision. Body/hand/face keypoints available; a smaller portion has richer multiview/3D capture. | Original sentence counts approximately 31,164 / 1,740 / 2,356. CiCo release: 31,085 / 1,739 / 2,348 usable videos. SEDS states 31,019 / 1,738 / 2,348. |
| [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/) | Chinese Sign Language; everyday scripted topics; about 23 h | 20,654 videos, 10 signers; Chinese sentences; repeated textual prompts across recordings | Gloss available. Pose requires extraction. | 18,401 / 1,077 / 1,176 videos. Group-based retrieval uses fewer text groups than videos. |
| [OpenASL](https://github.com/chevalierNoir/OpenASL) | ASL; online signing with English subtitles; about 288 h | About 98,417 clips, more than 200 signers; broader domains; subtitle/cropping noise | No dense gloss standard. Pose extraction needed if used. | 96,476 / 966 / 975. Exact C²RL retrieval grouping and filtering: **UNVERIFIED**. |
| [BOBSL / CSLR²](https://github.com/gulvarol/cslr2) | BSL broadcast interpretation; BOBSL about 1,447 h | About 1.2 M subtitle sentences and 39 interpreters in corpus; CSLR² uses about 993 K training sentences / 1,220 h | Automatically obtained sign-level labels and carefully aligned evaluation material; not equivalent to gloss-free target-only pretraining | Paper: 2 K sentence validation and 20 K sentence test. Repository README advertises a 25 K aligned test resource: version compatibility **UNVERIFIED**. |
| [SpreadTheSign / SignCLIP](https://aclanthology.org/2024.emnlp-main.518/) | Dictionary material, 44 sign languages, roughly 500 K clips | Mostly isolated lexical entries, multilingual labels; not continuous sentence retrieval | Pose input extracted from dictionary videos; no sentence-level gloss requirement | Dictionary-specific splits and lexical retrieval; incompatible with sentence benchmark recalls. |

The How2Sign feature protocol is especially consequential. **FACT FROM SOURCE:** SPOT-ALIGN reports T2V R@1 of about 5.9±0.6 for speech-aligned captions versus 24.5±0.2 for signing-aligned captions in its cross-modal model. A change in temporal cropping can dominate a proposed architecture gain. The How2Sign project warns that realigned sentence timestamps require recutting the videos; a new annotation file alone does not change old crops. [SPOT-ALIGN, alignment experiment and appendix](https://arxiv.org/abs/2201.02495), [How2Sign resources](https://how2sign.github.io/)

### Candidate-pool forensics

**FACT FROM SOURCE, independently counted from public annotations:** the CiCo release stores caption groups, with one or more video IDs per group. Unique strings do not always coincide with group IDs.

| Release partition | Caption groups / records | Videos | Distinct stored text strings | Consequence |
|---|---:|---:|---:|---|
| How2Sign train | 30,852 | 31,085 | 30,041 | Exact repeated text survives across group IDs. |
| How2Sign test | 1,969 | 2,348 | 1,930 | Candidate groups and unique strings are different quantities. |
| CSL-Daily train | 6,598 | 18,401 | 6,573 | Repeated recordings are heavily grouped. |
| CSL-Daily test | 798 | 1,176 | 798 | Standard group-based text pool has 798 entries. |
| PHOENIX train | 7,096 records | 7,096 | 6,827 English strings | Diagonal-only training has repeated-string off-diagonal pairs. |
| PHOENIX test | 642 records | 642 | 630 English strings | Nominal one-pair protocol still contains repeated wording. |

These counts describe the inspected files, not a replacement corpus definition. [CiCo data and loaders at audited revision](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL)

For How2Sign and CSL-Daily, the inspected evaluator first computes video–text scores and constructs a tensor with dimensions approximately \([G,R_{\max},G]\). Its T2V path takes a maximum across the recordings belonging to **every** candidate group, then ranks the resulting \(G\) group scores. V2T ranks \(G\) caption groups for each individual video.

| Protocol family | T2V query/candidate definition | V2T query/candidate definition | Reported metrics | Direct comparison requirements |
|---|---|---|---|---|
| CiCo-family How2Sign | 1,969 caption-group queries against 1,969 video groups; group score is maximum member score | 2,348 video queries against 1,969 caption-group candidates | R@1/5/10, MedR; UPRet also mean rank | Same group file, realignment, eligible videos, language preprocessing, and score aggregation |
| CiCo-family CSL-Daily | 798 caption queries against 798 video groups | 1,176 videos against 798 captions | R@1/5/10, MedR; UPRet mean rank | Same grouping; native Chinese versus translated English must be declared |
| PHOENIX standard | 642 textual records against 642 video records | 642 video records against 642 text records | R@1/5/10; MedR/MRR/mean rank vary | Same split, caption language, duplicate policy, feature and checkpoint-selection provenance |
| SAN stress test | No comparable full-pool T2V headline | For each video: original caption plus 40 automatically perturbed captions | V2T R@1/5/10 and MRR | Only compare identical frozen stress candidates; these are not ordinary PHOENIX full-pool recalls |
| C²RL OpenASL | Declared established test split; exact construction **UNVERIFIED** | Same qualification | R@1/5/10 | Cannot assume a 975×975 similarity matrix without evaluator or manifest |
| CSLR² BOBSL | Sentence retrieval on paper's 20 K test set | Reverse retrieval on same set | R@1/5/10, MedR | Do not compare 2 K validation to 20 K test, or substitute README's 25 K edition |

Group ranking is not identical to “rank all videos and accept any positive”: duplicate distractor recordings collapse into one candidate group in the former. Main experiments should retain the established evaluator. Any flat-video ranking check must be separately labeled as a diagnostic, not silently substituted into the headline table. [CiCo evaluation implementation](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/main_task_retrieval.py)

### Hidden comparison variables

| Variable | Verified concern | Required handling |
|---|---|---|
| Text language | CiCo translates PHOENIX German and CSL-Daily Chinese into English using Google Translate. Public records contain translated text and original text. | Native-language mBART and translated-English CLIP results are separate language/pretraining regimes. |
| Video pretraining | CiCo's I3D features include external sign-domain training and target pseudo-label adaptation. | “No target gloss” is not “no sign supervision anywhere.” Preserve feature provenance. |
| Text initialization | CLIP, GrOVLE, German GPT-2, mBART and BERT are materially different. | A stronger text encoder is a resource/architecture change, not isolated evidence for a loss. |
| Additional modality | SEDS adds extracted pose and SignBERT initialization; scaling adds non-manual information and extensive external pretraining. | Include matched RGB-only and matched-initialization controls. Pose comes from RGB, but its learned prior can add resources. |
| FPS / clips | SEDS uses 24 FPS; PHOENIX source is 25 FPS; I3D commonly uses overlapping 16-frame clips and caps output sequence length at 64. | Store original timestamps, clip stride, and selected feature indices. Do not conflate frame count with clip count. |
| Text truncation | CiCo-family typically caps text at 32 tokens; SAN's released mBART branch uses a larger cap. | Report actual truncated-caption rate and span eligibility for local losses. |
| Dev split | The inspected PHOENIX dev pickle contains 7,615 records, including all 7,096 training keys. | Resolve the intended 519 dev keys before validation; do not treat the entire pickle as dev. |
| Evaluation masking | A malformed multi-sentence indexing assignment appears in CiCo, UPRet and SEDS releases. | Repair in every reproduced arm, document the patch, and distinguish repaired runs from published numbers. |
| Padding in alignment | Local softmax is calculated before excluding the opposite modality's padded positions. | Correct masks in all controlled runs and quantify the repair separately from the method. |
| Model selection | Inspected CiCo-family training loops evaluate the test loader during training and track best test R@1. | Use dev-only selection. Whether the publications actually selected reported checkpoints this way is **UNVERIFIED**; do not infer misconduct. |
| Label semantics | Identical text can receive different group IDs; other semantically equivalent candidates may be unannotated. | Exact-duplicate handling is a control, not novel semantic annotation; do not relabel paraphrases automatically. |

Sources for these implementation findings: [CiCo](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo), [UPRet](https://github.com/xua222/UPRet/tree/046366227417e1d8ec14145965403462df345984), [SEDS](https://github.com/longtaojiang/SEDS/tree/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246), [SAN](https://github.com/joonmy/SAN/tree/82aba9cbc1beb403abef6e9a3875ca52479805c8).

**Results that should not be directly merged:** SAN's 41-candidate stress test and full-pool PHOENIX; dictionary retrieval and sentence retrieval; BOBSL validation and test; group ranking and flat-video ranking; native-language and translated-language experiments without a language control; SL1.5M-pretrained systems and target-only pretraining as a causal method comparison; C²RL arXiv numbers and unverified final journal numbers; translation BLEU/ROUGE and retrieval recall.

## 4. Evolution of SLRet Methods

### 2022: SPOT-ALIGN establishes the task and the importance of sign-domain features

**Bottleneck:** generic action features and incorrectly timed text do not provide reliable sign-semantic representations. SPOT-ALIGN iteratively combines sign spotting and representation learning; it also compares a learned cross-modal embedding route with sign-recognition-derived textual matching. The combined system benefits from both. **Evidence:** large improvements from signing-aligned crops, iterative sign-domain adaptation, and combining complementary scores. [Duarte et al., CVPR 2022](https://arxiv.org/abs/2201.02495)

| Aspect | Decomposition |
|---|---|
| Inputs / spatial-temporal representation | RGB; I3D features from overlapping 16-frame clips. Mean pooling wins the reported video aggregation comparison against alternatives such as LSTM/NetVLAD. No pose stream. |
| Text / common space | Word features, NetVLAD and gated projection; GrOVLE selected for English after comparison. German experiments use language-appropriate representations. Common embedding dimension 512. |
| Pretraining | Kinetics and sign-domain resources; iterative pseudo spotting, including external sign/dictionary information. Exact resources differ between comparison rows. |
| Alignment / losses | Sentence-level cross-modal ranking, plus sign-recognition-based matching; no modern CLCL all-token interaction in the basic embedding route. Margin-based ranking loss. |
| Negatives / uncertainty | Conventional training negatives; no sign-aware negative distribution or explicit probabilistic retrieval. |
| Stages / inference | Obtain and improve sign features, train cross-modal embeddings, optionally combine recognition-based and embedding-based scores. |
| Optimization / cost | Appendix: RAdam, LR 0.001, batch 128, 40 epochs, margin 0.2, weight decay \(10^{-5}\); select on validation geometric mean of recalls. Dense feature extraction is the major offline cost. |
| Evidence / remaining limitations | Pretraining and crop quality account for substantial gains; broad semantic matching remains weak on visually subtle alternatives. Representation and retrieval-head contributions are not interchangeable. |
| Code | Official project is public; a complete author training/evaluation repository was not verified. **UNVERIFIED** for turnkey reproduction. |

The paper explicitly motivates the challenge of sign-specific visual understanding. The independent limitation is that recognition-derived vocabularies and global retrieval can miss fine distinctions or lexical content outside their learned coverage. [Official project](https://imatge-upc.github.io/sl_retrieval/)

### 2023: CiCo treats retrieval as cross-lingual sequence matching

**Bottleneck:** a global video/text similarity hides the distinct visual and written realizations of the same sentence, while sign-domain mismatch limits I3D. CiCo fuses domain-agnostic and domain-aware sign features, then computes cross-lingual contrastive learning (CLCL) over contextual clip and text tokens. This is already a fine-grained alignment method. [Cheng et al., CVPR 2023](https://arxiv.org/abs/2303.12793)

For normalized clip and text features, \(A_{lm}=x_l^\top y_m\). Schematically, CLCL uses
\[
s_{V\to T}=\frac1L\sum_l\sum_m
\operatorname{softmax}_{m}(A_{lm}/\tau_a)A_{lm},
\quad
s_{T\to V}=\frac1{M'}\sum_m\sum_l
\operatorname{softmax}_{l}(A_{lm}/\tau_a)A_{lm}.
\]
The two directions are used in retrieval training. Each normalization acts independently; this is neither a globally capacity-constrained assignment nor a monotonic alignment. Masks are omitted from this conceptual equation, but must be correctly applied in implementation.

| Aspect | Decomposition |
|---|---|
| Inputs / video features | Frozen, pre-extracted RGB I3D; domain-agnostic and pseudo-adapted domain-aware 1,024-D features are fused. Typical mixture coefficient 0.8, PHOENIX 0.9. |
| Temporal / spatial encoder | I3D supplies local space-time information; a CLIP-initialized visual Transformer contextualizes up to 64 clip features. Input projection 1,024→768; 12 Transformer layers; output 512. |
| Text | CLIP ViT-B/32-associated text Transformer; typical cap 32 subwords; English, including translated PHOENIX/CSL text. |
| Objective / fusion | Two directional CLCL contrastive losses; feature fusion precedes contextual alignment. No pose, translation decoder, or probabilistic head. |
| Pretraining / pseudo labels | External sign-domain I3D plus target adaptation through pseudo-label mining. Reported threshold 0.6 and NMS window 24; “gloss-free target data” does not erase this provenance. |
| Negatives / augmentation | In-batch negatives; text augmentation includes random word swaps. This is an experimental heuristic, not a linguistically valid assertion that word order never changes meaning. |
| Training / inference | Train temporal/text encoders on frozen features; inference computes token-pair similarities, not just one vector cosine. |
| Optimizer / cost | LR \(10^{-5}\), batch 512, 200 epochs; released code uses BertAdam implementation with warmup. Alignment temperature 0.07, gradient clipping and logit-scale constraints. All-pair local affinities can dominate memory. |

**Gain attribution:** on How2Sign, the reported progression is T2V/V2T R@1 31.5/26.4 for the initial baseline, 33.3/30.9 with its sign encoder configuration, 54.0/50.0 after CLCL, and 56.6/51.6 with augmentation. CLIP initialization itself produces a substantial gain in a separate ablation. Thus “local alignment works” is supported, but claiming the complete improvement comes independently of pretrained CLIP is not. Remaining failures include visual confusability, pseudo-label error, shared-support reuse, and weak supervision of sentence composition. [CiCo ablations and implementation](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo)

### 2024: UPRet uses distributions during training

**Bottleneck:** paired captions and sign videos exhibit ambiguous correspondence, while deterministic training can be brittle. UPRet predicts token-wise Gaussian parameters and samples representations, then uses an OT-derived auxiliary similarity during training. **AUTHOR CLAIM:** this better captures semantic uncertainty and fine-grained relationships. Its controlled recall gains support a useful training effect; they do not by themselves establish calibrated semantic uncertainty. [Wu et al., ECCV 2024](https://arxiv.org/abs/2405.19689)

| Aspect | Decomposition |
|---|---|
| Input / encoders | Same broad frozen I3D + trainable CLIP temporal/text family as CiCo. No additional pose. |
| Distribution head | Attention-based heads predict mean and log-scale for token embeddings. Released default uses two representations: the mean plus one random sample. |
| Actual transport granularity | Code pools sampled token sequences to sentence vectors first, then forms a \(2\times2\) sample transport problem per video–text pair. It does not solve an \(L\times M'\) word–clip OT problem. |
| Global/local alignment | Deterministic token interaction remains; learned token-importance MLPs weight aggregation. Sample-level OT contributes training similarity. |
| Loss / negatives | Symmetric retrieval contrast; random sampling and OT contribution; standard batch negatives, no SAN-style mining. |
| Inference / uncertainty | The stochastic transport branch is bypassed at evaluation; deterministic weighted local similarity is used. Inference uncertainty calibration is therefore not demonstrated by this implementation. |
| Training / cost | Reported 200 epochs, LR \(10^{-5}\), batch 512, four A100 GPUs. Transport uses \(\epsilon=0.1\) and up to 100 iterations over tiny sample matrices. Reported training iteration 0.63→0.70 s and inference 2.54→2.55 s are setup-specific, not portable latency predictions. |
| Public source | Complete-looking research scaffold exists but needs path, masking and evaluator repairs; checkpoint-to-paper numerical reproduction was not run. |

**Gain attribution:** the paper's own How2Sign CiCo reproduction is 56.4/50.3 R@1, compared with UPRet 59.1/53.4. Its T2V ablation progresses from 56.4 to 56.8 with distribution modeling plus sampling, 57.8 with OT configuration, 58.4 with sampling plus OT, and 59.1 for full treatment. These are stronger causal evidence than comparing only against the original CiCo table. **YOUR INFERENCE:** distributional regularization is useful, but explicit calibrated multi-answer retrieval and true clip–word transport remain unproven. [UPRet model and distribution code](https://github.com/xua222/UPRet/tree/046366227417e1d8ec14145965403462df345984/modules)

### 2024: SEDS adds trainable manual pose and cross-stream alignment

**Bottleneck:** RGB can retain background/signer information while losing detailed hand structure; frozen RGB features also limit task-specific visual adaptation. SEDS combines an offline RGB stream with an online pose stream and cross-stream fusion. [Jiang et al., ACM Multimedia 2024](https://doi.org/10.1145/3664647.3681237)

| Aspect | Decomposition |
|---|---|
| Input / preprocessing | RGB I3D plus RTMPose-extracted landmarks; 24 FPS; filter low-quality frames; up to 300 pose frames and 64 clip features. |
| Pose representation | RTMPose supplies 133 landmarks, but the selected pose representation uses 49: 21 per hand and 7 body joints. It is not a dense facial-expression stream. |
| Pose encoder | Shared hand GCN initialized from SignBERT; body GCN; concatenate three 512-D outputs to 1,536-D frame features; aggregate 16-frame windows; CLIP-initialized temporal Transformer to 512-D. |
| RGB / text | Frozen 1,024-D I3D features projected to a 768-D temporal Transformer; CLIP text encoder; 512-D retrieval features. |
| Fusion | Cross-modal gloss-aware fusion (CGAF), using local/deformable interactions and cross-stream semantic aggregation. The name “gloss-aware” does not mean manually annotated gloss is consumed. |
| Losses | RGB–text, pose–text and fused–text contrastive objectives plus pose–RGB fine-grained matching. No sign-aware hard-negative generator or probabilistic calibration. |
| Training / inference | Train pose, temporal/text and fusion components; RGB extraction remains frozen. Both streams and fusion are used at inference. |
| Optimizer / cost | Batch 128, 200 epochs, Adam with cosine warmup; pose/fusion LR around \(10^{-4}\), Transformer/text around \(10^{-5}\). Pose extraction is additional offline work and the pose network adds online work. Exact portable FLOPs/latency: **UNVERIFIED**. |

**Evidence:** How2Sign RGB-only 54.3/48.3 and pose-only 55.9/50.9 rise to 62.5/57.9 in the full system; a simpler MLP fusion gives 60.4/55.5. Removing fine-grained stream matching gives 60.8/55.8. Complementary modalities are demonstrated, but the entire gain cannot be assigned to CGAF independently of pose and SignBERT pretraining. Authors motivate nuisance reduction and end-to-end pose adaptation; independently, incomplete facial information, detection failure, and fine semantic negatives remain. [SEDS ablations](https://arxiv.org/abs/2407.16394), [official source](https://github.com/longtaojiang/SEDS/tree/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246)

**Constraint compliance:** none of the proposed methods initializes from or distills a pretrained SEDS checkpoint. SEDS code and reported numbers are audit/comparison material only. The source's required upstream SignBERT file is distinct from a trained SEDS checkpoint, but its exact independent checkpoint provenance was not resolved sufficiently to make it a hidden prerequisite of either proposal.

### 2025: C²RL learns content and context through paired-text pretraining

**Bottleneck:** sentence contrast alone undertrains local content and generative context; image/action pretraining is not the same as sign-specific paired-language learning. C²RL combines implicit content learning (ICL) with explicit context learning (ECL), then reuses the learned visual representation for translation and retrieval. [Chen et al., TCSVT 2025](https://doi.org/10.1109/TCSVT.2025.3553052)

| Aspect | Decomposition |
|---|---|
| Pretraining input / visual model | RGB; ImageNet ResNet-18 trained end-to-end; one frame in four sampled; temporal Conv1D, kernel 3; a lightweight 3-layer, 512-D Transformer configuration. |
| Text / context | Translation text supplies token/content alignment and autoregressive context supervision. No new manual annotation is needed, but paired-text generation is more supervision use than retrieval CE alone. |
| Objectives | ICL uses cross-lingual fine-grained contrast; ECL predicts the paired sentence with a decoder. Their combination is the central learning principle. |
| Retrieval stage | Extract and freeze pretrained visual features; project to 1,024; use separate mBART-large-cc25 encoder instances for visual and textual sequences; CLCL-style retrieval alignment. |
| Modalities / negatives / uncertainty | No required pose stream; no dedicated SAN-style negative construction or probabilistic inference. Temporal/context learning does not guarantee exact sign order recovery. |
| Optimization | Pretraining: 200 epochs, SGD LR 0.01, momentum 0.9, cosine schedule, label smoothing 0.2, 8×3090 with batch 8/GPU. Retrieval: 80 epochs, Adam LR \(10^{-4}\), batch 16/GPU on eight GPUs, dropout 0.3. |
| Public implementation | An independently maintained SLT replication includes partially shared author code. A complete official C²RL retrieval pipeline was **UNVERIFIED**. |

**Evidence:** PHOENIX ICL-only T2V/V2T R@1 74.7/73.5; ECL-only 74.1/74.5; together 78.7/77.6. This is good evidence for complementary pretraining objectives within that configuration. It is not evidence that changing only the CiCo loss yields the same gain: ResNet training, sampling, pretraining objectives and mBART differ. **Remaining limitations:** exact full-pool grouping and final journal table equivalence are unresolved; explicit visual hard negatives and local evidence faithfulness are not isolated. [C²RL arXiv version, architecture and tables](https://arxiv.org/abs/2408.09949), [independent replication](https://github.com/ozgemercanoglu/sltbaselines)

### 2026: SAN identifies the difference between semantic and visual hardness

**Bottleneck:** a language-model hard negative may be semantically close yet visually easy; a sign-space confusable alternative may be missed by semantic mining. SAN uses high-confidence sign–word mappings, finds visually similar representations attached to different words, and substitutes those words into training captions. [Lee et al., ACL 2026](https://aclanthology.org/2026.acl-long.1302/)

| Aspect | Decomposition |
|---|---|
| Encoders | Experiments on GFSLT-VLP-based and CiCo-based retrieval; no newly required pose encoder or spatial architecture. |
| Teacher / mining | A trained GFSLT-VLP-based retriever supplies alignment and visual similarity; \(\alpha=0.7\), \(\beta=0.7\). Different word identity is used as a candidate filter. |
| Negatives / losses | Five hard negative captions per example; training typically substitutes two words; retrieval loss plus negative discrimination weighted 0.4. Test stress captions use one-word changes. |
| Training / inference | 100 epochs, SGD LR 0.01, cosine; paper batch 32 for GFSLT-VLP and 256 for CiCo. No negative generator required at ordinary inference. |
| Public implementation | Released mBART-oriented branch uses I3D features 1,024→mBART width 1,024 and text length up to 70. Precomputed negative table is expected; its complete generation pipeline is absent. A faithful released CiCo branch is **UNVERIFIED**. |
| Cost | Encoding \(K\) extra captions increases training work. Code forms more cross-pair similarities than needed before selecting within-example negatives. Direct \([B,K,L,M']\) scoring is sufficient. |

**Evidence:** its CiCo stress-test V2T R@1 rises 17.9→39.4, but ordinary T2V/V2T R@1 falls 69.2/70.1→68.1/67.8. GFSLT-VLP stress R@1 rises 16.8→49.1; ordinary T2V rises but V2T declines. The paper's hardness/weight ablations also reveal a tradeoff. **AUTHOR CLAIM:** negative-distribution mismatch is central. **YOUR INFERENCE:** the results support supervision relevance, not the stronger statement that model capacity is generally sufficient. A model with better manual/non-manual visual evidence could still help.

SAN improves the supervision distribution. It does not guarantee that a negative word is semantically false, that the inferred sign span is correct, or that the comparison uses the same visual evidence for the original and altered caption. The paper acknowledges limits of static/dictionary-level relationships for continuous dynamics; domain breadth is limited by the PHOENIX-only experiment. [SAN source](https://github.com/joonmy/SAN/tree/82aba9cbc1beb403abef6e9a3875ca52479805c8)

### Additional work that changes the interpretation of the field

| Work | Relevant principle, evidence, and remaining limitation |
|---|---|
| [Scaling up Multimodal Pre-training, TPAMI 2025](https://doi.org/10.1109/TPAMI.2025.3599313) | Manual/non-manual pose pretraining on SL1.5M. Its arXiv tables show CSL manual-only R@1 82.4, non-manual-only 31.4, combined 87.5; no external pretraining 75.2 versus full external pretraining 87.5. Strong evidence for complementary cues and a strong scale confound. Final journal table equivalence **UNVERIFIED**. |
| [CSLR², 2024 preprint](https://arxiv.org/abs/2405.10266) | BOBSL sentence and sign learning, pseudo sign labels, hard-negative NCE, contextual Video-Swin features and T5 text. Already handles some duplicate-label negatives in code. Qualitative results show visual confusion, multiple plausible matches, and one-sign/multiword expressions. Separate, much larger pool. |
| [SignCLIP, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.518/) | Pose-language contrast across 44 sign languages and dictionary entries. Relevant normalization/cross-lingual prior; no evidence that dictionary recall directly transfers to sentence SLRet. |
| [SignRep, ICCV 2025](https://arxiv.org/abs/2503.08529) | Sign-oriented masked visual representation learning, with hand/body/non-manual priors and public feature extraction. Relevant stronger visual representation, but adding its pretraining changes the resource comparison. |
| [SHuBERT, ACL 2025](https://arxiv.org/abs/2411.16765) | Multi-stream hand/face/body cluster prediction on substantial external ASL video. Relevant representation pretraining, not a verified sentence-retrieval leader. |
| [GeoSign, NeurIPS 2025](https://github.com/ed-fish/geo-sign) | Hyperbolic contrastive sign translation. Geometry is relevant; translation gains are not retrieval SOTA. |
| [SCL-SLT, ACL 2026](https://aclanthology.org/2026.acl-long.2116/) | Similarity trajectories and curriculum selection of negatives. Makes generic “adaptive negative filtering” an unsafe novelty claim; main evaluation is translation. |
| [SEA, ACL 2026](https://aclanthology.org/2026.acl-long.1401/) | Linguistic segmentation, SignCLIP and sequence alignment for subtitle timing. Sentence alignment does not justify imposing word/sign monotonicity. |
| [VTaMo, July 2026 preprint](https://arxiv.org/abs/2607.09126) | Non-monotonic frame/token Sinkhorn alignment, null token, global transform, training-time reordering and token contrast. Uses filtered pseudo-gloss-like text and recovery; evaluates SLT. Generic null-aware OT is already present in sign research. |
| [DualAnchor, July 2026 preprint](https://arxiv.org/abs/2607.27614) | Fixed-real-mass partial OT with two dustbins plus language-prior anchoring. Strong direct overlap with an unqualified “partial OT + teacher” proposal. Teacher regularizes decoder language distributions, not sampling-consistent correspondence. |
| [SignMatch, September 1, 2026 preprint](https://arxiv.org/abs/2609.01886) | Sign prototypes bridge continuous and dictionary domains; includes substantial sign-level/pseudo-span resources. Relevant visual matching; not free-form sentence T2V/V2T. |
| [GTRN, Neurocomputing 2025](https://ro.ecu.edu.au/ecuworks2022-2026/6028/) | Hierarchical reference graph for visual signing queries retrieving video documents. Its task is visual-query retrieval, not the text↔sentence-video table here. |
| [CMCM, CVIU 2026](https://doi.org/10.1016/j.cviu.2025.104631) | Relevant causality-inspired multi-grained cross-modal retrieval. Primary numeric tables and complete training details **UNVERIFIED**. Public code fragments cannot resolve published performance or full causal claims. |

### Source-code audit and reusable tensor interfaces

| Repository / audited revision | Exact inspected locations | Inputs → outputs and dependencies | Reuse judgment |
|---|---|---|---|
| [CiCo / SLRT, 38a4f7b](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo) | CLCL modules/modeling.py, module_clip.py, dataloaders, main_task_retrieval.py, metrics.py; I3D extraction/preprocessing | Loader features \([B,1024,64,1]\) → temporal tokens \([B,64,512]\); text \([B,32]\) → \([B,32,512]\); affinities \([B_v,B_t,64,32]\) → two score matrices. Frozen feature files; gradients through projection, temporal/text Transformers. | Good scaffold after documented fixes. Pretrained feature provenance must be retained. |
| [UPRet, 0463662](https://github.com/xua222/UPRet/tree/046366227417e1d8ec14145965403462df345984) | modules/modeling.py::flip_similarity_softmax, PDE.py; main_task_retrieval.py; train scripts; six dataset loaders; metrics.py | Same deterministic interface; PDE takes \([B,L,512]\) or \([B,M',512]\), returns mean/log-scale sequences; samples pool to \([S,B,512]\); OT costs \([B_vB_t,S,S]\), \(S=2\). | Primary scaffold. Keep original distribution branch as a baseline component; do not mislabel it token OT. |
| [SEDS, 434e3f7](https://github.com/longtaojiang/SEDS/tree/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246) | modules/modeling.py, modeling_signbert.py, pose/fusion definitions, dataloaders, train scripts, metrics | Pose \([B,F,49,c]\) → three 512-D part features → 1,536-D frames → \([B,64,512]\); RGB \([B,64,512]\); fused sequence same width. Upstream SignBERT hand-GCN state expected. | Shapes and fusion inspectable; trained SEDS weights excluded. Not the proposed initialization path. |
| [SAN, 82aba9c](https://github.com/joonmy/SAN/tree/82aba9cbc1beb403abef6e9a3875ca52479805c8) | models.py, datasets.py::generate_hard_negatives, train.bash and training configuration | Features \([B,L,1024]\); captions and \(K\) negatives become \([B(1+K),70,1024]\); precomputed word-candidate table required. SGD/cosine script uses \(\lambda=.4,K=5\). | Reuse negative-caption plumbing and loss control. Missing miner must be rebuilt transparently from training data; random fallback is not a reliable semantic negative. |
| [C²RL-related replication, f741330](https://github.com/ozgemercanoglu/sltbaselines/tree/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13) | models/models.py::cross_lingual_similarity_v2, train_vlp.py, train_slt.py, configs | ICL/ECL pretraining scaffolding and an author-shared CLCL function. Padding similarity is scaled, not truly removed from softmax. | Useful independent pretraining reference; not an official complete retrieval implementation or verified replication of journal recall. |
| [CSLR², 5f3bd69](https://github.com/gulvarol/cslr2/tree/5f3bd697e1d2e184a5882cbbd0da64ce9455fd28) | model definitions, loss/hn_nce.py, training loops and dataset configs | Frozen Video-Swin-Tiny features \([B,F,768]\) → 6-layer temporal Transformer → max pool/projection 256; frozen T5-large 1,024→256. Adam \(5\cdot10^{-5}\), batch 128, 20 epochs, temperature .07; duplicate-label masking. | Real implementation of joint sentence/sign learning; supervision and BOBSL pool differ. |
| [SignCLIP, a819944](https://github.com/J22Melody/fairseq/tree/a819944093876443119f294a2c50c663cecd65b0/examples/MMPT) | MMPT models/losses and baseline_final.yaml | Current config: pose 609-D, max 256 frames; 12-layer video width 768; BERT-base-cased text, max 64; masked pooling. Batch 448 on 80 GB setup, Adam \(5\cdot10^{-5}\), polynomial schedule/warmup. | Useful pose/global contrast code. This current config is not proof of the exact multilingual paper's final configuration. |
| [SignRep, 06f40b5](https://github.com/ryanwongsa/SignRep/tree/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2) | example_usage.py; models/final_models/FINAL_hiera_latent_model_head_v25_active.py | Public Hiera-based feature extraction/model definition. End-to-end reproduction of all pretraining and SLRet integration **UNVERIFIED**. | Optional separately labeled feature experiment only, not a free replacement for baseline I3D. |
| [CMCM, 5d45871, master](https://github.com/vddong-zjut/CMCM/tree/5d458719d1da2f082e188cc44705003d919e7e97) | Encoder.py, CSA_Module.py, CCG_Module.py and augmentation files | Encoder named i3d_encoder instantiates r2plus1d_18; pretrained_i3d_path is unused; classifier/shape path is not a complete feature extractor; undefined DEVICE and inconsistent causal-mask slicing occur. No complete train/eval driver. | Public partial code exists on master; do not incorrectly call the repository empty. Not currently a reliable baseline. |

Additional source/implementation distinctions:

1. **UPRet's training-only OT is sample-level after pooling.** The source's \(2\times2\) transport is the decisive shape fact.
2. **SEDS' pose input omits dense face landmarks.** “133-point detector” should not be reported as “133-point encoded representation.”
3. **SAN's README says work is forthcoming, but code fragments exist.** Conversely, the presence of code does not supply the missing negative-table constructor or both paper backbones.
4. **CiCo-family multi-caption indexing contains an apparent bug:** the third assigned tensor indexes segment IDs by the input mask, rather than filtering the input mask by caption-group indices. Static inspection warrants a repair, not an assertion that the published experiments ran that exact erroneous line.
5. **Padding creates a real mathematical sensitivity:** for affinities \((0.1,0)\) at temperature .07, the weighted score is approximately .0807; adding four zero-valued padded partners before softmax changes it to .0455. This small analytic example demonstrates the mechanism, not a measured retrieval gain.
6. **Tie handling needs care:** the generic exact-equality rank routine can count several tied positions per query. Main metric changes require explicit disclosure; duplicated tie records should not covertly become a proposed method gain.

For memory, an all-pair affinity tensor at \(B=512,L=64,M'=32\) has 536,870,912 elements: **2 GiB in float32 for one tensor**, before softmax intermediates, augmented text, gradients, and encoder activations. A \([B,K,L,M']\) local-negative tensor with \(K=5\) has only 5,242,880 elements, about **20 MiB** in float32. These are storage calculations, not measured peak GPU requirements.

## 5. Current SOTA as of 2026-09-12

### Interpretation and comparability

The tables provide the **strongest verified published-result frontier found**, not a falsely exhaustive certification. A relevant 2026 paper, CMCM, remains numerically **UNVERIFIED**. C²RL and Scaling entries use verified arXiv tables associated with later journal publications; equality to the final journal tables is **UNVERIFIED**.

“Directly comparable” below means comparable to the CiCo/UPRet fixed-feature, language and candidate regime for a method-level comparison. A system with different representations can still be a meaningful *system-level* performance target, while being **Not directly comparable** as an isolated loss ablation. “Uncertain” means missing protocol/version information prevents a reliable decision.

| Resource code | Video/text backbones | Modalities, supervision, pretraining | Code and comparison status |
|---|---|---|---|
| A: CiCo | I3D + CLIP-initialized temporal model; CLIP text | RGB; external sign pretraining, target pseudo adaptation; English/translated English; no target gloss labels in retrieval | Public. **Directly comparable** within A/B when identical released features, groups and selection are used. |
| B: UPRet | Same broad I3D/CLIP family plus distribution heads | Same modalities/resources; probabilistic training | Public. **Directly comparable** to its own CiCo reproduction; original-paper CiCo comparison has run/config uncertainty. |
| C: SPOT-ALIGN | Iteratively adapted I3D; GrOVLE/NetVLAD for English, language-specific German text configuration | Recognition-derived matching and external sign resources | Complete public training code **UNVERIFIED**. **Not directly comparable** to A/B for component attribution. |
| D: SEDS | I3D + pose GCN + CLIP temporal/text | RGB/pose; SignBERT initialization and detector; same target translation pairing | Public. **Not directly comparable** to A/B in resources/modalities; exact training filtering also differs. |
| E: C²RL | ResNet-18/temporal visual pretraining; mBART retrieval encoders | RGB; ImageNet initialization, target paired-text ICL/ECL, mBART language pretraining | Complete official retrieval code **UNVERIFIED**. **Uncertain** protocol/version; **not resource-matched** to A/B. |
| F: Scaling | Manual/non-manual pose encoder; frozen mBART text in pretraining, another text encoder in retrieval whose exact identity is not specified sufficiently | Large external SL1.5M pretraining; richer pose cues | Exact released retrieval reproduction **UNVERIFIED**. **Not directly comparable** to A/B or target-only training. |
| G: SAN | CiCo/GFSLT-VLP variants | Train-only visual negative mining, GFSLT-VLP teacher | Partial source. **Directly comparable** within each SAN-paper baseline pair; **Uncertain** relative to original CiCo language/backbone implementation. |

For Scaling, frozen mBART in pretraining is verified; the exact retrieval-specific text-backbone/checkpoint identity remains **UNVERIFIED**. The paper specifies 79 input landmarks (42 hand, 8 mouth, 18 facial, 11 upper body), separate eight-block manual/non-manual encoders with widths 1,024/1,536, and a 512-D shared space. Its retrieval configuration is AdamW, LR \(10^{-4}\), batch 32, 60 epochs, cosine schedule, backbone LR scale .1 and temperature .07. This is a materially different resource/encoder regime.

### How2Sign — established sentence retrieval

All values are percentages; recall triples are R@1 / R@5 / R@10.

| Paper / venue | Resource | T2V | V2T | MedR T/V | Comparison |
|---|---|---|---|---|---|
| [SPOT-ALIGN COMB, CVPR 2022](https://arxiv.org/abs/2201.02495) | C | 34.2 / 48.0 / 52.6 | 23.6 / 47.0 / 53.0 | 8 / 7.5 | Not directly comparable |
| [CiCo, CVPR 2023](https://arxiv.org/abs/2303.12793) | A | 56.6 / 69.9 / 74.7 | 51.6 / 64.8 / 70.1 | 1 / 1 | Directly comparable within A/B, subject to matched release |
| [CiCo reproduced by UPRet, ECCV 2024](https://arxiv.org/abs/2405.19689) | B control | 56.4 / 69.4 / 74.1 | 50.3 / 63.6 / 69.3 | 1 / 1 | Directly comparable to next row |
| [UPRet, ECCV 2024](https://arxiv.org/abs/2405.19689) | B | 59.1 / 71.5 / 75.7 | 53.4 / 65.4 / 70.0 | 1 / 1 | Directly comparable to own reproduced control |
| [SEDS, ACM MM 2024](https://arxiv.org/abs/2407.16394) | D | **62.5** / 75.1 / **80.1** | **57.9 / 70.4 / 74.9** | 1 / 1 | Not directly comparable in modalities/pretraining |
| [C²RL, TCSVT 2025; arXiv table](https://arxiv.org/abs/2408.09949) | E | 62.4 / **75.9 / 80.1** | 57.5 / 68.4 / 73.0 | Not reported here | Uncertain final table/protocol; different pretraining |
| [CMCM, CVIU 2026](https://doi.org/10.1016/j.cviu.2025.104631) | Unknown | **UNVERIFIED** | **UNVERIFIED** | UNVERIFIED | Uncertain; no inferred numerical claim |

UPRet additionally reports mean ranks 54.4 T2V and 76.4 V2T. Do not mistake its deterministic inference for a calibrated probabilistic ranking output. The different best R@1 and R@5 rows show why a single “SOTA model” label loses information.

### PHOENIX-2014T — standard full pool

| Paper / venue | Resource | T2V R@1/5/10 | V2T R@1/5/10 | MedR T/V | Comparison |
|---|---|---|---|---|---|
| [SPOT-ALIGN COMB, CVPR 2022](https://arxiv.org/abs/2201.02495) | C | 55.8 / 79.6 / **87.2** | 53.1 / 79.4 / 86.1 | 1 / 1 | Not directly comparable |
| [CiCo, CVPR 2023](https://arxiv.org/abs/2303.12793) | A | 69.5 / 86.6 / 92.1 | 70.2 / 88.0 / 92.8 | 1 / 1 | Directly comparable within matched A/B |
| [CiCo reproduced by UPRet](https://arxiv.org/abs/2405.19689) | B control | 70.4 / 88.2 / 92.7 | 70.9 / 87.2 / 92.5 | 1 / 1 | Directly comparable to next row |
| [UPRet, ECCV 2024](https://arxiv.org/abs/2405.19689) | B | 72.0 / 89.1 / 94.1 | 72.0 / 89.4 / 93.3 | 1 / 1 | Directly comparable to own control |
| [SEDS, ACM MM 2024](https://arxiv.org/abs/2407.16394) | D | 76.8 / 91.7 / 95.3 | **78.7 / 92.5** / 95.2 | 1 / 1 | Not directly comparable in resources |
| [C²RL, TCSVT 2025; arXiv table](https://arxiv.org/abs/2408.09949) | E | **78.7** / 92.2 / 94.9 | 77.6 / 91.3 / 94.2 | Not reported here | Uncertain final version/protocol |
| [Scaling, TPAMI 2025; arXiv table](https://arxiv.org/abs/2408.08544) | F | 74.5 / **93.3 / 95.6** | 75.1 / 92.1 / **95.3** | 1 / 1 | Not directly comparable; large external pretraining |
| [CMCM, CVIU 2026](https://doi.org/10.1016/j.cviu.2025.104631) | Unknown | **UNVERIFIED** | **UNVERIFIED** | UNVERIFIED | Uncertain |

The SPOT-ALIGN T2V R@10 above is 87.2 in its original result, not the inconsistent value reprinted in a later comparison. UPRet reports mean ranks 4.4/4.6.

**SAN must also be inspected, but within its own experiments:**

| SAN paper arm, ACL 2026 | Ordinary T2V R@1/5/10 | Ordinary V2T R@1/5/10 | Stress V2T R@1/5/10 | Stress MRR ×100 |
|---|---|---|---|---:|
| CiCo control | 69.2 / 87.2 / 92.2 | 70.1 / 87.7 / 92.9 | 17.9 / 55.3 / 79.1 | 35.0 |
| CiCo + SAN | 68.1 / 87.4 / 91.7 | 67.8 / 87.4 / 91.7 | 39.4 / 75.4 / 92.5 | 54.4 |
| GFSLT-VLP control | 67.9 / 88.4 / 93.8 | 69.4 / 88.7 / 93.3 | 16.8 / 53.1 / 78.0 | 33.9 |
| GFSLT-VLP + SAN | 70.2 / 89.3 / 94.4 | 67.4 / 85.4 / 90.5 | 49.1 / 85.9 / 94.9 | 64.1 |

These paired arms are **Directly comparable within the SAN paper**. The stress columns are **Not directly comparable** to ordinary-pool columns. Original-CiCo-to-SAN-CiCo equivalence is **Uncertain**. [SAN, Table 1](https://aclanthology.org/2026.acl-long.1302/)

### CSL-Daily — established group-based sentence retrieval

| Paper / venue | Resource | T2V R@1/5/10 | V2T R@1/5/10 | MedR T/V | Comparison |
|---|---|---|---|---|---|
| [CiCo, CVPR 2023](https://arxiv.org/abs/2303.12793) | A | 75.3 / 88.2 / 91.9 | 74.7 / 89.4 / 92.2 | 1 / 1 | Directly comparable within matched A/B |
| [CiCo reproduced by UPRet](https://arxiv.org/abs/2405.19689) | B control | 76.3 / 88.6 / 92.1 | 73.9 / 87.9 / 92.0 | 1 / 1 | Directly comparable to next row |
| [UPRet, ECCV 2024](https://arxiv.org/abs/2405.19689) | B | 78.4 / 89.1 / 92.0 | 77.0 / 89.2 / 92.7 | 1 / 1 | Directly comparable to own control |
| [SEDS, ACM MM 2024](https://arxiv.org/abs/2407.16394) | D | 85.8 / 94.4 / 95.6 | 85.4 / 93.8 / 95.8 | 1 / 1 | Not directly comparable in resources |
| [C²RL, TCSVT 2025; arXiv table](https://arxiv.org/abs/2408.09949) | E | **90.3 / 96.4 / 97.7** | **88.4 / 95.7 / 97.1** | Not reported here | Uncertain final version/protocol |
| [Scaling, TPAMI 2025; arXiv table](https://arxiv.org/abs/2408.08544) | F | 87.5 / 95.2 / 97.6 | 87.2 / 95.0 / **97.2** | 1 / 1 | Not directly comparable; SL1.5M external pretraining |
| [CMCM, CVIU 2026](https://doi.org/10.1016/j.cviu.2025.104631) | Unknown | **UNVERIFIED** | **UNVERIFIED** | UNVERIFIED | Uncertain |

UPRet reports mean ranks 6.7/5.5. A proposed loss beating UPRet here by a small amount would not justify claiming to have overtaken C²RL's reported system.

### OpenASL and BOBSL — separate protocols

| Dataset / paper | T2V R@1/5/10 | V2T R@1/5/10 | Other metric / resources | Comparison |
|---|---|---|---|---|
| OpenASL: [C²RL, arXiv table associated with TCSVT 2025](https://arxiv.org/abs/2408.09949) | **62.2 / 81.7 / 86.8** | **61.6 / 79.8 / 84.6** | RGB ResNet/mBART, paired-text pretraining; official complete retrieval code unverified | **Uncertain** exact grouping/version. Paper prose says 62.6 T2V R@1, table says 62.2; table value is retained. |
| BOBSL SENT-TEST 20 K: [CSLR², 2024 preprint](https://arxiv.org/abs/2405.10266) | 29.4 / 45.2 / 51.5 | 28.1 / 44.9 / 51.0 | MedR 9 both directions; Video-Swin/T5; pseudo sign labels; public code | **Not directly comparable** to other datasets; **Uncertain** compatibility with repository's 25 K edition |
| BOBSL SENT-VAL 2 K: same paper | 51.7 / 69.9 / 75.4 | 50.2 / 69.1 / 74.7 | Validation only | **Not directly comparable** to 20 K test |

No verified newer full-pool OpenASL result was established in the inspected sources. This is an evidence boundary, not proof that none exists. SignCLIP, GTRN and SignMatch results belong to different task/pool definitions and cannot fill these cells.

## 6. Failure-Mode Taxonomy

“Demonstrated” applies to the stated observation, not automatically to the proposed causal explanation.

| Failure mode | Evidence and status | What current systems leave unresolved |
|---|---|---|
| Visually confusable lexical alternatives | **Demonstrated:** SAN's examples and large stress-test gains; CSLR² qualitative retrieval confusions. | SAN improves which negatives are shown. It does not add missing image detail or prove correct grounding of each substituted word. [SAN](https://aclanthology.org/2026.acl-long.1302/), [CSLR²](https://arxiv.org/abs/2405.10266) |
| Fine-grained versus ordinary retrieval conflict | **Demonstrated:** SAN CiCo stress R@1 improves by 21.5 points while ordinary T2V/V2T drop 1.1/2.3 points. **Hypothesis:** some gradient pressure changes common sentence semantics or seeks evidence in the wrong region. | Adding the same full-caption negative loss to SEDS or C²RL need not remove the tradeoff. Local negative losses already exist in FSC-CLIP, making their adaptation a necessary control. |
| False negatives / semantic ambiguity | **Demonstrated structural conflict:** repeated exact text across group IDs in public annotations; SCL-SLT illustrates semantically identical captions. **Unverified:** frequency of unannotated non-identical paraphrases. | Diagonal CE treats many off-diagonal pairs as negative; a different word is not a proof of different meaning. UPRet does not supply semantic relevance labels. [SCL-SLT](https://aclanthology.org/2026.acl-long.2116/) |
| Temporal composition | **Demonstrated architecture fact:** local similarities are aggregated, and CiCo-family augmentation includes word swapping. **Plausible but unverified failure:** similar local signs in different composition produce wrong rankings. | Temporal Transformers encode order, so saying “these models ignore time” is false. C²RL's decoder improves context but does not isolate the relevant retrieval errors. |
| Unreliable frame/token alignment | **Demonstrated structural fact:** independent row/column softmax permits many tokens to favor the same region; text-conditioned negative alignment can move to another region. **Unverified:** causal share of real retrieval errors. | CiCo, SEDS, C²RL and UPRet local scores do not impose a common evidence comparison across altered captions. UPRet's transport is over pooled samples. |
| Sampling and granularity mismatch | **Demonstrated:** 16-frame windows overlap; long sequences are capped/subsampled; BPE length differs from lexical length. **Hypothesis:** counting observations rather than evidence biases ranking. | More clips can repeat the same observation; repeated signing at a different time is a different phenomenon. VTaMo/DualAnchor add OT but do not establish sampling-consistent correspondence. |
| Signer/domain variance | **Demonstrated:** CiCo's domain-aware feature ablations and SEDS's nuisance motivation/qualitative cases. **Unverified:** exact nuisance/linguistic decomposition in embeddings. | Clothing/background should generally not determine content; location, handedness conventions, body shifts and role reference cannot all be discarded indiscriminately. |
| Manual detail loss | **Demonstrated:** SEDS's RGB/pose/fusion ablations show complementary information. **Unverified:** an isolated effect size for hand orientation or hand shape on full-pool retrieval. | Frozen RGB may have already discarded a distinction; an improved loss cannot reconstruct it. Pose estimation also loses occluded/depth information. |
| Non-manual information loss | **Demonstrated:** scaling paper manual+non-manual ablation; SEDS selects only 49 manual/body joints. | Generic “add face” is already prior art. The open issue is reliable contribution at matched data/pretraining and under real pose noise. [Scaling](https://arxiv.org/abs/2408.08544) |
| Pose reliability / missing cues | **Demonstrated code behavior:** SEDS filters low-confidence frames. **Hypothesis:** dropping whole frames discards usable body/face/other-hand evidence. | No verified retrieval ablation isolates confidence-aware partial retention against a matched pose model. |
| Retrieval uncertainty | **Demonstrated:** UPRet's training/inference asymmetry. **Unverified:** whether its variance separates linguistic ambiguity, pose uncertainty, domain shift and ordinary model error. | Neither a Gaussian parameter nor a soft alignment is automatically calibrated uncertainty; there is no verified many-answer semantic oracle in these protocols. |
| Preprocessing and evaluator sensitivity | **Demonstrated:** crop-alignment effect, group versus video candidate construction, duplicate labels, mask defects. | Improvements can be artifacts unless all arms share a repaired, frozen protocol. These are experimental prerequisites, not sufficient method contributions. |

Two theoretical cautions matter. First, a global or local point-embedding model can rank several candidates closely; distributional embeddings are an inductive bias, not a logical prerequisite for ambiguity. Second, a non-monotonic transport plan can be correct about lexical support while wrong about grammatical scope. Neither proposed method should claim to solve all sign-language structure.

## 7. Research Gap Matrix

Scores are **YOUR INFERENCE**, on 1–5 scales: E evidence, N novelty after adjacent-field checking, G expected gain, F feasibility, R reproducibility, D reviewer defensibility. They are research-prioritization judgments, not measured outcomes. Ranking prioritizes evidence, causal tractability and feasibility; a high-scoring hygiene fix is still insufficient as a standalone paper.

| Rank / gap | Precise gap and why current work does not close it | Evidence / existing-data test | Novelty risk | Difficulty / upside | E/N/G/F/R/D |
|---|---|---|---|---|---|
| 1. Shared evidence for visual hard negatives | Original and altered captions can be compared using different visual support; discriminating their common context can disturb ordinary retrieval. SAN changes negatives; FSC-CLIP localizes matching but recomputes support per candidate. CiCo/SEDS/C²RL/UPRet do not isolate this intervention. | SAN tradeoff is demonstrated; support-hopping cause remains a hypothesis. Compare locked versus independently optimized support with the same negatives and encoders on PHOENIX/How2Sign. | Medium–high: FSC-CLIP, TACo and local metric learning are close. “Local HN” alone is rejected. | Low–medium / Medium–high | 5/3/4/5/5/4 |
| 2. Repeated-positive training conflict | Group IDs and caption strings do not define the same relevance relation. | Direct annotation counts plus diagonal CE. Test exact duplicate exclusion or existing group-positive numerator in every control. | Very high: already handled in CSLR², ALBEF-style group targets and related contrastive work. | Low / Medium | 5/1/3/5/5/5 |
| 3. Finite, sampling-consistent alignment evidence | Overlapping sampled clips can receive separate mass despite representing the same observation. Independent local matches have no physical-time capacity. | Code structure demonstrated; actual error contribution unverified. Duplicate fixed features with timestamps, vary ordered sampling, measure ranking drift and full-pool performance. | Medium–high: OT matching, weighted marginals, VTaMo and DualAnchor. New claim must concern evidence accounting and correspondence consistency. | Medium / Medium–high | 4/3/4/4/5/4 |
| 4. Reliable non-manual information at fixed resources | Does non-manual information still help when video pretraining, detector, FPS and training budget are held fixed? | Scaling's complementary-stream ablation supports importance but includes external data. Existing face keypoints or extraction suffice. | High: multi-stream pose and sign pretraining already exist. | Medium–high / High | 4/2/4/3/4/4 |
| 5. Brief-cue survival under fixed clip budgets | Uniform 64-clip sampling can suppress brief discriminative events; merely increasing frames changes cost. | Caps/overlap are verified; isolated brief-cue errors remain unverified. Compare fixed-budget pooling on existing videos, stratify by duration. | High: temporal pyramids, adaptive sampling, SEA and SignMatch. | Medium / Medium–high | 4/2/4/3/4/4 |
| 6. Unmatched evidence with meaningful retention | Null matching exists, but fixed uniform token masses or unconstrained rejection can discard crucial content or misrepresent how much is matched. | VTaMo code fixes column masses; DualAnchor controls total real mass. Test informative-span suppression and candidate ranking under identical features. | High: generic partial OT is solved; only retention/calibration questions remain. | Medium / Medium | 4/2/3/4/4/3 |
| 7. Order-sensitive supervision beyond contextual encoding | Does the loss preserve distinctions involving sequence composition rather than only local membership? | Transformer/aggregation/augmentation facts; causal linguistic errors unverified. Order-preserving augmentation control and existing-data perturbation diagnostics. | High: temporal grounding, sequence alignment, action-order work. | Medium / Medium–high | 3/3/4/4/4/3 |
| 8. Reliability of sign-aware pseudo negatives | Word identity and visual similarity do not certify semantic falsity or correct localization. | SAN definition and missing automatic semantic oracle; repeated-caption evidence. Examine mined-support stability, duplicate rate, learning dynamics and real-pool gains. | High: uncertainty weighting, SCL-SLT, PCME++, self-training. | Medium / Medium | 4/2/3/4/4/3 |
| 9. Partial pose evidence retention | A bad hand estimate need not invalidate body or other-hand evidence in that frame. | SEDS filtering behavior demonstrated; retrieval effect unverified. Inject controlled missing landmarks into established inputs and train confidence-aware masking. | Medium–high: pose action recognition and masked multimodal learning. | Medium / Medium | 3/3/3/4/4/3 |
| 10. Selective signer invariance | Separate identity/background from body-relative location and discourse shifts without deleting linguistic signal. | Domain adaptation gains; weak direct representation-level evidence. Existing signer metadata and controlled background/body-relative perturbations. | High: domain adversarial learning, re-identification and SignCLIP normalization. | Medium–high / Medium | 4/2/3/3/4/3 |
| 11. Identifiable uncertainty at retrieval time | Distinguish epistemic error, missing visual evidence and multiple plausible meanings under limited relevance labels. | UPRet train-only stochastic branch; no decisive calibration study. Evaluate error prediction under existing relevance and synthetic sensor noise, clearly separating them. | High: PCME++, ProLIP, probabilistic retrieval. | High / Uncertain | 3/2/3/3/3/2 |
| 12. Method-versus-pretraining attribution | Published improvements mix architecture, external data, text language and training objectives. | Strongly demonstrated by tables and source. Match features and pretraining, then transfer promising losses to the stronger representation regime. | Very high; an experimental prerequisite, not a standalone method. | Medium / Indirect | 5/1/2/3/3/4 |

The final methods are selected from gaps 1 and 3. Gap 2 is mandatory hygiene in their controls; it is not packaged as a second novel component. Gaps 4 and 5 could have higher ultimate ceilings, but currently require more representation engineering and are harder to distinguish from stronger pretraining.

## 8. Research Questions

| Question | Hypothesis | Independent variable | Dependent variables / predicted observation | Falsification |
|---|---|---|---|---|
| RQ1: Does fine-grained retrieval remain bottlenecked because alternative captions are contrasted on different visual support? | Holding support fixed isolates the relevant visual distinction and reduces the fine/full-pool tradeoff. | Same mined negatives and loss family; independent versus shared positive support; changed-span versus all-token scoring | Both-direction full-pool R@1/5/10; existing SAN stress recall where protocol compatible; gradient cosine and support movement. Shared support improves the Pareto frontier. | Tuned FSC-CLIP+SAN or a simple span-only loss matches it; shared support adds no benefit or only helps synthetic candidates. |
| RQ2: Does clip multiplicity bias local retrieval independently of new visual information? | Overlapping/duplicated observations spuriously increase support for a caption. | Duplicate fixed embeddings with preserved timestamps; uniform versus physical-coverage masses | Score/rank drift under duplication, concentration of support, ordinary recall. Evidence-aware scoring is stable and improves real errors. | Drift is negligible, uncorrelated with errors, or removing it changes no ordinary retrieval metric. |
| RQ3: Do exact repeated captions create inconsistent gradients under established group relevance? | Some diagonal-only negatives are known or strongly implied positives under the same annotation convention. | Exclude exact repeats from negative denominators versus group-positive training; unchanged evaluator | Recall, seed variance, off-diagonal gradient contribution. Contradictory pressure falls without sacrificing unique-caption performance. | No measurable effect; still retain a transparent label policy, but drop it as a research claim. |
| RQ4: Is manual-only representation still a bottleneck after matched pretraining? | Non-manual evidence adds information beyond RGB/manual pose at fixed resources. | Add/remove existing face/head observations with matched parameter/compute controls | Recall and errors stratified by existing annotations or clearly labeled perturbations | Gain disappears under matched pretraining or is explained entirely by detector quality/extra compute. |
| RQ5: Does order-destroying augmentation encourage an avoidable compositional shortcut? | Valid sequence-preserving augmentation can retain ordinary robustness without erasing some order distinctions. | Original random-swap text augmentation versus no swap, with all other factors fixed | Retrieval; response to temporal reversal/local shuffling; real error examples | No ordinary gain or targeted change; perturbation sensitivity alone does not establish linguistic correctness. |
| RQ6: Does null matching help only if informative evidence is retained? | Partial matching reduces transition/function-token mismatch, but excessive rejection masks difficult distinctions. | No null / one-sided uniform null / fixed-mass two-sided null; equal encoder | Recall, real mass, per-word retention, matched versus random evidence removal | Improved loss with no ranking gain, or rejection disproportionately removes negation/numbers/relations. |
| RQ7: What does learned retrieval uncertainty actually measure? | Some variance heads mostly encode nuisance or candidate popularity rather than semantic ambiguity. | Controlled missing-cue perturbations and independent candidate ambiguity strata | Calibration to observed relevance/error, variance–error association and recall | Variance does not predict realized error beyond deterministic scores; no support for a probabilistic-retrieval contribution. |

All questions use existing data. Subset analyses and perturbations are diagnostics of the established system, not a proposed dataset, benchmark, or substitute main evaluation.

## 9. Adjacent-Domain Literature Mining

### Techniques worth transferring, and techniques already too close

| Primary work / venue | Original task and useful principle | Official code / inspection | SLRet implication and decision |
|---|---|---|---|
| [TACo, ICCV 2021](https://arxiv.org/abs/2108.09980) | Video–text alignment; token-aware contrast and cascade hard negatives | Paper equations inspected; not a borrowed implementation dependency | Token-level contrast and content-token weighting are not new. Its token/video matches are independently optimized, unlike the proposed shared-support intervention. |
| [FineCo, AACL 2022](https://aclanthology.org/2022.aacl-main.53/) | Frame-level contrast for video–text retrieval without frame annotations | Paper inspected; no module imported | Relevant-frame selection is established. Must outperform a straightforward local-loss control before claiming new grounding. |
| [FSC-CLIP, EMNLP 2024](https://arxiv.org/abs/2410.05210) | Image/text compositional retrieval; local HN loss plus focal/label-smoothed regularization | [Official repository](https://github.com/ytaek-oh/fsc-clip); inspected src/training/losses/loss.py and model utilities, revision 604015db | Closest Method 1 control. It recomputes visual support separately for positive/negative tokens and aggregates all tokens. Generic “local negatives preserve global capability” is explicitly rejected as novelty. |
| [CE-CLIP, CVPR 2024](https://arxiv.org/abs/2306.08832) | Contrast intra-modal negatives and rank cross-modal negatives with adaptive margins | [Official repository](https://github.com/lezhang7/Enhance-FineGrained); relevant paper equations inspected; no implementation imported | Bounded pushing and adaptive margins are old ideas. A hinge-loss control is necessary; a margin alone is not a contribution. |
| [HBI, CVPR 2023](https://arxiv.org/abs/2303.14369) | Hierarchical frame/word interaction using Banzhaf interactions, token merging and self-distillation | [Official repository](https://github.com/jpthu17/HBI); paper inspected, not a borrowed code component | Fine-grained interaction attribution and hierarchy are crowded. Its extra machinery is not justified for the primary hypothesis. |
| [TokenFlow, 2022 preprint](https://arxiv.org/abs/2209.13822) | Late interaction with globally informed token weighting / OT-inspired reasoning | Paper inspected; peer-reviewed venue **UNVERIFIED** | Weighted token matching and global guidance are prior art. Do not claim them as new sign-language mechanisms. |
| [READ-PVLA, AAAI 2024](https://github.com/nguyentthong/READ) | Low-resource temporal grounding/summarization; partial visual–language transport | Official TLG/models/model.py::partial_ot inspected, revision 14ce59f2 | Partial OT already has a visual-language history. Literal implementation is rejected because of mass/shape issues; no dependency on it. |
| [SuperGlue, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) | Feature matching with explicit unmatched points and log-domain Sinkhorn | [Official source](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/superglue.py), functions log_sinkhorn_iterations and log_optimal_transport inspected | Reuse only the stable log solver. Its unit point masses, one-to-one extraction and keypoint GNN are inappropriate defaults for signs/words. |
| [ALBEF, NeurIPS 2021](https://arxiv.org/abs/2107.07651) | Image/text learning with a momentum teacher and soft targets | [Official source](https://github.com/salesforce/ALBEF/blob/b9727e43c3040491774d1b22cc27718aa7772fac/models/model_retrieval.py), teacher update/copy and target loss inspected | Reuse an EMA of the same student, with no extra pretrained teacher or external captions. Adapt the supervised object from pair labels to correspondences between two known samplings. |
| [Drop-DTW, NeurIPS 2021](https://arxiv.org/abs/2108.11996) | Ordered sequence alignment with outlier dropping | [Official source](https://github.com/SamsungLabs/Drop-DTW/tree/32ce9c82c6a0d717a94f4139b1902ad146923444), dp/soft_dp.py, exact_dp.py, dp_utils.py, model losses inspected | Requires order-compatible sequences. Reject as a direct word↔sign constraint; could align same-video views, but timestamps already supply that mapping more simply. |
| [PCME++, ICLR 2024](https://arxiv.org/abs/2305.18171) | Probabilistic image/text embedding and noisy relevance | [Official loss](https://github.com/naver-ai/pcmepp/blob/8829c0d75b734e36c193b998b5963c1187fa59b2/pcmepp/criterions/pcmepp.py) inspected | Closed-form distance, BCE and optional pseudo positives are implementable; not selected without evidence that variance improves relevant ambiguity. |
| [ProLIP, ICLR 2025](https://arxiv.org/abs/2410.18857) | Probabilistic vision-language pretraining at large scale | [Official repository](https://github.com/naver-ai/prolip/tree/59f632451dc9608d7afa35e583292a2532f12657), src/prolip/loss.py and model.py inspected | Large-scale VLM results do not isolate a low-cost SLRet method. Do not import its external model/data advantage as methodological evidence. |
| [GARE, 2025 preprint](https://arxiv.org/abs/2505.12499) | Pair-specific semantic increments to mitigate contrastive gradient tension | Primary formulation inspected; not imported | Generic “reduce hard-negative gradient conflict” is also prior art. A shared-support claim needs its own intervention and visual diagnostics. |
| [PSCL, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/660cf2a1eabe448920a3ab6754555adb-Abstract-Conference.html) | Part-aware semantic contrast for fine-grained recognition | [Official repository](https://github.com/joker-lin9/PSCL); primary paper screened, no imported implementation | Part/category semantics can help recognition, but named visual parts are not a ready-made lexical sign ontology. |
| [SLAP, August 2026 preprint](https://arxiv.org/abs/2608.08840) | Partial alignment in fine-grained re-identification with local visual/textual evidence | Primary paper inspected; not imported | Partial OT plus local semantic learning is already used beyond sign language. No claim of first such combination is defensible. |
| [VTaMo](https://arxiv.org/abs/2607.09126) and [DualAnchor](https://arxiv.org/abs/2607.27614), July 2026 | Sign translation with token-level OT, unmatched support and auxiliary constraints | VTaMo official ot_sinkhorn.py, model.py and clip_loss.py inspected; DualAnchor source code availability **UNVERIFIED** | They eliminate generic OT/null/teacher combinations as a novelty claim for Method 2. |

Person re-identification, pose action recognition and fine-grained recognition provide useful principles—part reliability, structured body features, nuisance suppression—but their usual identity invariance can conflict with sign-language location and role information. No additional pose/re-identification module is included merely to cover those literatures. Similarly, mixture-of-experts and multiscale fusion are plausible capacity tools, but no inspected evidence establishes a need for them in either selected hypothesis.

### Exact reusable-code contracts

| Component | Required tensor input | Output / gradients | Dependencies, stability, overhead |
|---|---|---|---|
| SuperGlue log Sinkhorn | Log-kernel \([P,L+1,M'+1]\); log row/column masses \([P,L+1]\), \([P,M'+1]\); \(P\) here is number of scored pairs, not pose | Log plan of same shape. Differentiable through iterations if desired. | Pure tensor operations; no keypoint checkpoints. Use float32, feasible equal total marginals, exact exclusion of padding and dummy–dummy edge, residual checks. \(O(I P L M')\) time; retaining every iteration for autograd multiplies memory. |
| ALBEF EMA | Student parameter tensors and matching frozen teacher tensors; initialize teacher from student | \(\bar\theta\leftarrow\mu\bar\theta+(1-\mu)\theta\). Teacher outputs computed without gradients. | One parameter copy and a no-grad forward on the teacher view. Do not transplant its queue, cross-encoder, image backbone or large external training. No teacher optimizer state. |
| FSC-CLIP local HN, as control | Image/local video tokens \([B,L,d]\), text \([B,M',d]\), negative texts and masks | Candidate-dependent pooled visual token features, then per-caption local logits and focal/CE losses | Existing source detaches the affinity map before pooling; it **does not** lock positive and negative support together. Min–max normalization has a zero-range stability risk; use its softmax option or a documented epsilon consistently. |
| PCME++ distance, considered but rejected | Means \([B,d]\), “std” field containing log variance \([B,d]\), relevance labels | Pairwise CSD logit from squared mean distance plus variances; BCE | Avoid materializing \([B,B,d]\) when a norm/dot formulation suffices. Requires calibrated use of labels; variance regularization and pseudo-positive thresholds can be unstable. |
| Drop-DTW, considered but rejected | Step/frame features or pairwise match/drop costs; ordered lengths \(K,N\) | Alignment cost/table or assignments, depending on routine | \(O(KN)\) dynamic programming with implementation-specific loops and memory. Alignment semantics, not tensor compatibility, is the reason not to impose it on word/sign sequences. |

**YOUR INFERENCE about probabilistic ranking:** PCME++'s inspected CSD contains \(\|\mu_q-\mu_c\|^2+\sum_d(\sigma_{qd}^2+\sigma_{cd}^2)\). For one fixed query, its variance sum is constant across candidates and cannot directly reorder them. Candidate variance can reorder them, and probabilistic training can change the means; neither fact demonstrates calibrated query ambiguity. This is why probabilistic embeddings are a research option rather than the primary recommendation.

## 10. Method 1 — Highest-Confidence SOTA Candidate

### Name and central hypothesis

**Shared-support sign contrast** — a proposed method, not an existing publication.

**YOUR HYPOTHESIS:** when a caption is altered at a small number of lexical spans, independently aligning the original and alternative can let the model compare different visual evidence. Comparing the alternatives on the same positive-reference support should make the visual distinction more learnable while reducing unnecessary pressure on shared sentence content.

The demonstrated motivation is SAN's fine/full-pool tradeoff. The specific support-hopping explanation is unverified and must be tested before claiming causality. The design does not claim that local hard negatives, stopped gradients, or margin losses are new.

### Baseline and scope of the SOTA attempt

Use [UPRet's official repository](https://github.com/xua222/UPRet/tree/046366227417e1d8ec14145965403462df345984) as the first implementation baseline, trained independently from the published CiCo I3D feature regime and CLIP initialization. Preserve UPRet's original retrieval/distribution objective, feature mixture and inference scorer. Repair evaluator and padding issues identically in all reproduced arms.

This is the strongest practically inspectable RGB/CLIP scaffold in the audited set, not the numerically strongest system on every benchmark. The fair hurdle is the **best** of independently reproduced UPRet, SAN-style training and FSC-CLIP+SAN-style training with identical resources. A broader SOTA claim additionally needs comparison against the SEDS/C²RL system frontier, and transfer of the same loss to a stronger independently reproduced paired-text-pretrained representation if the RGB feature ceiling blocks progress. That transfer is a required publication gate, not a hidden component of the initial method.

No SEDS checkpoint, SEDS teacher output, new gloss labels, external negative-generation LLM, or extra manually annotated data is used.

### Complete architecture and tensor path

| Path | Representation | Trainability |
|---|---|---|
| Video input | RGB video \([B,F,H,W,3]\) | Existing I3D extraction done offline; no backbone gradient |
| Local sign features | Fused domain-aware/agnostic I3D \([B,L,1024]\), \(L\le64\) | Frozen feature values, original clip indices retained |
| Contextual video | Projection 1,024→768, CLIP-initialized temporal Transformer, final projection → \(X\in\mathbb R^{B\times L\times512}\) | Trainable; shared by base retrieval and new loss |
| Text input | Original captions \([B,M']\), \(M'\le32\); character-to-subword span mapping | Same tokenizer and language as the baseline |
| Contextual text | \(Y\in\mathbb R^{B\times M'\times512}\) | Trainable through original retrieval objective |
| Base alignment | Pairwise \([B_v,B_t,L,M']\), original UPRet weighted directional aggregation | Unchanged except common documented masking repair |
| Base global features | Existing pooled/sample sentence features \([S,B,512]\), \(S=2\), for UPRet training | Original distribution branch retained; no new global head |
| Reference model | Frozen copy of the independently trained baseline | Supplies support and original/alternative span targets; never a SEDS model |
| New branch | Support \([B,K,E,L]\), span differences \([B,K,E,512]\), margins \([B,K,E]\) | Only student video features receive the auxiliary gradient |

\(K\) counts negative captions; \(E\) is the number of changed spans in one negative, normally one for diagnosis and up to two for SAN-style training. The ordinary inference path is exactly the reproduced baseline.

### Train-only miner, with explicit limitations

The SAN release expects a precomputed table and does not provide a complete constructor. A reproducible implementation must therefore define that constructor rather than assume it exists.

1. Train the reference baseline using training pairs and dev-only model selection.
2. Map complete lexical spans to BPE indices. Exclude special/padding tokens; skip any edit truncated by the established token cap.
3. For each training word occurrence, use reference clip–span affinities to obtain a soft visual support and pooled visual descriptor.
4. Build a training-only word candidate table from high-confidence descriptors. A simple first version uses normalized per-word prototype means and stores occurrence counts/dispersion. A word with too few stable occurrences is ineligible.
5. Rank distinct word candidates by visual-descriptor cosine similarity, following SAN's visual-hardness principle. Exclude identical forms and alternatives already present in the caption. Do not classify a pair as semantically different merely because its strings differ.
6. Generate the same cached alternatives, with the same seeds/counts, for every hard-negative control and proposed arm.

This is a **SAN-inspired reimplementation**, not a claim to reproduce its unpublished miner exactly. Prototype averaging can hide polysemy and signing variation; this is deliberately a shared limitation of all arms, not another claimed contribution. If the official full miner becomes available, replace the shared miner in all arms. Miner thresholds start from SAN's reported .7 confidence/similarity settings, but their calibration in a different encoder/language is **UNVERIFIED**.

The number of selected pairs, occurrence thresholds, caption-edit frequency, exact-duplicate rate and visual similarity distribution must be reported. No additional teacher pretraining is concealed in this stage.

### Novel module and equations

Let \(\bar f,\bar g\) be the frozen reference encoders and \(f_\theta,g_\phi\) the student. For original caption \(T_i\), altered caption \(T^-_{ih}\), and changed span \(e\), define normalized contextual span features
\[
\bar q^+_{ihe}=
\operatorname{norm}\left(\frac1{|J^+_{ihe}|}\sum_{m\in J^+_{ihe}}\bar y_{im}\right),
\quad
\bar q^-_{ihe}=
\operatorname{norm}\left(\frac1{|J^-_{ihe}|}\sum_{m\in J^-_{ihe}}\bar y^-_{ihm}\right).
\]
\(J^+\) and \(J^-\) are the complete positive/negative BPE spans, not arbitrary individual subwords.

Obtain **one** reference support from the original span:
\[
a_{ihe,l}=
\operatorname{stopgrad}\left[
\operatorname{softmax}_{l\in\mathcal I_i}
\left(\bar x_{il}^{\top}\bar q^+_{ihe}/\tau_s\right)
\right],
\]
where \(\mathcal I_i\) is the valid clip-index set and \(\tau_s\) is a support temperature. No negative-specific support is recomputed in the proposed branch.

The discriminative margin is
\[
\Delta_{ihe}=
\sum_{l\in\mathcal I_i}a_{ihe,l}\,
x_{il}^{\top}
\operatorname{stopgrad}(\bar q^+_{ihe}-\bar q^-_{ihe}).
\]
The same visual sequence and the same weights support both alternatives. Only changed spans are contrasted; unchanged words are not additional negative targets.

Use
\[
\mathcal L_{\rm shared}
=
\frac{\sum_{i,h,e}c_{ihe}\,[m-\Delta_{ihe}]_+}
{\max(1,\sum_{i,h,e}c_{ihe})},
\qquad
\mathcal L=\mathcal L_{\rm UPRet}+\lambda\mathcal L_{\rm shared}.
\]
\([z]_+=\max(z,0)\); \(m\) is a small cosine-margin target; \(\lambda\) controls auxiliary strength; \(c\in[0,1]\) is an optional support-reliability weight. The hinge stops penalizing already separated alternatives. The original UPRet objective still trains both student encoders and preserves ordinary paired retrieval pressure.

At the normalized feature-slot level,
\[
\frac{\partial \Delta_{ihe}}{\partial x_{il}}
=a_{ihe,l}(\bar q^+_{ihe}-\bar q^-_{ihe}).
\]
Thus the auxiliary signal has an explicit support and a discriminative text direction. The normalization Jacobian applies when backpropagating to raw features.

**Scope of the claim:** contextual video slots have a global receptive field. This derivative does not prove that only pixels in an actual sign interval change, nor does it prove global sentence geometry is preserved. Similarly, contextual span differences can contain contextual effects. These are measured limitations, not “causal localization” guarantees.

### Supporting mechanism

Start with \(c=1\) for eligible edits. Add reliability only if the reference supports prove unstable. A simple annotation-free gate measures agreement between supports computed from two mild, order-preserving samplings of the same feature sequence, mapped back to common clip indices. Use normalized entropy and overlap/agreement to reject diffuse or inconsistent supports.

This gate measures **alignment stability**, not semantic truth. It must not be advertised as resolving synonymy or false negatives. Do not add an uncertainty head, new decoder or extra pose stream to make an unstable teacher look reliable.

### Training procedure

1. Reproduce the repaired baseline with dev-only selection and log all resource/protocol details.
2. Freeze one selected reference copy; cache tokenized negative captions, changed-span indices and reference text span vectors.
3. For each batch, run the student through the unchanged baseline forward path.
4. Obtain reference visual support for the same sampled clips; either compute it online without gradients or cache it when the sampling is fixed.
5. Compute only the \([B,K,E,L]\) auxiliary comparisons. Backpropagate to the student's shared video encoder, not a disconnected auxiliary head.
6. Fine-tune for a short, predefined extra budget; give every comparison arm the same extra optimizer steps and teacher/mining budget.
7. Select hyperparameters/checkpoint on ordinary dev retrieval in both directions. Evaluate the untouched standard test pool once per locked experiment.

A suggested **initial experimental grid**, not established optimal values, is \(K\in\{1,5\}\), \(m\in\{0.05,0.1\}\), and \(\lambda\in\{0.05,0.1,0.2\}\), with \(\tau_s\) initially .07. Use one \(K\) for the decisive matched comparison before expanding the grid. Do not tune dozens of combinations against test stress performance.

### Inference and cost

Discard the reference model, candidate table and auxiliary branch at test time. Cache the same video/text features and use the same UPRet scorer/evaluator as the baseline. **No extra inference cost is required.**

Training adds a no-grad reference visual forward when support cannot be cached, negative-text encoding/caching, and small within-example contractions. It does not add a second RGB backbone training pass or an all-\(B^2K\) negative affinity tensor. Actual runtime/peak memory must be measured; a numeric speedup is not claimed.

### Why earlier work is not already this method

| Earlier method | Overlap | Required distinction |
|---|---|---|
| CiCo | Contextual token–clip alignment | Does not compare changed positive/negative spans on a single fixed support. |
| SEDS | Better visual representation and stream alignment | Does not impose this evidence-sharing constraint; adding pose alone does not isolate negative-loss effects. |
| UPRet | Weighted local similarity and probabilistic training | OT operates on pooled samples; no shared-support lexical negative intervention. |
| C²RL | Content/context learning | Does not isolate sign-aware alternative discrimination with locked evidence. |
| SAN | Visually informed caption alternatives | Supplies negative distribution; full-caption loss does not force alternatives to use the same support. |
| FSC-CLIP | Local HN and calibration; the closest prior | Source detaches each candidate's affinity map but recomputes it separately. Proposed branch ties that map and contrasts only edited spans. |
| TACo / ConSLT / MCL-SLT | Token contrast / token negatives | Token contrast itself is old; changed-span evidence sharing is the proposed distinction. Final MCL-SLT full-text overlap remains **UNVERIFIED**, so no universal “first” claim is made. |
| VTaMo / DualAnchor | Token grounding and auxiliary regularization | No matched-positive-support comparison of visual hard-negative alternatives; their reported primary task is generation. |

### Novelty defense, expectations, risks and kill criteria

**Hostile objection:** “This is SAN plus FSC-CLIP.” That objection is valid against the initial generic local-negative idea. The design is narrowed to the independently testable support-sharing intervention. If FSC-CLIP+SAN with span-only scoring matches it, the novel claim fails; do not rescue it by renaming losses or adding modules.

**Expected result:** moderate improvement over a strong matched baseline is plausible; exceeding the broader How2Sign system frontier is uncertain; a large jump past C²RL on CSL-Daily from fixed I3D features is not a credible default expectation.

Risks are wrong support, semantic false negatives, contextual leakage, insufficient hand detail in frozen features, reference/student space drift, sparse eligible edits, and overfitting PHOENIX vocabulary. The base retrieval loss mitigates some drift but provides no guarantee.

**Kill criteria:** abandon the method if, under identical negatives and optimization, locked support gives no reproducible advantage over independently aligned span contrast; improvements occur only on synthetic stress candidates; ordinary V2T consistently degrades; random support works as well; or the method's gains disappear when evaluator/padding/duplicate fixes are applied equally to controls. If reference support quality is too poor to discriminate matched from random temporal evidence, stop rather than pile on teacher modules.

## 11. Method 2 — Cross-Paper Compositional Candidate

### One unifying principle

**Sampling-consistent partial alignment:** local matching should reflect the amount and location of observed signing evidence, rather than the number of overlapping clips or subword pieces used to represent it.

This is a higher-risk hypothesis. Independent local normalization and overlapping windows are verified facts; a causal connection to substantial real retrieval errors has not yet been demonstrated.

### Borrowed components and provenance

Two major ideas are borrowed. The partial-matching principle has additional close sign-language precedents, which are acknowledged rather than repackaged.

| Component | Source paper / venue / original task | Public code and exact principle | SLRet adaptation | Why compatible |
|---|---|---|---|---|
| Stable constrained matching | SuperGlue, CVPR 2020, keypoint matching; partial-matching precedents READ-PVLA/AAAI 2024 and DualAnchor/2026 | [Official SuperGlue solver](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/superglue.py), log_sinkhorn_iterations; log-domain marginal balancing | Replace point masses with physical-coverage and word masses, use two-sided unmatched capacity and fixed real match mass; no keypoint GNN, no one-to-one argmax matching | Operates directly on the existing normalized clip/token affinity matrix; no backbone/checkpoint dependency |
| Stable self-teaching | ALBEF, NeurIPS 2021, image/text learning | [Official retrieval model](https://github.com/salesforce/ALBEF/blob/b9727e43c3040491774d1b22cc27718aa7772fac/models/model_retrieval.py), copy_params and _momentum_update | Teacher is the EMA of the same SLRet student; supervise correspondence agreement after projecting two samplings to the same physical time axis | Uses the same representation, no extra annotations or pretrained teacher; supplies stable targets for an otherwise moving assignment |

READ's public partial_ot function is **not imported**, because the inspected mass/shape implementation is unsuitable. VTaMo's token/frame/null formulation and DualAnchor's fixed-mass two-dustbin construction are closest prior art. Fixed-mass partial OT itself is explicitly not a novel component here.

### Architecture and evidence accounting

Use the same reproduced RGB/CLIP baseline encoder and normalized \(X\in\mathbb R^{B\times L\times512}\), \(Y\in\mathbb R^{B\times M'\times512}\). The inference scorer will change, so retain the baseline scorer as a controlled residual. No new spatial, pose, language or foundation-model backbone is added.

For one video, clip \(l\) covers known original-frame indices \(\mathcal I_l\). Let
\[
c_t=\sum_l\mathbf1[t\in\mathcal I_l],\qquad
Z=|\{t:c_t>0\}|,\qquad
a_l=\frac1Z\sum_{t\in\mathcal I_l}\frac1{c_t}.
\]
Then \(\sum_l a_l=1\): overlapping observations divide coverage mass rather than creating new mass. Store timestamps/indices from the existing extraction pipeline; these are sampling metadata, not sign-boundary labels.

Exact duplicate observations at the same interval are merged before computing coverage masses, transport and time projection. The displayed equations operate on these unique observations. An implementation retaining explicit identical copies must split the original assigned mass and use the corresponding split projector consistently, rather than recomputing overlapping coverage incorrectly. **A repeated sign at a different time is not a duplicate observation and retains its own mass.** General invariance across different contextualized samplings is only approximate; the exact duplication property below assumes identical fixed feature values.

For text with \(W\) lexical units and BPE set \(\mathcal B(w)\) for each word,
\[
b_m=\frac1{W|\mathcal B(w)|}\quad\text{for }m\in\mathcal B(w).
\]
Padding/special tokens have no mass. This avoids giving a word extra evidence merely because it is split into more BPE pieces. Retain function words in the basic experiment: they may express important grammatical distinctions, and “content-word filtering” is not a sign-language oracle.

### Partial transport score

Let \(A_{lm}=x_l^\top y_m\). For a preset \(\rho\in(0,1]\), define
\[
\mathcal U_\rho(a,b)=
\left\{P\ge0:P\mathbf1\le a,\ P^\top\mathbf1\le b,\
\mathbf1^\top P\mathbf1=\rho\right\}.
\]
Real mass \(\rho\) is transported; the remaining mass on each side goes to unmatched states. To implement this with the audited balanced solver, append one dummy row and one dummy column:
\[
\tilde a=[a;1-\rho],\quad
\tilde b=[b;1-\rho],\quad
\tilde P_{L+1,M'+1}=0.
\]
Both total marginals equal \(2-\rho\). The forbidden dummy–dummy edge forces real-to-real mass to equal \(\rho\). For \(\rho=1\), use the real balanced problem without zero-mass dummy rows.

Use an entropic objective relative to the mass prior:
\[
F(A)=\max_{\tilde P}
\left\{\langle P,A\rangle
-\epsilon\,\operatorname{KL}(\tilde P\Vert\tilde Q)\right\},
\quad
\tilde Q=\frac{\tilde a\tilde b^\top}{2-\rho},
\]
with the specified marginals and forbidden edge. The KL is the generalized nonnegative-measure KL; invalid entries are excluded consistently. Define
\[
s_{\rm OT}(V,T)=\frac{F(A)-F(0)}{\rho}.
\]
Subtracting the zero-affinity reference removes an entropy/marginal offset that otherwise changes with lengths and mass configurations. Both the reference subtraction and refinement-aware masses must be ablated: their value is not assumed.

When an identical observation's mass is split across identical copies, a KL-to-mass-prior formulation permits its transport to split proportionally without creating a score advantage. This is an evidence-accounting property, not a new general OT theorem. It does not imply invariance to real motion changes, omitted signs or changed contextual features.

Retain an ordinary retrieval score:
\[
s=(1-\eta)s_{\rm base}+\eta s_{\rm OT},
\]
where both scores are put on a comparable pre-temperature scale and \(\eta\) is selected on dev. The base scorer retains the original contextual alignment bias; the transport term tests whether coherent evidence allocation adds value.

### Correspondence consistency across samplings

An EMA teacher \(\bar\theta\leftarrow\mu\bar\theta+(1-\mu)\theta\) processes one ordered sample of a training video. The student processes another mild ordered sample covering substantially the same interval. No temporal reversal, synthetic sign order or cross-sentence monotonic correspondence is assumed.

Project clip assignment to physical time using
\[
D_{tl}=\frac{\mathbf1[t\in\mathcal I_l]}
{Zc_ta_l}.
\]
Each column of \(D\) sums to one. If \(P\mathbf1\le a\), then the real assignment \(Q=DP\) has total mass at each covered frame at most \(1/Z\). Append each clip's unused mass \(a_l-\sum_mP_{lm}\) as a null column before projection; then form a normalized per-time distribution \(q_t\) over text tokens plus null.

On the physical time positions observed by both views, minimize
\[
\mathcal L_{\rm cons}=
\frac1{|\mathcal O|}
\sum_{t\in\mathcal O}
\operatorname{KL}\!\left(
\operatorname{stopgrad}(\bar q_t)\Vert q_t
\right).
\]
\(\mathcal O\) is the common observed support, obtained from existing timestamps. A coarse fixed time grid can reduce memory; it is an internal computation, not a benchmark annotation.

This transfers ALBEF's stable self-teaching principle to **correspondence**, rather than transferring its full model or soft pair-label queue. The teacher is not asserted to be correct; the loss tests whether a stable shared observation should maintain a stable alignment.

### Objectives, stages and inference

Train with the original symmetric retrieval objective using \(s\), retaining baseline-specific training regularizers equally where applicable:
\[
\mathcal L=\mathcal L_{\rm ret}(s)+
\mathcal L_{\rm baseline\ auxiliary}+
\lambda_{\rm cons}\mathcal L_{\rm cons}.
\]
Warm up the reproduced baseline first. Introduce physical-coverage partial scoring, then add consistency only after marginal residuals and assignments are numerically stable. Do not jointly introduce OT, teacher consistency, new data and new encoders.

At inference, discard the EMA teacher, compute student features, and score the complete established candidate pool with chunked partial transport plus the base score. **Inference is more expensive than Method 1.** A top-\(K\) reranker or distilled fast score would be an additional experimental variant, not an undisclosed shortcut in the main result.

### Compatibility analysis

| Component pair | Representation / functional relationship | Loss, sampling and gradient compatibility | Memory / stage decision |
|---|---|---|---|
| Coverage masses + partial transport | Masses define constraints on the same \([L,M']\) affinity | No competing loss. Masses are fixed metadata; gradients update similarities, not timestamps. | Negligible mass storage; introduce first. |
| Transport + EMA consistency | Teacher/student plans concern the same video/text, projected to common time | Retrieval determines correct-pair ranking; consistency can conflict when the teacher is wrong. Warm up and keep its weight small; remove it if it hurts. | Extra teacher parameters/no-grad forward and positive-pair plan graph. No teacher optimizer. |
| Base scorer + transport | Same contextual clip/text features; complementary independent versus constrained matching | Score scales must be matched; otherwise \(\eta\) is meaningless. Same positive pairs and pool. | Adds per-pair solver cost; no duplicate backbone. |
| Coverage + two ordered samplings | Both rely on known physical indices | Compatible only on their observed intersection. Comparing raw token indices across views is wrong. | Projection can use sparse interval weights or a bounded time grid. |
| Existing UPRet sample regularizer + new clip/token transport | Sample-level pooled uncertainty versus actual local correspondence | Not identical functions; retain it in initial matched arms, then ablate if redundant. No claim that keeping both is automatically beneficial. | Tiny sample OT is cheap relative to local transport. |

A default full differentiable solver at \(B=512\) would be wasteful. Compute plans in float32 pair blocks; for the converged regularized score, the envelope gradient with respect to \(A\) is the real plan \(P\), so the retrieval branch need not retain every solver iteration. Use a verified implicit/envelope implementation or recomputation, rather than pretending a detached plan gives the exact gradient of an arbitrary non-regularized score. The consistency branch needs differentiable student plans only for positive pairs.

At \(L=64,M'=32\), a block of 256 augmented cost matrices has \(256\times65\times33\) entries, about 2.1 MiB in float32. This excludes activations and intermediate buffers. With \(I\) Sinkhorn iterations, matching time is \(O(I P L M')\); total full-pool work can be materially larger than baseline late interaction. Report real latency, peak memory and ranking cost.

Initial experimental values might be \(\rho\in\{0.7,0.85,1.0\}\), \(\epsilon\in\{0.05,0.1\}\), and \(\mu=.995\); these are **proposed starting settings**, not paper-backed optimal values. Verify marginal residuals and numerical stability before any hyperparameter search.

### Novelty, expected upside and kill criteria

An initial “partial OT + null + teacher” proposal is rejected: READ, VTaMo and DualAnchor already cover too much of it. The retained claim concerns **physical-evidence capacities and correspondence consistency under changes in observation sampling**. Novelty risk remains medium–high; merely adding weighted marginals to DualAnchor is insufficient for a strong paper.

**Expected improvement:** uncertain. It may help long or heavily overlapping sequences if repeated evidence contributes to errors. It may instead suppress legitimate simultaneous/multiword content, over-discard short critical signs, or be too slow for its gain.

**Kill criteria:** no meaningful baseline rank sensitivity to redundant observations; uniform or simple deduplicated pooling matches the full method; consistency helps no more than ordinary feature consistency; discarded mass concentrates on informative words; or full-pool improvement is negligible relative to inference cost. If a simple quadrature-weighted CLCL score resolves the problem, use that simpler result and reject this compositional architecture.

## 12. Method 1 vs Method 2

Scores are **YOUR INFERENCE**, 1–10, with larger values favorable. For difficulty, cost, tuning and reviewer risk, larger means easier, cheaper, less tuning and lower risk. The probability rows are ordinal judgments, not estimated numerical probabilities.

| Criterion | Method 1 | Method 2 | Reason |
|---|---:|---:|---|
| Novelty after closest-prior checking | 5 | 5 | Both require narrow claims; FSC-CLIP/SAN and DualAnchor/VTaMo are close. |
| Expected practical gain | 6 | 5 | Method 1 targets an observed training tradeoff; Method 2 first needs evidence that sampling sensitivity causes errors. |
| Implementation ease | 8 | 5 | Method 1 adds a local contraction/loss and mining metadata; Method 2 adds constrained scoring and correspondence projection. |
| Computational affordability | 8 | 4 | Method 1 leaves inference unchanged; Method 2 scores candidate pairs through a solver. |
| Reproducibility | 7 | 6 | Both inherit baseline repairs; Method 1 additionally needs a transparent miner, Method 2 numerical convergence checks. |
| Low dependence on hyperparameter tuning | 7 | 4 | Method 2 introduces mass, entropy, mixing and consistency settings. |
| Low reviewer risk | 6 | 4 | Method 1 can isolate one intervention; Method 2 may reduce to weighted OT plus regularization. |
| Compatibility with existing SLRet datasets | 8 | 7 | Both use existing paired data. Method 1 needs eligible lexical edits; Method 2 needs trustworthy extraction timestamps. |
| Public source availability | 7 | 8 | UPRet/FSC/SAN scaffold versus audited SuperGlue/ALBEF routines; neither proposal itself is implemented yet. |
| Likelihood of beating the strongest resource-matched control | 6 | 4 | Moderate versus Low research confidence, conditional on diagnostics. |
| Likelihood of beating the entire published system frontier | 3 | 3 | Unverified newer results and stronger representations make a broad claim premature. |
| Publishable insight if recall does not reach SOTA | 7 | 6 | Clear causal ablations can reveal support/negative failure; sampling effects must be substantial to matter. |

**Primary recommendation: Method 1.** Its principal intervention is small enough to compare directly against sophisticated prior art. Failure can be recognized early, and success need not depend on more parameters, data or test-time computation.

**Backup recommendation: Method 2.** Pursue it only if frozen-feature diagnostics show that redundant observations meaningfully alter erroneous rankings. A merely elegant transport formulation is insufficient motivation.

If Method 1 cannot beat a tuned FSC-CLIP+SAN control, do not automatically switch to the larger architecture. First check whether the feature representation or negative validity, rather than loss design, is the limiting factor.

## 13. Recommended Experiments and Ablations

### Prioritized sequence

1. **Establish one trustworthy baseline.** Start with PHOENIX for rapid debugging and How2Sign for broader-domain confirmation. Verify splits, groups, caption language, feature hashes and valid masks. Reproduce published behavior as far as possible, but separate historical numbers from repaired dev-selected runs.
2. **Measure the alleged mechanism before training the method.** On the baseline, measure how much the support moves between original/altered captions, whether moved support is associated with mistakes, and whether a locked-support score helps on real candidate confusions. These associations are not causal proof, but a null finding is an early stop.
3. **Run the decisive controlled intervention.** Same baseline, negatives, span encoding, margin, stopped-gradient policy, sampler, optimization and number of updates; change only independent versus shared support.
4. **Add the reliability gate only if reference instability is measurable.** It must beat both ungated shared support and the same gate applied to the independent-support control.
5. **Replicate on How2Sign and CSL-Daily.** A PHOENIX-only result can reflect weather vocabulary. If the method survives, test transfer to a stronger independently reproduced representation regime before making a broad SOTA claim.
6. **Run Method 2 only after its diagnostic gate.** Compare simple duplication/coverage weighting first; add OT, then correspondence consistency. Do not skip the cheap explanation.

### Minimum convincing Method 1 suite

All rows use the same corrected evaluator, paired data, baseline initialization, extra training budget and candidate pool. The four-row A/B factorial operates on the same strong hard-negative control; a reliability gate is not meaningful as a standalone loss without negatives.

| Arm | What changes | What it tests |
|---|---|---|
| Reproduced UPRet | No new negative objective | Strong public scaffold and absolute progress |
| UPRet + SAN-style negatives | Published-style full-caption hard-negative objective; shared new miner | Whether visual negative selection alone suffices |
| UPRet + FSC-CLIP-style local negatives + same miner | Candidate-dependent local matching plus its calibrated-loss control | Closest cross-paper baseline; rules out a simple transfer being enough |
| Strong span control | Changed-span scoring, independent positive/negative support, same hinge and frozen reference text/support | Controls span restriction, loss type, teacher and text-gradient changes |
| **+ Component A** | Lock positive/negative support in the strong span control | Central shared-evidence hypothesis |
| **+ Component B** | Reliability gate on the strong span control, retaining independent support | Whether filtering alone explains the effect |
| **Full A+B** | Shared support plus the same gate | Interaction between evidence sharing and support reliability |
| Random-support control | Shuffle support within the same video, preserving entropy/weight histogram | Whether correct visual localization matters |
| Matched global-margin control | Same negatives and hinge saturation, ordinary caption score | Whether simple loss bounding explains the gain |

If Component B is ineffective, the final method is Component A alone. Do not retain a component to satisfy an architectural diagram.

### Diagnostics that explain improvements

| Diagnostic | Construction using existing data | Interpretation / caveat |
|---|---|---|
| Visual versus semantic hardness | Record reference visual descriptor similarity and baseline text similarity separately for mined alternatives; use train-fitted bins on dev/test analyses | Show which notion of hardness predicts improvement. Similarity is a proxy, not a label of semantic equivalence. |
| Existing SAN stress candidates | Use the published frozen stress set only when its language/tokenization is compatible with the reproduced branch | Keep its V2T recall separate from standard full-pool results. An English translation of the stress captions is not automatically the same test. |
| Real candidate confusions | Analyze naturally occurring nearest distractors in the ordinary test pool; report baseline/model ranks and errors | Prevents a result based solely on artificial word substitutions. |
| Support movement | Compare original/negative attention overlap, center shift, entropy and selected temporal intervals | The expected benefit of sharing should be greater where independent support moves incorrectly; attention alone is not a faithful explanation. |
| Evidence removal | Remove/mask selected feature intervals versus matched-length random intervals, preserving a clearly stated input-corruption protocol | Correct support should matter more for the discriminative margin. This tests model dependence, not linguistic sign-boundary accuracy. |
| Gradient conflict | Measure cosine between base-retrieval and auxiliary gradients on shared visual parameters; compare changes in unchanged-token scores | A reduced conflict accompanies, but does not prove, the proposed mechanism. |
| Temporal order | Reverse or locally shuffle video features; compare with valid ordered subsampling and no text-swap training | Report as perturbation sensitivity only. Do not assign new semantic ground truth to the altered video. |
| Signer variation | Use existing signer IDs to report within-split per-signer performance and consistency | Do not create a new signer split as the headline benchmark. Small signer counts limit inference. |
| Length / truncation | Stratify by original duration, retained clips, text length and truncated edits | A gain only on short non-truncated weather captions has limited generality. |
| Exact duplicate captions | Report grouped/unique-text strata under the unchanged official relevance relation | Distinguish label-policy gains from the new loss. No automatic paraphrase labels. |
| False-negative risk | Report alternatives already present in the caption, duplicate substitutions, low-confidence teacher support and unstable word prototypes | These are auditable error indicators, not a complete semantic validity estimate. |
| Model 2 evidence budgets | Duplicate identical stored features at the same timestamps; separately duplicate actual temporal content at new timestamps | Only the first manipulation should have a duplication-invariance claim. |
| Model 2 null behavior | Inspect transported mass and sensitivity to dropping high-mass, low-mass and random evidence | Low OT cost alone can conceal rejection of meaningful distinctions. |

### Statistical and fairness requirements

Use at least three paired seeds for the decisive comparison if resources permit. Report mean and spread; use paired bootstrap intervals over the actual query unit, respecting group structure and correlated recordings. Select hyperparameters on dev; report both retrieval directions. A tiny R@1 gain with overlapping uncertainty and worsened higher-rank recall is not persuasive evidence of SOTA.

Keep equal feature extraction, FPS, clip selection, text preprocessing, external initialization, negative counts and optimizer steps across primary arms. Give controls an equal tuning budget, including their negative-loss weight and calibration choices. The reference teacher and mining pass count toward training cost.

For a broader system claim, evaluate or faithfully reproduce SEDS/C²RL under their respective declared resource regimes and report all differences. Existing target gloss can be used for separately labeled post-hoc diagnostics if appropriate; neither training nor checkpoint selection for the proposed gloss-free method should consume additional gloss supervision.

## 14. Reviewer Attack

### Reviewer A — Sign Language Expert

**“You are treating written words as signs, and calling a different word a valid linguistic counterfactual. Your attention map may be driven by context or mouthing; your ‘local’ features see the whole sentence.”**

**Assessment:** valid. Word spans are weak textual anchors, not annotated signs. Contextual slots are not isolated visual segments. A replacement may remain semantically acceptable, and not every written token has a directly corresponding manual sign.

**Modification:** the method is described as shared *model evidence*, not phonological or causal sign localization. It imposes neither one-word/one-sign nor monotonic matching. It retains the full retrieval objective and tests dependence on actual visual evidence. Ambiguous candidates receive no claim of certified semantic falsity. No fabricated gloss labels or universal sign vocabulary are introduced.

**“Your augmentations can remove grammatical order, role information or non-manual cues.”**

**Assessment:** valid if temporal reversal/word swapping is treated as a positive linguistic augmentation or if location/face information is indiscriminately removed.

**Modification:** the new mechanisms use ordered samplings of the same content. Reversal and shuffling are diagnostic corruptions, not positive examples. Original baseline augmentation is either retained as a clearly labeled reproduction condition or removed in every matched arm. Neither method claims invariance to all signing-space transformations.

### Reviewer B — Retrieval Expert

**“Method 1 is FSC-CLIP plus SAN. You changed notation and called it a new loss.”**

**Assessment:** valid for generic local hard negatives. FSC-CLIP already addresses the local/global tradeoff and its code already stops gradients through affinity maps.

**Modification:** the only new claim is shared positive/negative support with edited-span contrast. Require independent-support span matching, FSC-CLIP+the same miner, and global margin controls. If shared support adds nothing, reject the proposal. A performance gain from the miner, hinge, or text freezing alone is not evidence for the claimed contribution.

**“Method 2 is DualAnchor's partial OT plus ALBEF. Weighted marginals are standard numerical practice.”**

**Assessment:** largely valid. Generic transport, dustbins, fixed mass and EMA are not new.

**Modification:** remove those novelty claims. The scientific contribution must be a diagnosed sampling-induced retrieval error, a representation of unique physical evidence, and correspondence consistency on the same underlying video. Compare against uniform OT, plain coverage-weighted CLCL, deduplication and generic feature consistency. If these suffice, reject the larger composition.

**“You can always win by defining a weak ‘fairly comparable’ baseline and ignoring stronger methods.”**

**Assessment:** valid and serious.

**Modification:** report two hurdles: the best matched-resource control and the broader verified system frontier. Do not call a small gain over UPRet “current SOTA” while C²RL/SEDS remain stronger. Resolve CMCM and final-version uncertainty before any exhaustive superiority claim.

### Reviewer C — Implementation/Reproducibility Expert

**“Your baseline is not reproduced. Public repositories have masking bugs and incomplete miners; the reported gains may be fixes, extra compute, or a stronger teacher.”**

**Assessment:** valid. This investigation establishes code behavior and method feasibility, not numerical reproduction.

**Modification:** freeze a repair commit, data/group hashes and upstream initialization manifest before experiments. Publish repaired controls separately from historical values. The Method 1 teacher is a frozen copy of the same independent baseline; Method 2's teacher is its EMA. Equalize extra training and mining budget. No pretrained SEDS weights are allowed anywhere in the dependency graph.

**“Full-pool token transport is impractical, and your detached solver may optimize the wrong objective.”**

**Assessment:** valid unless implemented carefully.

**Modification:** Method 2 uses a regularized score with a justified envelope gradient, convergence/residual checks, pair blocking and positive-pair-only differentiable consistency. Measure end-to-end scoring latency; do not count only encoder FLOPs. Abandon the compositional method if a cheaper score attains the same result.

**“Selecting the best test epoch or correcting group construction selectively makes every result suspect.”**

**Assessment:** valid.

**Modification:** use the true dev set, preserve the established test candidate construction, and apply every correction to all arms. Do not use test-set observations to select the support threshold or transport mass.

## 15. Novelty Verification

The post-design check searched combinations of sign-language retrieval, local hard negatives, shared/positive support, counterfactual visual evidence, partial optimal transport, overlapping observations, sampling consistency and capacity-constrained cross-modal matching. Primary papers and available implementation paths—not search snippets—determine the claims below. Publication/availability cutoff is September 12, 2026.

| Proposed or initially considered claim | Closest verified work | Conflict and decision |
|---|---|---|
| “Use visual rather than semantic hard negatives” | [SAN, ACL 2026](https://aclanthology.org/2026.acl-long.1302/) | Already central SAN contribution. Borrow as a shared control/mining principle; not new. |
| “Use local HN to preserve global retrieval” | [FSC-CLIP, EMNLP 2024](https://arxiv.org/abs/2410.05210) | Extremely similar initial concept. Reject generic novelty and require its official implementation as a control. |
| “Stop gradient through local attention” | [FSC-CLIP source](https://github.com/ytaek-oh/fsc-clip/blob/604015db3f009a8f7485f1fb7e21d8343c67664a/src/training/losses/loss.py) | Already implemented by detach(). No novelty claim. |
| “Bound negative pressure / adapt hardness” | [CE-CLIP](https://arxiv.org/abs/2306.08832), [GARE](https://arxiv.org/abs/2505.12499), [SCL-SLT](https://aclanthology.org/2026.acl-long.2116/) | Existing ranking, gradient and curriculum directions. Hinge/reliability are supporting controls. |
| “Token-level sign contrast” | [ConSLT](https://arxiv.org/abs/2204.04916), TACo, C²RL; MCL-SLT final full text **UNVERIFIED** | Broad claim rejected. Method 1 must demonstrate the narrower shared-evidence intervention. |
| “Partial frame/token matching with null” | [READ-PVLA](https://github.com/nguyentthong/READ), [VTaMo](https://arxiv.org/abs/2607.09126) | Already published in adjacent/sign tasks. Generic Method 2 concept rejected. |
| “Two dustbins and prescribed real transport mass” | [DualAnchor, July 2026](https://arxiv.org/abs/2607.27614), equations 13–16 | Exact close construction already exists. The report credits this overlap explicitly; it is not a novel solver. |
| “Partial OT plus teacher regularization” | DualAnchor; ALBEF; local-alignment work including [SLAP, August 2026](https://arxiv.org/abs/2608.08840) | Too generic to support novelty. Retain only the sampling/evidence hypothesis and correspondence-specific adaptation. |
| “Shared positive/negative visual support for changed-span sign-aware contrasts” | FSC-CLIP independently aligns candidates; SAN changes negative distribution | No exact implementation of this narrowed combination was located in inspected primary sources. This is **not proof of firstness**; novelty remains conditional on comparison and unresolved close full text. |
| “Physical-coverage capacities plus time-projected correspondence consistency for SLRet” | Weighted OT / temporal consistency; VTaMo and DualAnchor | No exact SLRet implementation located in inspected sources. General ingredients are established, so a top-tier claim needs a strong diagnosed failure and isolation against simpler weighting. |

The second search also confirmed that a 2026 date alone does not imply a new sentence-retrieval system: sign production, reverse dictionaries, visual-to-visual matching and SLT decoder papers must be separated from full-pool text↔video retrieval. Conversely, a paper primarily about translation can defeat a proposed alignment novelty claim.

**Remaining novelty uncertainty:** full CMCM text, final MCL-SLT details and some very recent non-indexed material were not verified. Therefore the report supports two concrete research hypotheses, not an unconditional novelty certification. The claims have already been narrowed rather than hiding close work.

## 16. Implementation Blueprint

### Baseline and modifications

Baseline: [xua222/UPRet, revision 046366227417e1d8ec14145965403462df345984](https://github.com/xua222/UPRet/tree/046366227417e1d8ec14145965403462df345984). Use existing CiCo feature files and CLIP initialization with explicit provenance. Start a separate experiment implementation; the following files/modules are a blueprint, not code already written.

| Existing or proposed location | Required modification | Important implementation detail |
|---|---|---|
| modules/modeling.py::forward | Optionally return contextual video features and masks alongside the unchanged base loss | Share the exact features used by retrieval; a loss attached only to an independent new head cannot improve the retriever. |
| modules/modeling.py::get_text_feat, get_video_feat, get_text_video_feat | Expose a stable reference-encoding interface | Preserve normalization, feature orientation and padding semantics; separate train-mode dropout from deterministic reference inference. |
| modules/modeling.py::flip_similarity_softmax | Common padding correction in every controlled arm | Mask the partner axis before softmax. Do not count the correction as Method 1's contribution. |
| Proposed modules/shared_support_loss.py | Span pooling, common support, margin contraction, optional gate | Inputs \(X[B,L,d]\), support \(a[B,K,E,L]\), fixed span differences \(D[B,K,E,d]\), weights \(c[B,K,E]\); output scalar loss and detached diagnostics. |
| Proposed preprocess/build_visual_negative_table.py | Training-only miner and cache | Output word candidates, occurrence statistics, source pair IDs, tokenizer/config/teacher hashes; no dev/test examples in prototype fitting. |
| Proposed preprocess/map_edit_spans.py | Character/word edit to complete BPE spans | Handle insertion/deletion length changes, repeated words, punctuation, translated captions and truncation. Never assume token position is unchanged after an edit. |
| dataloaders/dataloader_H2_retrieval_train.py and analogous PH/CSL loaders | Return negative IDs/spans and original clip indices | Preserve existing video-group sampling. Do not oversample easy duplicated weather templates unless it is a declared control. |
| main_task_retrieval.py::train_epoch | Load frozen reference, obtain/cached support, combine losses and log metrics | Teacher parameters require no gradient/optimizer state. New auxiliary branch only scores within-example negatives. |
| main_task_retrieval.py::eval_epoch | Shared evaluator repair, dev/test separation | Correct malformed mask indexing for grouped evaluation; retain official group aggregation and positive definition. |
| metrics.py | Validate ranking/tie behavior and document any repair | No silent candidate-pool or relevance changes. Preserve an explicitly named compatibility mode if historical exact behavior must be reproduced. |
| Experiment config / launch scripts | Remove author-specific paths; record all resources and hyperparameters | Include feature hashes, split/group hashes, language, FPS, windows/stride, CLIP source, seed, temperature, batch, LR/schedule and selection metric. |

The inspected main module exposes get_text_feat, get_video_feat and get_text_video_feat; the public API does not already return a ready-made shared-support loss. The wrapper below is intentionally pseudocode for the proposed integration.

### High-level PyTorch-style pseudocode

~~~python
# Proposed wrappers around UPRet, not existing repository APIs.
# The reference is our own reproduced baseline, never a SEDS checkpoint.
reference = frozen_eval_copy(reproduced_baseline)
student = copy_for_finetuning(reproduced_baseline)

for batch in train_loader:
    # All arms receive the same features, captions, edits and optimizer budget.
    base_loss, x, video_valid = student.base_forward_with_features(batch)
    x = F.normalize(x, dim=-1)              # [B, L, d]

    with torch.no_grad():
        x_ref = reference.video_tokens(batch)
        x_ref = F.normalize(x_ref, dim=-1)
        q_pos, q_neg, edit_valid = cached_reference_span_vectors(batch)
        # q_pos, q_neg: [B, K, E, d]; complete lexical spans.
        q_pos = F.normalize(q_pos, dim=-1)
        q_neg = F.normalize(q_neg, dim=-1)

        support_logits = torch.einsum("bld,bhed->bhel", x_ref, q_pos)
        support_logits = support_logits / support_temperature
        support_logits = support_logits.masked_fill(
            ~video_valid[:, None, None, :], float("-inf")
        )
        support = support_logits.softmax(dim=-1)
        confidence = edit_valid.float()
        if use_stability_gate:
            confidence *= reference_support_agreement(batch, support)
        direction = q_pos - q_neg

    # Same support for both alternatives; no separate negative alignment.
    delta = torch.einsum("bhel,bld,bhed->bhe", support, x, direction)
    shared_loss = (confidence * F.relu(margin - delta)).sum()
    shared_loss /= confidence.sum().clamp_min(1.0)

    loss = base_loss + shared_weight * shared_loss
    optimizer.zero_grad()
    loss.backward()
    clip_shared_trainable_gradients_if_configured(student)
    optimizer.step()
    scheduler_step_as_in_reproduced_baseline()
~~~

Handle an entirely invalid video before softmax; all-\(-\infty\) logits produce NaNs. For batches with no eligible edits, return a differentiable zero auxiliary loss and still run the ordinary retrieval update. Support agreement must use the same temporal coordinate system across views.

### What changes in data, sampling and evaluation

There is no new data split or annotation requirement. The dataloader gains cached negative text/spans, confidence metadata and feature indices. Initial main runs retain the established sampling and cap; additional ordered-sampling diagnostics are explicitly separate. Negative construction uses training data only.

There is no Method 1 inference change. Evaluation repairs and duplicate-label policy are applied to the baseline first and shared by all arms. For exact repeated text that appears under distinct official IDs, prefer excluding it from hard-negative construction; any multi-positive training reformulation is a separate common control, not an undocumented relevance change.

### Memory-sensitive operations

Avoid constructing \([B^2K,L,M']\) affinities for negatives belonging only to their own video. Cache frozen reference text span vectors, not entire gradient-bearing negative-text sequences. Reference video activations need no autograd state. Dense base CLCL still dominates memory; chunking it is an engineering change shared with controls.

For Method 2, new modules would be coverage_masses.py, partial_alignment.py and time_projection.py. Reuse SuperGlue's log solver with custom marginals; check row/column residuals, real transported mass and forbidden-edge mass. A custom envelope-gradient or recomputation path must be validated against a small fully differentiable solver before scaling.

### Debugging and required verification gates

| Risk | Meaningful check |
|---|---|
| Mask convention reversal | Compare lengths/counts from data to valid-token masks before every softmax; UPRet has multiple mask conversions. |
| Wrong grouping | Match independently counted caption/video/group totals; verify group max scoring on a small hand-calculated matrix. |
| Span misalignment | Round-trip edited character spans through the tokenizer; skip truncated/incomplete edits. |
| No useful gradient reaches retrieval | Confirm auxiliary gradients reach the shared video projection/Transformer, while frozen teacher and reference text receive none. |
| Frozen-feature ceiling | Compare random versus teacher-selected support and inspect whether the baseline features distinguish natural visual confusions at all. |
| Teacher/student drift | Monitor span margins, ordinary validation retrieval and similarity to reference features; do not add a new distillation loss by default. |
| Spurious teacher confidence | Test support agreement and feature-removal effects; softmax sharpness alone is insufficient. |
| Duplicate false negatives | Log collisions by exact normalized caption and group ID; do not hide them inside random fallback. |
| Method 2 numerical mismatch | Compare analytic/envelope and autograd gradients on tiny finite problems; verify mass feasibility and duplicate-splitting behavior. |
| Wrong checkpoint selection | Test loader is invoked only after dev-selected configuration is locked; record the actual dev manifest. |

No GPU training or end-to-end reproduction was performed for this report. Runtime, convergence, numerical recall and final superiority remain experimental questions. The blueprint specifies how to test them without silently changing the task.

## 17. Final Verdict

### A. If there were only one chance to build a method for outperforming current SLRet systems, what would I build?

**Shared-support sign contrast, beginning with the independently reproduced UPRet feature regime and testing against the strongest matched SAN/FSC-CLIP controls.** Keep the inference model unchanged. Pursue a broad SOTA claim only after comparison or transfer against the stronger SEDS/C²RL representation frontier; do not use any pretrained SEDS parameters.

### B. Why this method rather than the alternatives?

It targets a demonstrated fine-grained/full-pool tradeoff, has a small intervention with an identifiable gradient path, uses existing paired data, and can fail decisively in an inexpensive controlled experiment. Adding pose, bigger language models, broad probabilistic machinery or several transport/temporal modules would make contribution attribution harder before this hypothesis has been tested.

The choice is not based on a claim that the shared-support operation is radically new. Its value depends on demonstrating a specific visual-evidence failure and fixing it beyond close prior art.

### C. What is the single most important experiment?

**Hold the baseline, negative captions, edited spans, teacher, loss, optimization, sampling and evaluation constant; change only independent versus shared positive/negative visual support.** Measure ordinary full-pool retrieval in both directions, with the existing fine-grained test as a separate diagnostic when compatible. The shared-support arm must beat the best tuned FSC-style/span control, not just a plain baseline.

### D. What result would cause abandonment?

Shared support provides no reproducible gain over the independent-support control, gains occur only on generated stress captions, random support works equally well, or ordinary V2T deteriorates. Any of these undermines the central causal argument. Better mask handling, duplicated-caption treatment or extra training time alone is not success for the method.

### E. Confidence of outperforming the strongest fairly comparable baseline

**Moderate**, conditional on establishing usable reference support and beating the strongest resource-matched controls. **Low** for outperforming the entire current published system frontier across all datasets; CMCM's verified existence but unavailable numerical details and unresolved journal-version/protocol information prevent a stronger field-wide statement.

Method 2 remains a concrete backup with **Low** confidence of exceeding the strongest matched control before its sampling-error hypothesis is tested. Neither proposal has measured SOTA performance. Both have explicit failure tests and comply with the no-new-dataset, no-new-benchmark, no-extra-manual-training-annotation and no-pretrained-SEDS constraints.

### Principal source register

The linked tables throughout give source-specific evidence and pinned code locations. The principal publications and their roles are:

| Source | Publication / version used | Role |
|---|---|---|
| [Sign Language Video Retrieval with Free-Form Textual Queries](https://arxiv.org/abs/2201.02495) | Duarte et al., CVPR 2022; arXiv paper/appendix | Task, original retrieval, preprocessing and feature effects |
| [CiCo](https://arxiv.org/abs/2303.12793) | Cheng et al., CVPR 2023 | Cross-lingual matching, features, language protocol |
| [SEDS](https://arxiv.org/abs/2407.16394) | Jiang et al., ACM MM 2024; [publisher](https://doi.org/10.1145/3664647.3681237) | RGB/pose representation, ablations, frontier tables |
| [UPRet](https://arxiv.org/abs/2405.19689) | Wu et al., ECCV 2024; final ECVA paper and supplement also inspected | Probabilistic training, controlled recalls, sample-OT behavior |
| [C²RL](https://arxiv.org/abs/2408.09949) | Chen et al., arXiv v1; [TCSVT 2025 identity](https://doi.org/10.1109/TCSVT.2025.3553052) | Paired-text pretraining and verified arXiv retrieval tables |
| [Semantic Hardness Is Not Visual Hardness](https://aclanthology.org/2026.acl-long.1302/) | Lee et al., ACL 2026 | Fine-grained negative supervision and ordinary-retrieval tradeoff |
| [Scaling up Multimodal Pre-training](https://arxiv.org/abs/2408.08544) | 2024 arXiv tables; [TPAMI 2025 identity](https://doi.org/10.1109/TPAMI.2025.3599313) | External scale and manual/non-manual confounds |
| [CSLR²: A Tale of Two Languages](https://arxiv.org/abs/2405.10266) | 2024 preprint | BOBSL retrieval, sign-level learning and hard negatives |
| [SignCLIP](https://aclanthology.org/2024.emnlp-main.518/) | EMNLP 2024 | Multilingual pose/text representation and dictionary scope |
| [SignRep](https://arxiv.org/abs/2503.08529) | ICCV 2025 | Sign-specific visual representation |
| [SCL-SLT](https://aclanthology.org/2026.acl-long.2116/) | ACL 2026 | Adaptive negative selection and duplicate semantics |
| [SEA](https://aclanthology.org/2026.acl-long.1401/) | ACL 2026 | Segmentation/subtitle alignment |
| [VTaMo](https://arxiv.org/abs/2607.09126) | July 2026 preprint | Close token-OT/null alignment prior and source audit |
| [DualAnchor](https://arxiv.org/abs/2607.27614) | July 2026 preprint | Exact close partial-mass/dustbin construction |
| [CMCM](https://doi.org/10.1016/j.cviu.2025.104631) | CVIU 2026 | Relevant unresolved newer retrieval system |
| [FSC-CLIP](https://arxiv.org/abs/2410.05210) | Oh et al., EMNLP 2024 | Closest local-negative control, official code inspected |
| [TACo](https://arxiv.org/abs/2108.09980) and [FineCo](https://aclanthology.org/2022.aacl-main.53/) | ICCV 2021; AACL 2022 | Token/frame contrast precedents |
| [CE-CLIP](https://arxiv.org/abs/2306.08832) and [HBI](https://arxiv.org/abs/2303.14369) | CVPR 2024; CVPR 2023 | Negative margins and hierarchical interaction |
| [ALBEF](https://arxiv.org/abs/2107.07651) and [SuperGlue](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) | NeurIPS 2021; CVPR 2020 | Code-backed teacher update and numerical matching solver |
| [PCME++](https://arxiv.org/abs/2305.18171) and [ProLIP](https://arxiv.org/abs/2410.18857) | ICLR 2024; ICLR 2025 | Probabilistic retrieval alternatives and code contracts |
| [Drop-DTW](https://arxiv.org/abs/2108.11996), [READ repository/paper](https://github.com/nguyentthong/READ) | NeurIPS 2021; AAAI 2024 | Structured sequence and partial-alignment precedents |
| [GARE](https://arxiv.org/abs/2505.12499), [SLAP](https://arxiv.org/abs/2608.08840), [SignMatch](https://arxiv.org/abs/2609.01886) | Versions public before the cutoff | Recent overlap and scope checks |

Backward citation tracing covered sign spotting, sign-domain pretraining, gloss-free translation, CLIP and contrastive alignment antecedents. Forward tracing covered the seed successors, later journal versions, 2026 negative/alignment papers, and related representation/visual retrieval work. This source register records verified evidence; a missing result is left missing rather than inferred from a later paper's title or abstract.
