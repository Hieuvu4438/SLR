# Targeted decision literature — Goal V4

## Post-C19 mechanism screen — 2026-09-21

The next candidate screen focuses on train-only modality exposure, because the
native RGB branch is much stronger than pose on the fixed DEV readout while C19
showed that adding another inference-time exchange path did not help. Neverova et
al., ModDrop, TPAMI2016, randomly drops modality channels during gradual fusion to
preserve modality-specific representations ([primary paper](https://doi.org/10.1109/TPAMI.2015.2461544)).
This supports the mechanism class, not efficacy on SLRet.

Counterevidence is explicit: Dai et al., CVPR2024, report that ordinary modality
dropout can improve missing-frame robustness yet hurt complete-input AVSR through
dropout-induced modality bias ([CVF primary paper](https://openaccess.thecvf.com/content/CVPR2024/html/Dai_A_Study_of_Dropout-Induced_Modality_Bias_on_Robustness_to_Missing_CVPR_2024_paper.html)).
Therefore a candidate, if registered, must be low-strength, training-only and
anneal the intervention to zero so the late phase trains on complete inputs. This
is a prospective synthesis and not a novelty clearance or measured gain.

## Post-C20 3D representation screen — 2026-09-21

C09's raw H4W camera-axis representation explicitly left view rotation unresolved.
The next prepared contrast combines three established observations: learned view
regulation can reduce 3D-skeleton viewpoint variation
([ICCV2017 primary](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.html));
joint and bone-direction representations can be complementary
([CVPR2019 primary](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html));
and short-term 3D hand pose dynamics can support longer action aggregation
([CVPR2023 primary](https://openaccess.thecvf.com/content/CVPR2023/html/Wen_Hierarchical_Temporal_Transformer_for_3D_Hand_Pose_Estimation_and_Action_CVPR_2023_paper.html)).

C21 does not import those architectures. It deterministically canonicalizes H4W
hands/body into anatomical frames and adds adjacent-clip local motion before the
existing small C09 branch. This removes a diagnosed representation mismatch, but
may amplify estimator noise or discard meaningful global orientation. No source
establishes SLRet gain or novelty; both remain prospective.

## C19 decision2026-09-20

Queries: 'Multimodal Bottleneck Transformer NeurIPS 2021 attention bottlenecks
multimodal fusion paper'; 'sign language retrieval cross modal global local
bottleneck fusion 2025 2026'. Recent recognition/translation hits are leads only,
not verified retrieval gains or a novelty clearance. No additional donor download.

Nagrani et al., Attention Bottlenecks for Multimodal Fusion, NeurIPS2021:
[primary paper](https://papers.neurips.cc/paper_files/paper/2021/file/76ba9f564ebbc35b1014ac498fafadd0-Paper.pdf).
Read Sec3.2/3.3 and relevant fusion ablation text online. Borrow compact latent
exchange, independently implemented; no MBT backbone/code/checkpoint imported.
C19 differs in placement and two-pass adapter; it is not faithful MBT reproduction.

[SEDS v1](https://arxiv.org/html/2407.16394v1), CGAF method and fusion ablation:
local fusion is deliberate and unrestricted cross-attention may include irrelevant
distant features. Counterevidence recorded before launch, not ignored. C19 retains
CGAF and tests a restricted additional path. Source analysis confirms contextual
encoders already exist and learned offsets need not stay local. No claim SEDS is
blind to global context. See METHOD_GLOBAL_EXCHANGE_C19.md for exact hypothesis.

2026-09-20; not an exhaustive review or novelty clearance. Primary sources only
support mechanism claims; recent search hits not fully read are not donor evidence.

| Paper/version | Mechanism / read scope | Source and decision |
|---|---|---|
| Decoupled Contrastive Learning, Yeh et al., ECCV2022; arXiv2110.06848v2, 2021-10-23 | Remove positive from contrastive denominator; method Eq1–6, relevant small-batch experiments and A.6 limitations read. Image SSL results do not imply SLRet gains. | [Method](https://arxiv.org/html/2110.06848v2), [ECVA final](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860653.pdf). C17 borrows Eq5 for fused cross-modal logits, no DCLW/extra data. |
| Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup, Gao et al., RepL4NLP2021 | Gradient caching separates encoder activations from contrastive batch size. Abstract/source availability scanned, full method not read yet. | [Primary paper](https://aclanthology.org/2021.repl4nlp-1.31.pdf), [paper-linked code](https://github.com/luyug/GradCache). Alternate lead, not implemented; attribution needs matched exposure/update accounting. |
| AMD: Adaptive Momentum and Decoupled Contrastive Learning Framework for Robust Long-Tail Trajectory Prediction, ICCV2025 | Official title/venue/abstract scanned only; a different trajectory problem, not SLRet transfer evidence. | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Rao_AMD_Adaptive_Momentum_and_Decoupled_Contrastive_Learning_Framework_for_Robust_ICCV_2025_paper.html). Do not import momentum/trajectory modules merely for recency. |

Queries: 'Scaling Deep Contrastive Learning Batch Size GradCache 2021';
'sign language retrieval 2025 negative contrastive learning';
'Decoupled Contrastive Learning 2022 Yeh'; '2025 contrastive learning negative
positive coupling retrieval'; 'Decoupled Contrastive Learning official github Yeh'.
Recent2024–2025 and foundational sources searched; bounded search did not identify
a better immediately compatible donor. No claim current SOTA/novelty is verified.
C16 SignRep/RKD provenance and read scope stay in their existing method cards.
Implementation decision: C17 loss-only contrast first; no broad repo download.
