# Focused literature map — multi-vector and dataset-first SLRet

Search date: 2026-09-21. Sources searched: ACL Anthology, CVF/ECVA proceedings,
arXiv primary pages, and official paper repositories. Queries combined sign
language retrieval/alignment/representation with late interaction,
compositionality, segmentation, and dataset bias. This is a targeted decision
review, not a claim of exhaustive systematic-review coverage.

## Evidence that shapes the lead

| Source | Verified mechanism / result used here | Boundary for this project |
|---|---|---|
| [SEDS](https://arxiv.org/html/2407.16394v1) | Sentence SLRet with offline RGB, online pose and contextual fusion; reports PH TEST 76.8/78.7 T2V/V2T R@1. | Incumbent/reference only. CoSign-LI is not attached to its encoders or fusion. |
| [C²RL](https://arxiv.org/html/2408.09949v1) | Joint contrastive and autoregressive representation pretraining; reports strong PH/CSL retrieval under a different resource regime. | Establishes that better representation learning matters; it already occupies the generic joint generation+retrieval space. |
| [VAP, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05894.pdf) | Greedy token-level visual/text matching and max aggregation for SLT pretraining; its retrieval table does not beat CiCo in both directions. | Token alignment alone is not sufficient evidence for retrieval SOTA. |
| [Video-ColBERT, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Reddy_Video-ColBERT_Contextualized_Late_Interaction_for_Text-to-Video_Retrieval_CVPR_2025_paper.html) | MeanMaxSim over spatial and temporal visual tokens; dual sigmoid losses improve generic T2VR over matched variants. | Direct donor for multi-vector scoring and loss, not evidence of SLRet efficacy or novelty by itself. |
| [VideoComp, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_VideoComp_Advancing_Fine-Grained_Compositional_and_Temporal_Alignment_in_Video-Text_Models_CVPR_2025_paper.html) | Shows ordinary video-text benchmarks can miss compositional/temporal failures and evaluates targeted disruptions separately. | Supports a diagnostic slice/stress test, not changed official relevance or synthetic-negative training here. |
| [SAN, ACL 2026](https://aclanthology.org/2026.acl-long.1302/) | Sign-aware visual hard negatives strongly improve a separate 40-negative fine-grained test; its CiCo coarse R@1 decreases in both directions in the read v1 table. | Hard-negative mining remains closed. Fine-grained and full-gallery scores stay separate. |
| [SEA, ACL 2026](https://aclanthology.org/2026.acl-long.1401/) | Segments continuous signing into sign units, embeds them, and aligns subtitles with dynamic programming across four datasets. | Supports sign-unit structure. Its alignment task and dynamic program are not imported into the first retrieval model. |
| [VTaMo, arXiv 2026](https://arxiv.org/html/2607.09126) | Uses partial, non-monotonic token/frame alignment with a null token plus global and contrastive objectives for SLT; official repo currently lacks released checkpoints/evaluation code. | Eliminates “add OT alignment” as a novelty claim; transport/dustbin methods are also user-closed locally. |
| [SignSeek, arXiv v1 2026](https://arxiv.org/html/2609.03695) | Pose-based articulator pretraining across 266K samples yields strong cross-corpus isolated-sign retrieval and transfers to alignment. | Promising alternative representation, but isolated dictionary retrieval differs from sentence SLRet and no public code/checkpoint was found in the screened record. |
| [Sign-language dataset survey, ACL 2026](https://aclanthology.org/2026.acl-long.1928/) | Catalogues 120 resources and identifies annotation granularity, modality imbalance, and signer bias as recurring benchmark concerns. | Motivates explicit dataset cards/slices; does not validate our locally measured overlap. |
| [TokenBinder, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Zhang_TokenBinder_Text-Video_Retrieval_with_One-to-Many_Alignment_Paradigm_WACV_2025_paper.html) | Coarse-to-fine candidate comparison improves generic T2VR. | Candidate-aware reranking is not the lead because it changes cost and overlaps locally closed reranker/residual families. |

## Synthesis

Three evidence streams converge. First, SLRet performance depends strongly on
representation quality, but current leading sentence systems mostly emit pooled
or globally contextualized retrieval scores. Second, generic video retrieval has
strong evidence for retaining token banks and using late interaction. Third,
recent sign work shows that continuous signing has recoverable local units and
articulator structure, while our annotations show that eval captions are usually
new arrangements of familiar lexical material.

The open, testable gap is therefore not “another alignment loss.” It is whether a
retrieval model should preserve multiple contextual sign-window vectors through
the scoring boundary rather than distilling the video to one sentence vector.
The nearest collision is VAP's token MaxSim; CoSign-LI differs in task, direct
full-gallery retrieval objective, two pretrained sign-window levels, matched
pooled control, and bidirectional official evaluation. This difference supports
an experiment, not yet a novelty claim.

## Contradictions and unresolved risks

- SAN shows a targeted fine-grained gain can coexist with worse coarse R@1;
  therefore official T2V and V2T guardrails are mandatory.
- VAP shows token matching is not automatically superior for retrieval.
- SignRep direct auxiliary transfer in C16 did not beat its matched continuation;
  CoSign-LI must demonstrate that using the representation directly at inference
  is materially different, not merely more parameters.
- Public-pretraining scale may dominate architecture. The pooled control must use
  exactly the same SignRep and text inputs.
- The literature set is concentrated in computational benchmark papers and recent
  2024–2026 work. This is appropriate to the fast-moving RQ but leaves linguistic
  and human-evaluation evidence underrepresented.
