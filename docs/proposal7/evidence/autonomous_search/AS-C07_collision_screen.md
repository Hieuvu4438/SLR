# AS-C07 targeted collision screen

## Material Passport

- Origin Skill: academic-research-suite; inline bibliography, verification, synthesis.
- Date / last searched: 2026-09-15.
- Status: ANALYZED; search-bounded mechanism screen, not exhaustive novelty clearance.
- AI disclosure: source selection and analysis by one Codex model; no human-read
  attestation, independent reviewers, or reproduced external results claimed.
- Sources read through browser: not acquired as local original artifacts. No
  `source_verified_against_original` passport flag is asserted.

### Claim-intent manifest (before synthesis)

```json
{"manifest_version":"1.0","manifest_id":"M-AS-C07-collision-20260915","emitted_by":"synthesis_agent","emitted_at":"2026-09-15","claims":[{"claim_id":"C1","claim_text":"Query-bank normalization is established adjacent prior.","intended_evidence_kind":"definitional","planned_refs":["qb2022"]},{"claim_id":"C2","claim_text":"Teacher-derived batch transport targets for contrastive learning are established prior; the present candidate has not supplied a materially distinct mechanism.","intended_evidence_kind":"theoretical","planned_refs":["otter2022","teachtext2021","cprd2024","sclip2023"]}],"manifest_negative_constraints":[{"constraint_id":"N1","rule":"Do not assert exact implementation equivalence, exhaustive absence of novelty, or external gains on PH."}]}
```

## Reproducible search scope

RQ: Does replacing independent teacher scores with cohort-constrained targets,
then distilling into an independent scorer, constitute a new causal mechanism?
English, all years through search date; official proceedings, arXiv, author
projects. Include cross-modal ranking/distillation, transport pseudo-labels and
query-bank inference. Exclude secondary summaries as evidence; no broad SLRet
survey, programmatic bibliographic API gate, or PRISMA completeness claim.
Deduplication by title; search-engine total hit counts not available/reported.

Queries used (followed by exact-title and official-source retrieval):

- `"Cross Modal Retrieval with Querybank Normalisation"`
- `retrieval distillation optimal transport teacher student listwise ranking cross modal`
- `"sign language retrieval" "distillation"`
- `site:openaccess.thecvf.com retrieval "distillation" "ranking"`
- `site:arxiv.org "retrieval" "optimal transport" "distillation"`
- `site:openaccess.thecvf.com "sign language retrieval" distillation`
- `"retrieval" "Sinkhorn" "distillation" teacher`
- `"retrieval" "transductive" "distillation"`
- `"sign language retrieval" distillation teacher knowledge`
- `"contrastive" "distillation" "assignment" "retrieval"`
- `"S-CLIP" "optimal transport"`
- `"TeachText" crossmodal generalized distillation`
- `"How to Make Cross Encoder a Good Teacher" arxiv`
- `"contrastive" "optimal transport" "soft labels" teacher retrieval`
- `"Data Efficient Language-supervised Zero-shot Recognition with Optimal Transport Distillation"`

## Included sources and read scope

| Key; bibliographic identity | Primary source; actual reading scope | Mechanism relevant to this screen |
|---|---|---|
| qb2022; Bogolin, S.-V., Croitoru, I., Jin, H., Liu, Y., & Albanie, S. (2022). *Cross Modal Retrieval with Querybank Normalisation*. CVPR. | [Proceedings PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bogolin_Cross_Modal_Retrieval_With_Querybank_Normalisation_CVPR_2022_paper.pdf), indexed abstract/introduction; direct HTML fetch failed403. | Probe-bank similarities adjust hubness without concurrent evaluation-query access. Adjacent prior for A; exact A reduction is our separate algebra, not a theorem attributed to this paper. |
| otter2022; Wu, B., Cheng, R., Zhang, P., Gao, T., Vajda, P., & Gonzalez, J. E. (2022). *Data Efficient Language-Supervised Zero-Shot Recognition with Optimal Transport Distillation*. ICLR. | [Official conference record](https://iclr.cc/virtual/2022/poster/6582); [arXiv v3 HTML](https://arxiv.org/html/2112.09445v3), abstract and §§2–3.3, equations1–6. HTML is the December2023 revision, not silently the original proceedings text. | Teacher pair similarities feed balanced entropic transport; matching probabilities become contrastive soft targets. Equal row/column mass and entropy control joint assignment. Diagonal supervision is mixed separately. |
| teachtext2021; Croitoru, I., Bogolin, S.-V., Leordeanu, M., Jin, H., Zisserman, A., Albanie, S., & Liu, Y. (2021). *TeachText: CrossModal Generalized Distillation for Text-Video Retrieval*. ICCV. | [Author project](https://www.robots.ox.ac.uk/~vgg/research/teachtext/), abstract and Method overview; indexed proceedings abstract. Direct proceedings PDF fetch failed403. | Aggregated batch similarity matrices from frozen teachers supervise a student alongside retrieval loss; no added inference overhead. |
| cprd2024; Chen, Y., Ma, Z., Zhang, Z., Qi, Z., Yuan, C., Li, B., Pu, J., Shan, Y., Qi, X., & Hu, W. (2024). *How to Make Cross Encoder a Good Teacher for Efficient Image-Text Retrieval?* CVPR,26994–27003. | [Proceedings metadata/abstract](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_How_to_Make_Cross_Encoder_a_Good_Teacher_for_Efficient_CVPR_2024_paper.html); [arXiv HTML](https://arxiv.org/html/2407.07479v1), abstract. PDF direct fetch failed403. | Relative hard-negative order, rather than raw teacher scale, is distilled with a contrastive objective. Not cohort matching itself. |
| sclip2023; Mo, S., Kim, M., Lee, K., & Shin, J. (2023). *S-CLIP: Semi-supervised Vision-Language Learning using Few Specialist Captions*. NeurIPS. | [Proceedings PDF](https://papers.neurips.cc/paper_files/paper/2023/file/c06f788963f0ce069f5b2dbf83fe7822-Paper-Conference.pdf), abstract/introduction/related work via browser. | Transport from unpaired to paired images produces caption-distribution targets; additional keyword objective. Different supervision setting from B, useful adjacent rather than exact collision. |

Source assessment: five computational experimental papers (LevelIII design);
official conference identities confirmed, suitable primary evidence of published
mechanisms. External effect sizes, comprehensive methods quality, COI/retraction
status, ORCID, commercial venue indexes not audited; no overall A-grade inference
or independent performance verification. The official author repository also
confirms OTTER identity; its abbreviated README bibliography omits authors, so
the paper title/author line above takes precedence. OpenReview returned browser
verification; arXiv PDF exceeded browser size; accessible HTML supplied equations.

DISTRIBUTIONAL_SKEW_ADVISORY: computational methods5/5. Intentional given RQ;
not evidence of comprehensive sign-language or linguistic-validity coverage.

## Synthesis and decision

The fixed-bank inference route is adjacent to [Bogolin et al. (2022)](https://arxiv.org/abs/2112.12777)
<!--ref:qb2022--><!--anchor:section:Abstract-->. Its fatal reduction remains
AS-C07-ALGEBRA, not a literature-only dismissal.

Candidate B's generic structure is already represented by [Wu et al. (2022)](https://arxiv.org/html/2112.09445v3)
<!--ref:otter2022--><!--anchor:section:3.3-->: teacher similarities → joint
balanced matching → soft contrastive supervision. The present B differs in fold
provenance, application, teacher and potentially hard/soft target construction.
These are real implementation differences, but B has supplied no distinct causal
operator or argument beyond them. **Reject this specification for proposal
selection; do not spend fold-training compute or rename it.** This is a
search-bounded conceptual collision, NOT a proof all cohort-to-independent
learning is exhausted. In particular, the prior's off-diagonal masking and label
mixture mean it is not algebraically identical to our full Hungarian diagnostic.

[Croitoru et al. (2021)](https://www.robots.ox.ac.uk/~vgg/research/teachtext/)
<!--ref:teachtext2021--><!--anchor:section:Method%20overview--> and
[Chen et al. (2024)](https://arxiv.org/html/2407.07479v1)
<!--ref:cprd2024--><!--anchor:section:Abstract--> additionally prevent novelty
claims based merely on similarity/ranking distillation or removing the teacher
at inference. [Mo et al. (2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html)
<!--ref:sclip2023--><!--anchor:section:Abstract--> shows another transport-target
construction, not the same cohort teacher or paired-data regime.

Cross-paper tension inventory: OTTER/TeachText (shared teacher-supervision
question) and OTTER/S-CLIP (shared transport-target question) both have
`pair_assessment: no_material_conflict`, `resolution_status: not_applicable`,
`scholar_confirmation: pending`. Pointers: OTTER§3.3, TeachText Method overview,
S-CLIP Abstract. They supply different target operators/settings, not opposing
PH findings. Coverage: five papers, two assessed pairs; not exhaustive pairwise
contradiction detection.

Unresolved gaps: clean out-of-fold transfer on PH has not been measured;
independent-query gains from capacity-informed supervision are not established.
Neither gap is itself novelty. No sign-language semantic equivalence or dataset
label correction is inferred. Resume another open causal layer: Q21 exact text
augmentation effects, measured without modifying the baseline or official task.
