# Post-C42: discourse context does not yet define an admissible method

## Material Passport

academic-research-suite / deep-research / three-way-scan, synthesis; 2026-09-15.
ANALYZED, AI-assisted inline phases. No external reviewer or human validation.
[Claim intent](POST_C42_discourse_claim_intent.json) was emitted before this text.
[Primary-source matrix and reading limits](POST_C42_discourse_sources.md),
[locked metadata protocol](POST_C42_discourse_boundary_protocol.md).

## Decision

Do not promote generic neighboring-context conditioning to AS-C43. The metadata
prerequisite is unmet in the inspected inputs, and the literature already contains
preceding-context models. No local retrieval or linguistic failure signature was
measured here. This is a bounded pre-candidate rejection, not evidence that
discourse never matters, that PH is intrinsically ambiguous, or that the search is
exhausted. AS-C42 remains completed and inadequate; its residuals are not reused.

## Measured input boundary

[Census JSON](POST_C42-DISCOURSE-METADATA.json), SHA256
`3834eb7c776279176564d5bf54aa11f6ab3c27c725b552e2d18cd4d238864c2b`.
Code SHA256 `073500bd5a7a079074b895093cb8e721b0991837fbaea6563d826a040e4baf5a`.
Completed with exit 0 in 0.647134 seconds (internal timer), 3 unit tests pass.
Full probe regression suite: 74 tests passed in 1.47 seconds, exit 0.
No GPU/model experiment; 7,619 metadata files read and hashed: four JSONL files
and all 7,615 referenced temporal JSON files.

| Inspected relation | TRAIN (7,096) | DEV (519) |
|---|---:|---:|
| Inferred filename prefixes | 643 | 315 |
| Numeric predecessor in same inspected split | 5,617 | 117 |
| Numeric predecessor in the other inspected split | 312 | 292 |
| Numeric predecessor absent from inspected TRAIN/DEV | 1,167 | 110 |

All DEV prefixes occur in TRAIN. A separate direct-string membership enumeration
agrees exactly with the indexed suffix counts. This validates enumeration only:
neither implementation establishes real temporal or linguistic adjacency.

All temporal files declare `coordinate_system=input_frame` and verification scope
`generated_by_deterministic_elsc_i3d_extractor`. Their source path stems equal
sentence IDs; receptive-field starts begin at zero and intervals stay inside the
decoded clip. Every forensic `source` equals the filename prefix. The complete
top-level schemas contain no independently verified original-recording offset or
discourse-link field. The generating function, `shared/slr_common/features/i3d.py`
`_temporal_metadata`, explicitly constructs these local-frame coordinates.

Therefore these files verify clip extraction, not cross-sentence continuity.
Missing predecessor IDs were not searched in TEST. No claim is made that original
recording linkage is absent everywhere in the workspace or official distribution.
No raw video was opened and no linguistic annotation was inferred from filenames.

## Literature synthesis and tensions

Source effect inventory: the only human prevalence used is Tanzer's own sampled
How2Sign result; model gains are author reports on translation, not local SLRet
measurements; the remaining sources establish task/mechanism existence or resource
description. These scopes are unchanged across sections.

The human baseline and context-conditioned model papers converge on a useful
motivation, but answer different questions. [Tanzer et al. (2024)](https://aclanthology.org/2024.emnlp-main.360/)
<!--ref:tanzer2024--><!--anchor:section:4.1-4.2--> assess understanding under added
information; [Sincan et al. (2023)](https://openaccess.thecvf.com/content/ICCV2023W/ACVR/html/Sincan_Is_Context_all_you_Need_Scaling_Neural_Sign_Language_Translation_ICCVW_2023_paper.html)
<!--ref:sincan2023--><!--anchor:section:Abstract--> and
[Jang et al. (2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Lost_in_Translation_Found_in_Context_Sign_Language_Translation_with_CVPR_2025_paper.html)
<!--ref:jang2025--><!--anchor:section:Abstract--> implement preceding-context
translation. Anchor justification: the inspected human methods/results describe
extra-information conditions, while both model abstracts explicitly identify
preceding context. None of these inspected results measures the cause of our PH
retrieval errors. Translation gains can include predictive language shortcuts;
they do not establish missing sign evidence in a particular retrieval pair.

[Yin et al. (2021)](https://aclanthology.org/2021.emnlp-main.405/)
<!--ref:yin2021--><!--anchor:section:Abstract--> and
[Baltatzis et al. (2026)](https://arxiv.org/abs/2609.02796v1)
<!--ref:baltatzis2026--><!--anchor:section:Abstract--> also make generic spatial
reference tracking an existing research topic, not a novelty claim by itself.
Anchor justification: their abstracts explicitly name coreference tasks and
mechanisms; the second concerns text-to-gloss, not video retrieval.

PH's [official resource description](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
<!--ref:rwthph--><!--anchor:section:Detailed Description--> specifies interpreter-box
frames and 386 forecast editions. Anchor justification: those are explicit
resource statements. Inference: importing background-show descriptions from the
Jang setting is not an equal-input intervention on the documented PH frames. The
643 local TRAIN prefixes must not be casually equated with verified editions.

```yaml
cross_paper_tensions:
  - pair_id: DC-CP-1
    paper_a: tanzer2024
    paper_b: sincan2023
    candidate_basis: shared RQ subtopic
    overlap_topic: what context gains demonstrate
    a_finding: human understanding can require additional discourse information
    a_evidence_pointer: sections 2.2 and 4.1-4.2
    b_finding: preceding-context model improves reported translation metrics
    b_evidence_pointer: Abstract
    pair_assessment: conditional_difference
    resolution_status: resolved_in_synthesis
    resolution_pointer: Literature synthesis and tensions, paragraph 2
    scholar_confirmation: pending
  - pair_id: DC-CP-2
    paper_a: sincan2023
    paper_b: jang2025
    candidate_basis: shared construct/outcome/measure
    overlap_topic: preceding-context translation mechanisms
    a_finding: explicit context encoder for preceding sequences
    a_evidence_pointer: Abstract
    b_finding: previous translations among LLM input cues
    b_evidence_pointer: Abstract
    pair_assessment: no_material_conflict
    resolution_status: not_applicable
    scholar_confirmation: pending
```

Coverage: five papers, two candidate pairs assessed. This is a scoped advisory
comparison, not exhaustive contradiction detection; other pairs remain unchecked.
No scholar confirmation is simulated.

## Internal closure and deployment screen

These are rejected routes, NOT three materially distinct surviving candidates:

- Concatenate previous clips/captions: unverified linkage and extra input; existing
  generic context mechanism. In particular, previous gold captions can leak
  annotation information unavailable to an independently presented query.
- Teach a sentence-only student using a context decoder and protect retrieval:
  collides with the closed RPCA generation/context-preservation family. Changing
  the decoder, teacher or loss does not reopen it.
- Condition scores on inferred source/template: risks the existing nuisance or
  candidate-prior calibration mechanism. Source correlation is not linguistic
  coreference and provides no new causal rationale.

Closure anchors: [negative registry](../../Negative_Results_Registry.md), RPCA and
conceptual-closure rows; [user loop](../../Astra_SLRet_Autonomous_Research_Loop.md),
non-negotiable task/positive/split constraints. The ARS workflow influenced this
decision by separating acquired evidence, author claims and inferred implications,
and requiring a reading-scope-aware collision check before promotion.

## Limits and next search consequence

Two gaps remain: a verified task-compatible context linkage, and a PH-specific
linguistic/retrieval signature attributable to context. Neither gap is filled by
prefix counts or an inadequate residual model. This audit does not authorize
external context acquisition, neighboring-caption inference, new relevance
labels, a changed benchmark or a more expensive context model.

Move to another open mechanism with an explicit decision consequence; first
screen internal closure and existing-input testability. No AS-C43 registration or
GPU launch yet. The global goal stays active. No model was selected, trained or
evaluated in this audit; no B0–B3, three-seed, confidence-interval, second-dataset,
expert-linguistic or method-GO gate is claimed to have passed.
