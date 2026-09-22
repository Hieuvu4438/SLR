# Handshape labels: a PH-compatible name is not independent video evidence

2026-09-22. academic-research-suite, fact-check mode, inline. AI-assisted;
no human annotation, independent DGS review, model fitting or TEST access.

## Question and outcome

Can an existing released handshape resource supply the missing independent
visual-contrast evidence for Q04/Q17 without using MY DGS or collecting labels?
This is an evidence-admission check, not a new method or a full literature review.

**Outcome:** PHOENIX14T-HS is a concrete additional source, but its documented
annotation procedure does not establish token-level visible handshape truth.
Do not admit handshape-grounded confuser labels from this source alone. Resource
existence and usefulness as weak supervision are different from the evidentiary
requirement of this diagnostic. This is not a finding that handshape is irrelevant.

## Primary-source findings

[V] [Zhang and Duh, Findings of EMNLP2023, §3.3](https://aclanthology.org/2023.findings-emnlp.198.pdf)
describe gloss-level labels for both hands, derived from SignWriting plus manual
completion. They explicitly caution that dictionary variants were randomly
selected and can mismatch the video; co-articulation was not annotated. The
annotator reports no formal DGS training. Their single/dual-encoder handshape
auxiliaries and joint CTC/frame-CE objectives are also relevant prior art.

[V] The [author repository](https://github.com/Este1le/slr_handshape) links the
dataset and pretrained models and describes a TwoStreamSLT-based implementation.
Its dataset landing link returned404 in this browser on2026-09-22. No dataset
payload or ID join was inspected. The repository displays an MIT license badge;
the license-file fetch failed. This does not establish terms for the separately
hosted data, upstream videos or pretrained weights.

[V] [Sarkar et al., arXiv:2608.10588v1](https://arxiv.org/html/2608.10588v1)
is an adjacent isolated-handshape benchmark, not PH sentence relevance judgments.
Its Data availability section promises access after publication by request,
subject to institutional conditions. The article's CC BY notice is not evidence
of an unrestricted dataset release. No request was sent.

## Consequence for current research

[I] This removes a tempting shortcut: finding a dataset that names PHOENIX14T
does not make its handshape labels an independent explanation of our errors.
Joining by gloss could propagate a dictionary mismatch into the supposed ground
truth. Comparing model predictions against that join would conflate realization
differences, annotation noise and representation failure. It would not establish
which explanation is responsible, even with a successful ID join.

No teacher conversion, fuzzy gloss match, synthetic minimal pair, new positive,
or handshape auxiliary is introduced. P3's generic relational-head rejection
and P1/C27's scientific admission limits remain intact. The existing paper is
a collision target for a future auxiliary, not proof all possible handshape
mechanisms are equivalent or that its SLR result transfers to retrieval.

This check does not depend on MY DGS permission. That question remains local to
the MY DGS branch; do not repeatedly ask it as a prerequisite for all research.
Do not revisit this resource simply to find another mirror: a live payload alone
would not resolve the annotation-validity issue. Revisit only for a genuinely
different question consistent with weak labels, or evidence of independently
curated token-level annotations with documented provenance. Neither is currently
a supported pilot. Primary method and controlled retrieval gains remain missing.

## Verification limits and search record

- Source grade: publisher paper, peer-reviewed descriptive/method study (LevelVI
  for the resource description); repository is direct implementation documentation;
  arXiv item is a preprint. Strong for the stated procedures, not reproduced gains.
- Read scope:2023 PDF abstract, introductory architecture description, §2.2–3.3
  including limitations and Table1; repository README. Not a full code audit or
  performance reproduction.2026 HTML abstract, related-resource tables and data
  availability only; not a full-paper review.
- The2026 reference led to the2023 source; shared citation is not independent
  validation. Author-produced resources have intellectual/institutional interests.
  No independent annotation-quality review or bibliographic API check was run.
  Journal-predation screening is not applicable to the ACL publisher record or
  institutional code release; peer review is not claimed for the preprint.
- Queries: `German sign language phonological annotations dataset license
  HamNoSys sign2mint`, `DGS lexical database handshape movement annotations open
  dataset license`, exact2026 title, `PHOENIX14T-HS dataset github`.
  Sign2MINT portal returned no readable text; no license conclusion was drawn.
- Local title/resource keyword search across `docs/` and `research/` Markdown
  found no dedicated PHOENIX14T-HS record. This is not a first-discovery claim
  across every file or all earlier conversations.
