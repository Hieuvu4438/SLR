# C²RL-derived code found: provenance boundary and dynamic padding

2026-09-16. ANALYZED; AI-assisted academic-research-suite source verification.
This is new source evidence, not a new method or an original-C²RL reproduction.

## What changed

The public [sltbaselines repository](https://github.com/ozgemercanoglu/sltbaselines)
contains a C²RL-derived contrastive kernel. Its README identifies the project as
an independent SLT reimplementation and acknowledges partial code shared by the
C²RL authors. `models/models.py:272` attributes the v2 kernel to that shared
implementation. This establishes an attributable public lead; it does not
establish byte-level author lineage, original retrieval code, or final-paper
recipe parity. The earlier official-retrieval-code gap therefore remains.

Pinned commit: `f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13`, default branch main,
tree `a85db6090781006a295f03326ea920319a44e63f`. GitHub tree response was
untruncated. Only source/config/README text was fetched; the repository's
`outputs/` prediction/reference assets were not opened or downloaded.

## Enabled source path

[Pinned model source](https://github.com/ozgemercanoglu/sltbaselines/blob/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13/models/models.py#L271)
and [training call](https://github.com/ozgemercanoglu/sltbaselines/blob/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13/train_slt.py#L519):
the c2rl branch calls `cross_lingual_similarity_v2`, then adds its contrastive
loss to translation loss. This is enabled, not an unused alternative. The kernel
attenuates padded similarities by1e−5 before softmax rather than excluding them;
outer averages are masked. The
[collator](https://github.com/ozgemercanoglu/sltbaselines/blob/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13/dataloader/datasets.py#L208)
uses batch-dependent padding. Hence width can vary with batch composition.

Reading scope: full v2 kernel; surrounding call/constructor/encoder paths;
training loss branch; collator and selected feature padding; PH stage1 config;
README identity, attribution and recipe sections. Not every file or the full
translation pipeline. No training command was run.

## Executed fixed certificate

[Protocol](C2RL_reimplementation_padding_protocol.md),
[script](../../../../methods/information_probe/c2rl_reimplementation_padding.py),
[result](C2RL-REIMPLEMENTATION-PADDING.json).

The source hash was checked before AST extraction. Exactly four empty `.cuda()`
calls were removed for CPU execution; no scoring expressions changed. Full
upstream modules were never imported. This is not GPU-runtime parity.

For one synthetic video query, caption A has one valid token at cosine.21;
caption B has two valid tokens at cosine.19. Padding vectors are nonzero unit
vectors orthogonal to both videos. Valid representations are identical across
the two widths; logit scale1, source temperature.07.

| Text width | I2T A | I2T B | A−B |
|---|---:|---:|---:|
|2|.200040567|.190000000|+.010040567|
|32|.082566672|.095296186|−.012729514|

One I2T channel ordering reverses. T2I and an independent inner-exclusion
reference are exactly unchanged. This is NOT a claimed full deployed mixed-score
reversal or a retrieval correction; no relevance labels exist for these vectors.
The source contrastive loss changes .726412546→.728690144; these values measure
execution sensitivity, not objective quality. Run completed exit0.

For n equal valid similarities c and m exactly neutral pad scores, an independent
derivation gives

`E = n*c*exp(c/.07) / (n*exp(c/.07)+m)`.

It agrees with the source fixture within2.78e−17. The mechanism is retained
softmax denominator mass, not a floating-point tie effect. For arbitrary finite
unit padding vectors, attenuation bounds their cosine contribution by1e−5;
this does not make their softmax mass zero. Real embeddings need not resemble
the fixed fixture, so no population effect size is inferred.

## Consequence for method discovery

This identifies a concrete implementation dependency worth accounting for in a
fair reproduction. It does not establish that the original authors used this
exact function in the published retrieval runs or that it harms those results.
The independent study concerns translation; its outcomes cannot rank SLRet
methods or explain our CiCo errors. Author-code attribution is a repository
statement, not independently verified provenance.

This is related to the existing C14 inner-padding question, but distinct from
C38's fixed-shape numerical batching audit. Do not reinterpret it as permission
to rerun the failed masking variants or as novel batch-invariant retrieval.
Ordinary mask exclusion alone is a reproduction control. A new method still
needs a measured bottleneck, non-colliding intervention and the original gates.

The next justified step for this particular resource would be **provenance and
task verification before any trained-effect test**. No compatible trained C²RL
retrieval checkpoint/pipeline is verified by these source reads. Do not build a
replacement pipeline, download prediction assets, or train the SLT reproduction
as a proxy for the missing retrieval baseline. No candidate or global barrier.

## Provenance and limits

Fetched raw-file SHA256s:

- models/models.py: `81edd2f15e09ac5d388303a142798ac425c04896edffc6ed036d03d1b79e54a2`
- train_slt.py: `0672ed4fe06b0ec9ea730689185aef393712c85415ec1778036c3c3e1558bb25`
- dataloader/datasets.py: `f3092c071aad003908d5c8149b6eb0972e982ab2c8b412af6b1c38e6e088d0f2`
- configs/phoenix/config1.yaml: `1fb793744d338e1ffc81d8728df956a769974d8acfa7e8544de503554fdff4d1`
- README.md: `b372c455847f3d30bc5e05fe50e21e1215869930c210faa108e6949ed562b85a`

Discovery queries: `"C2RL" "github" sign language retrieval` and
`"C²RL" sign language code`. Secondary hits were discovery only. Browser raw/API
opens failed; read-only direct HTTP fetched the pin and allowed source paths.
No full clone, dataset, checkpoint, GPU, benchmark TEST examples/predictions,
SEDS assets, upstream changes, author contact or external-model uploads.
Published benchmark references in web results were not used for selection.

Evidence grade: strong for behavior of this pinned kernel under the declared
fixture; unverified for original-C²RL lineage/parity, and absent for real
retrieval harm. The scope checks—not the fixture's existence—prevent a false GO.
