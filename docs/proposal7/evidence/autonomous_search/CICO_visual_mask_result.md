# CiCo visual masking: protected valid states, content-dependent masked readouts

2026-09-16. ANALYZED; AI-assisted academic-research-suite source verification.
This is an information-flow certificate, not a trained-effect experiment or method.

## What the source establishes

Pinned SLRT commit `38a4f7b00da7a858d59b7fabe5093876a84db8e0`:

- [PH loader](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/dataloader_ph_retrieval.py#L151)
  zero-pads visual features; mask1 marks CLS and padding, mask0 marks valid slots.
  The local `shared/slr_common/upstream/cico_bridge.py:35` uses the same polarity.
- [FeatureTransformer](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/module_clip.py#L397)
  projects each input slot independently in the configured 2D/sum path, adds
  positional embeddings and applies tokenwise LayerNorm.
- [Visual attention](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/module_clip.py#L278)
  supplies `mask==1` as key_padding_mask at every layer. This excludes masked
  keys/values, **not masked queries**. The distinction agrees with
  [PyTorch's documented contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html).
- [Retrieval scorer](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/modeling.py#L497)
  removes masked visual rows from the I2T outer average, but includes them in
  the T2I inner visual softmax. Returned projected masked states are not zeroed.

At fixed mask and positions, split a layer's inputs into valid V and masked M.
Its dependency structure is

`V_next = f(V); M_next = g(M, V)`.

Attention keys/values come only from V; residuals, MLPs and LayerNorm are
tokenwise. Induction preserves this structure across layers. Therefore
`d V_final / d M_input = 0` in the mathematical model, while the reverse
cross-block derivative need not vanish. This argument requires finite inputs,
at least one valid key and the configured slotwise input projection. It is not
a claim about all-padding cases, NaNs, alternate concatenation/3D paths, changes
to validity or positional layout, or bitwise parity across execution shapes.

CLS and zero-padded slots begin with learned, position-dependent states. Their
queries can read valid video content despite never serving as attention keys.
Calling their outputs *readouts* describes this dependency, not author intent,
learned semantic specialization, or equivalence to another architecture.

## Fixed executed certificate

[Protocol](CICO_visual_mask_protocol.md),
[source fixture](../../../../methods/information_probe/visual_mask_contract.py),
[JSON](CICO-VISUAL-MASK.json).

Unmodified AST classes were extracted after a source SHA256 check. Fixed random
two-layer width8 encoder, four feature slots/two valid, CPU FP32, seed42. This is
not a trained CiCo checkpoint. Interventions preserve masks and tensor shape.

| Measurement | Result |
|---|---:|
| Max valid-output change after changing padded inputs | 0 |
| Max padded-input gradient from fixed valid-output functional | 0 |
| Max masked-output change after changing valid inputs | .341427028 |
| Valid-input gradient norm from fixed masked-output functional | .705776989 |

All preregistered assertions passed; process exit0. Saved code hash matches the
executed script. Nonzero values establish possible flow in this fixed fixture,
not trained prevalence, semantic usefulness or a lower bound on retrieval gain.
No dataset/checkpoint access, retrieval scoring, GPU or optimizer update.

## Consequence for method discovery

The proposed *padded inputs contaminate valid visual states* explanation is
unsupported for the checked path. No encoder-mask repair follows. Conversely,
masking the scorer's visual padding removes content-dependent output pathways,
not merely numerical zeros. This changes how to interpret the earlier
[C14 interventions](AS-C14_result.md), without rerunning them: their negative
results do not identify whether lost readouts, changed competition, or training/
inference mismatch caused the loss. This fixture does not distinguish those
explanations either. It is not a causal explanation of C14's measured effect.

Three tempting interventions fail the current proposal gate:

1. Remove those outputs: this repeats the failed C14 inference-mask family.
2. Replace variable masked slots with a fixed learned query bank: attention
   pooling with learned seeds is established prior art, including
   [Set Transformer §3.2](https://arxiv.org/html/1810.00825v3#S3.SS2). No novel
   mechanism or measured local bottleneck is established by the replacement.
3. Add register tokens to protect valid states: the alleged contamination is
   absent here. [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
   concerns a different artifact mechanism; its abstract is adjacent prior, not
   evidence of a CiCo defect. These masked queries cannot write back to valid
   states, so a generic register analogy is insufficient.

These are rejected formulations, not the research loop's required three surviving
materially distinct candidates. Do not launch a query-count, pooling, register,
mask or temperature sweep on this basis. A useful future question would require
evidence that a particular semantic distinction is unavailable to the deployed
readout, with an adequate control and an intervention outside the closures.
No such evidence is supplied by this certificate. Move to another mechanism.

## Scope and provenance

Source SHA256: `8fae51a298872c9d9e07e898a7f09a117cdfbd21a0fc51c2f3a608af2ee6d2d5`.
Read scope: relevant visual classes, encode_image, mask construction, local
bridge and scorer; not a new full-repository audit. Set Transformer §3.2 and
abstract/metadata inspected; Registers abstract only. No full-paper review or
exhaustive novelty claim. No TEST examples, SEDS assets, upstream edits or trained
model changes. Strong evidence for the stated code dependency; no efficacy
evidence, independent replication or bibliometric/retraction/COI certification.
ARS verification prevented promoting a mask concern into a new method.
No Q38, Proposal8, GO or global barrier. Research goal remains active.
