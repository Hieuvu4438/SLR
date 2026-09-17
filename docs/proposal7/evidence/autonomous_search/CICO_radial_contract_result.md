# CiCo radial-information contract: conditioning prevents a simple conclusion

2026-09-16. ANALYZED, AI-assisted academic-research-suite source verification.
No method GO, new candidate or retrieval improvement is established.

## Outcome

The claim that cosine normalization necessarily removes an independent text
magnitude coordinate is too strong for this architecture. In exact arithmetic,
the learned post-LayerNorm affine constraint can determine the radius from the
direction. However, its direct inverse is not sufficiently accurate on the
existing native-precision TRAIN cache: **11.770844% p95 relative error** versus
the locked1% recovery screen. This is an inconclusive information-loss check,
not evidence that the discarded quantity is semantic or useful for retrieval.

The [locked protocol](CICO_radial_contract_protocol.md),
[executable check](../../../../methods/information_probe/radial_contract.py) and
[result JSON](CICO-RADIAL-CONTRACT.json) retain the unsuccessful screen.
Execution exited0 in1.642s; exit0 certifies completion, not recovery success.

## Source and algebra

Pinned SLRT revision38a4f7b00da7a858d59b7fabe5093876a84db8e0:
`module_clip.py:602,624` applies LayerNorm then projection to visual/text tokens;
`modeling.py:483` onward normalizes their norms before token cosine scoring.
The cache writer and bridge preserve these projected tokens before that step.
Actual checkpoint text projection is512×512FP16; visual projection768×512FP16;
LayerNorm affine parameters areFP32. Historical checkpoint selection used DEV;
this check performs no new checkpoint selection or DEV access.

For row-vector output z=(gamma*u+beta)P with sum(u)=0, set
a=P^{-1}(1/gamma), b=sum(beta/gamma). Then z*a=b and, for d=z/||z||,
||z||=b/(d*a), when P is invertible, gamma has no zero coordinate and b≠0.
LayerNorm epsilon does not invalidate the exact zero-mean constraint. Finite
precision can perturb it. This is elementary local algebra, not a novel method
or a claim that the current scorer learns this inverse. Background:
[Ba, Kiros & Hinton, Layer Normalization](https://arxiv.org/abs/1607.06450),
metadata/abstract inspected; exact dtype path read from pinned source.

The real projection condition number is3824.9794, minimum singular value
.0002716032, minimum absolute gain.7284733, and b=59.26845. The coefficient
norm4164.8319 makes small affine-plane perturbations potentially important.
FP64 linear-solve relative residual is4.17e−13; the fixed synthetic exact
control reconstructs radii within1.26e−15 relative error. Zero-bias and
rectangular-projection inputs are explicitly refused, not silently inverted.

## Existing TRAIN-cache results

All7096 TRAIN sequences inspected; no fitting or selection. FP64 reconstruction
uses stored native output values promoted to double precision.

| Scope | Slots | Median relative error | p95 | Maximum |
|---|---:|---:|---:|---:|
| Mask-valid text tokens |125246|1.364909%|11.770844%|51.805272%|
| All text slots, including padding |227072|2.967280%|13.558349%|51.805272%|

All reconstructed radii are positive and finite; no denominator is zero.
Only43.9535% of valid slots fall within1% error. Rounding the normalized
directions toFP32 before the inverse changes valid-slot p95 to11.770896%,
so that last rounding step alone is not the explanation.

## Explicitly posthoc precision isolation

[Protocol](CICO_radial_precision_validation_protocol.md) and
[separate result](CICO-RADIAL-PRECISION.json):64 fixed synthetic pre-LayerNorm
rows, same checkpoint parameters, actual source-extracted LayerNorm class.
This is CPU dtype-stage emulation, not real hidden-state or GPU-kernel replay.

| Operation path | p95 relative error | Maximum |
|---|---:|---:|
| FP64 LayerNorm and projection |3.64e−13|4.97e−13|
| Source FP16-output LayerNorm, FP64 projection |.017654%|.026860%|
| Source FP16-output LayerNorm and FP16 projection |2.072366%|2.756542%|

Precision alone can therefore cause this inversion check to fail on valid
inputs. This does **not** explain the full11.77% real-cache error: the synthetic
hidden distribution differs. Independently, recovered/true radius agrees with
1/(1+delta), delta=(z*a−b)/b, within9.22e−14 across these fixtures.
Validation exited0. Three dedicated unit tests passed, including a nonsymmetric
projection orientation check and zero-gain refusal.

## Decision and limits

- Neither 'text norms are fully recoverable in practice' nor 'normalization
  destroys useful semantic information' survives as a demonstrated conclusion.
- Do not use this failed numerical inverse as permission for a norm/confidence
  gate, cosine-removal sweep, precision-training rescue or closed geometry method.
- The visual projection is rectangular; this sufficient inverse does not apply.
  That is not proof of visual information loss or a measured bottleneck.
- One existing checkpoint, TRAIN representations only. No encoder forward,
  optimizer, retrieval score, GPU use, new annotation, TEST data or SEDS assets.
- This refines Q01's information-loss claim boundary; no Q38, Proposal8, method
  GO or global research barrier. A future radial proposal needs independent
  admissible evidence that a useful signal is inaccessible to the deployed scorer,
  plus novelty and closure checks. Another inverse/precision sweep is not that.

Provenance is machine-readable in both JSONs. TRAIN cache SHA256
9e0169b33b01f1120997818e550f3b9969effcc6e8af0e67164f3ec7d1d83062;
checkpoint ee45bd2dcad57ff4237eaeb2fc326d288698400c13aac0e51d9df01496b36351;
manifest f032260cf21578876fcff997bef1df0d46094635d30032edb6b01c80693ace53.
