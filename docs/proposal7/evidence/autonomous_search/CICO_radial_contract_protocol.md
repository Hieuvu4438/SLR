# CiCo post-LayerNorm radial-information check

Locked before numerical reconstruction, 2026-09-16. AI-assisted source fact-check
using academic-research-suite; not a method pilot. Checkpoint tensor metadata has
already been inspected: text projection512×512FP16, visual768×512FP16,
LayerNorm parametersFP32. No reconstruction result has been inspected yet.

## Question and derivation

Does unit-normalizing CiCo's post-LayerNorm projected text tokens necessarily
discard an independent magnitude coordinate? This is distinct from corpus
centering/whitening (C11), visual pre/post ridge prediction (C13), and the
two-channel objective identity. It is not a norm/confidence fusion proposal.

For row vectors, z=(gamma*u+beta)P where sum(u)=0. If square P is invertible
and each gamma coordinate is nonzero, define a=P^{-1}(1/gamma),
b=sum(beta/gamma). Then z*a=b. Writing d=z/||z|| gives
||z||=b/(d*a), provided b and d*a are nonzero. This uses the exact zero-mean
constraint, not an approximation that LayerNorm variance equals1; epsilon does
not invalidate zero mean in real arithmetic. Floating-point rounding can.
It does not assert that CiCo's dot-product scorer can implement this inverse.

If b=0 this particular inverse supplies no radius. The non-square visual
projection does not satisfy these sufficient conditions. Neither case proves
that all magnitude information is lost or useful for semantic ranking.

## Fixed check

- Source pin SLRT38a4f7b00da7a858d59b7fabe5093876a84db8e0. Trace encode_text,
  encode_image and flip_similarity_softmax plus cache writer/bridge.
- Use only existing seed42 checkpoint and frozen_train.pt; verify checkpoint
  and TRAIN manifest hashes. No other seeds, DEV/TEST caches, raw media or labels
  used to fit/select anything. TRAIN ids only establish row count/provenance.
- Extract actual LayerNorm gamma/beta and text/visual projection tensors.
  Report projection dimensions/condition and gamma minimum magnitude, b, solve
  residual. Use CPU float64 for algebra, preserve native values when converting.
- Reconstruct every stored TRAIN text-token norm in chunks. Report separately
  mask-valid and all slots, including denominator signs/zero/nonfinite counts,
  relative-error median/p95/max and fractions within .1%,1%,5%. Compare with
  normalized directions rounded to float32 as an explicit precision sensitivity.
  No adaptive tolerance, pseudoinverse rescue, fitted correction or sweep.
- Fixed synthetic control: random seed42,32 tokens,dimension16, invertible
  projection, positive gamma, nonzero b, epsilon1e-5. Norm inverse must agree
  within1e-10 relative error. Zero-bias control must be classified unsupported
  by this inverse rather than divide by zero. Check rectangular refusal.
- Real p95<=1% and all positive finite radii supports only **approximate radial
  recoverability in this checkpoint/cache**; otherwise inconclusive due to
  numerical/structural conditioning, not proof of a semantic bottleneck.
- Maximum runtime120s, CPU threads2. JSON includes hashes and all summaries.
  No optimizer, retrieval scoring, encoder execution or upstream edits.

## Decision boundary

Recoverability would challenge the simple 'cosine throws away independent text
norm evidence' rationale. Failure would require isolating precision/conditioning
before any information-loss claim. Neither outcome justifies removing cosine,
adding confidence gates or a new training run. Video remains outside this inverse.
No Q38/GO follows from a source/representation certificate alone.

Primary background: [Layer Normalization](https://arxiv.org/abs/1607.06450)
(metadata/abstract inspected), with the exact operation verified in pinned
module_clip.py. The algebra above is a local derivation, not a novelty claim.
