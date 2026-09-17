# AS-C11: train-fitted feature geometry screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run; autonomous experiment design authorized by governing user file.
- Date: 2026-09-15
- Verification Status: preregistered diagnostic, not method GO.

Previous turn classification: progress (AS-C07 collision and AS-C08–10 measured).
No live experiment found. Q08 is a different causal layer from the latest
optimization/nuisance screens. No broad literature search before a positive
mechanism. Centering/PCA/whitening are controls, not claimed research novelty.

Inputs: fixed R0 seed42 frozen train/dev contextual tokens. Normalize each
token; fit valid-token moments giving equal weight to each sequence and then
equal weight to video/text modalities. Use all7096 train examples; no dev
moment, positive-pair label, grouping label, or negative mining enters fitting.
Save means/covariance/eigenvalues/transforms and exact source/cache hashes.

Seven fixed variants, no parameter sweep or dev selector:

1. Identity (original tokens): rank parity and max channel error<=2e-5 relative
   to saved R0 channels, consistent with previously validated standalone kernel.
2. Shared centering: subtract the equally weighted shared mean.
3. Separate centering: subtract each modality's train mean.
4. Shared centered PC1 removal.
5. Shared centered random1 removal (fixed seed42), same dimensional reduction.
6. Shared centered ridge whitening: W=(.9*C+.1*trace(C)/512*I)^(-1/2).
7. Shared centered random-orientation whitening: identical spectrum as6,
   Haar-QR basis from fixed seed42, testing nonspecific metric distortion.

Transform ALL slots, including padding, to avoid an additional masking-policy
intervention; fit statistics exclude masked slots. Scorer retains historical
inner padding softmax and outer valid-token averaging. This choice is explicit,
not proof a positive effect would be independent of padding interactions.
Fixed full519 gallery, original positives, directions,R1/R5/R10 and persistent
ranks. No trainable encoder/readout, optimizer, updates, test or SEDS assets.

Report all variants, no significance/GO from a single fixed backbone. An effect
that is also explained by a random control does not support anisotropy-specific
attribution. Even a positive controlled diagnostic needs a distinct candidate,
closest-prior screen, three training seeds and all original method gates.
An invertible transform preserves information; a ranking loss cannot establish
representation irreducibility. Hard timeout600seconds, terminal failures recorded.
