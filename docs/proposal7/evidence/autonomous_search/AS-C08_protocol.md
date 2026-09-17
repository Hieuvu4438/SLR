# AS-C08 preregistration: exact text-augmentation stress

Date2026-09-15. Q21, D/G layer. Frozen seed42 checkpoint; no optimization,
dev fitting, changed positives or test access. Diagnostic, not a new method.

Use the deployed deterministic50% one-word-swap recipe, seeds42/1337/2026 at
epoch0. Clean control re-encodes captions using the same128-item batch shapes
as the existing frozen cache. Require exact contextual-token parity before
interpreting results. Same full519 dev gallery; report clean, both-channel
augmented stress, and actual training-style channel asymmetry (A clean/B aug).
The latter is NOT augmented deployment performance or a method improvement.

On train, compare each pair against its fixed clean-R0 strongest confuser, using
the same paired kernel as AS-C03/C04. Report negative/near-tie margins separately
for same versus different clean deployed text inputs, and lost/won positive
margins. These are fixed-pair diagnostics, NOT full7096-gallery retrieval.
Store margins and all dev ranks/scores; no candidate selector. Check whether
augmentation supplies distinguishable train errors absent in the clean probe.

Limits: augmentation seeds are not independently trained backbones; no expert
validity judgment of swapped sentences; perturbation sensitivity cannot show
that removing swaps would improve training. If substantial effects appear,
localize before candidate generation. A trivial augmentation toggle is a control,
not sufficient research novelty. Prior negative clean-only probes remain valid
for their specification, but cannot characterize the historical augmented loss.
