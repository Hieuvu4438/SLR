# DIVE-SLR v2 baseline adaptations

The official SEDS repository is available at the exact clean detached commit
`434e3f714fcb6a7d1f4001fb9a246bbd93ec0246`. No upstream source patch is applied. DIVE instead owns
a side-effect-free adapter in `methods/dive/src/dive/adapters/seds.py`; real checkpoint parity and
controlled reproduction remain blocked on the separately distributed SEDS assets.

Implemented adapter-level corrections and taps:

- validates the clean pinned checkout, complete locked checkpoint-to-model tensor equality,
  reproduction-config hash and required native model fields;
- provides a lazy official model factory that imports `modules.modeling` without importing the
  distributed training entry point and fails closed when CLIP/SignBERT/checkpoint resources are
  absent;
- retains fusion as the only selected baseline branch, preserves both directional matrices in
  `[video,text]` orientation, mixes by locked `dual_mix=0.5`, and emits the equivalent unscaled
  prelogit cosine score instead of adding `exp(logit_scale)` logits to DIVE residuals;
- masks invalid video/text positions before each directional softmax; an unpadded fixture proves
  equality to the upstream formula and a padded fixture proves invariance;
- explicitly converts SEDS `0=valid` video masks and `1=valid` text masks, removing exactly one
  leading video CLS only for local evidence;
- exposes contextual native pose/RGB/text outputs separately from the pre-Transformer RGB cache and
  cloned `gcn_emb -> fixed window -> sign_conv -> mean` local pose path;
- accepts synchronized right/left/body pose trees in the shared evidence warm-up/student runners and
  records conservative half-open raw-frame RF intervals (nominal 16, pose bound up to 24 frames).

Still open before a controlled B0 claim:

- true train/dev/test routing and dev-only checkpoint selection;
- score/tap parity on the actual released How2Sign checkpoint and processed I3D/RTMpose samples;
- a complete native reproduction config plus the external `ViT-B-32.pt`, SignBERT initialization,
  processed features and locked SEDS checkpoint.

The existing CiCo compatibility work for proposal 1 is not silently reused as a SEDS adaptation.
