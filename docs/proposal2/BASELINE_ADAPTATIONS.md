# DIVE-SLR v2 baseline adaptations

The official SEDS repository is available at the exact clean detached commit
`434e3f714fcb6a7d1f4001fb9a246bbd93ec0246`. No upstream source patch is applied. DIVE instead owns
a side-effect-free adapter in `methods/dive/src/dive/adapters/seds.py`; real checkpoint parity and
controlled reproduction remain blocked on the separately distributed SEDS assets. The native
How2Sign reproduction contract is checked in as
`methods/dive/configs/seds_how2sign_reproduction.yaml`.

## Controlled How2Sign split routing

Pinned SEDS exposes `data_h2/train.pkl` and `data_h2/test.pkl` but no `dev.pkl`. Its training loop
evaluates and selects on the `test` dataloader. DIVE therefore treats that behavior only as the
published/debug protocol. The controlled protocol preserves the upstream train/test identities,
uses the disjoint local `labels.dev.json` for checkpoint selection, and reserves `test.pkl` for the
single locked final evaluation. This is a protocol repair required by INV-02, not a change to the
SEDS architecture or score.

Implemented adapter-level corrections and taps:

- validates the clean pinned checkout, complete locked checkpoint-to-model tensor equality,
  reproduction-config hash, exact train/eval script hashes, typed published flags and required
  native model fields;
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
- routes controlled manifest rows through a manifest-driven native input builder that reuses the
  pinned loader's pose, RGB, tokenizer and collate methods without its hard-wired split lookup;
  validates feature roots, IDs, synchronized grids, masks and layouts, restores its unused RNG draw,
  and executes source in memory so Python caches do not dirty the pinned checkout.
- captures the exact raw-frame indices retained after the native `max_length_frames` subsampling and
  union-of-valid-hand-frame filter. Clip/RF reports now traverse this non-identity map; tests cover
  both every-other-frame subsampling and removal of a hand-invisible frame. Stored RTM pose and raw
  video remain identity-aligned before this native selection. RGB raw-interval alignment remains
  explicitly pending inspection of the released I3D artifacts rather than being inferred from the
  stored pose file alone.

The pinned `modeling.py` reads `task_config.freeze_exfusion`, although the pinned CLI parser never
defines that argument. The checked contract records `freeze_exfusion=false` as an explicit
compatibility value; it also supplies `not_load_visual=false`, which the upstream checkpoint loader
requires. These values repair construction only and do not patch upstream source or alter the
published score.

Still open before a controlled B0 claim:

- score/tap parity on the actual released How2Sign checkpoint and processed I3D/RTMpose samples;
- the external `ViT-B-32.pt`, SignBERT initialization, processed I3D features and locked SEDS
  checkpoint.

The `dive baseline validate` stage is wired to this adapter and controlled manifest bridge. It
requires `validate_data.audit` plus its checksummed native frame-lineage directory and rejects any
replay difference before encoding. It performs a real unpadded parity comparison against the upstream
`get_similarity_logits(..., is_train=True)` dispatcher after removing `exp(logit_scale)`, checks
prelogit cosine bounds, evaluates the full ID-addressed dev gallery, and registers checksummed
scores and metrics. The command is dev-only so its existence cannot provide an accidental test
selection route.

The existing CiCo compatibility work for proposal 1 is not silently reused as a SEDS adaptation.
