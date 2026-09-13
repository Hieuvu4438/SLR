# Strong-control adaptation ledger

These controls are comparison baselines, not Shared-Support Sign Contrast novelty.

| Control | Pinned source principle | Method 1 adaptation | Status |
|---|---|---|---|
| `caption_hn` | SAN commit `82aba9cbc1beb403abef6e9a3875ca52479805c8`: within-video positive plus hard-caption classes, token softmax at temperature .07, valid visual-position mean, learned scale | Student UPRet text tokens, the shared mined captions, corrected masks and variable-K class masking; CE with original at class 0 | PASS_SYNTHETIC |
| `fsc_local` | FSC-CLIP commit `604015db3f009a8f7485f1fb7e21d8343c67664a`: detached token-to-patch support, patch-wise min–max, normalized pooled visual token, masked token log-sum-exp and hard-negative class loss | UPRet clips act as patches; invalid clips are excluded before min/max; constant affinity falls back to uniform-valid support; initial focal γ=1 and negative-label smoothing .1 | PASS_SYNTHETIC |
| `fsc_local_caption_hn` | Combination of the two transferred controls | Adds both explicitly weighted losses once each on top of the unchanged corrected UPRet base | PASS_SYNTHETIC |

Primary source locations:

- SAN: `https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/models.py`
- FSC-CLIP: `https://github.com/ytaek-oh/fsc-clip/blob/604015db3f009a8f7485f1fb7e21d8343c67664a/src/training/losses/loss.py`

The code is a narrow, attributed reimplementation of the loss behavior. It does not vendor
either framework or claim exact reproduction with their original backbones/training recipes.
