# `upret_main_corrected_v1` repair ledger

This ledger separates common baseline corrections from the Shared-Support Sign Contrast
intervention. No item below is claimed as Method 1 novelty.

| Upstream behavior | Corrected behavior | Verification status |
|---|---|---|
| NCCL initialized at module import | Distributed initialization occurs only inside runtime setup | PASS: patched upstream imports on CPU without initializing a process group |
| `get_video_feat` feature default follows an invalid MLP path | `encode_local` passes `shaped=True, video_frame=1` and unpacks all nine values | PASS: mock contract plus full UPRet B=2 forward/backward |
| `get_similarity_logits` invokes `pdb` and eval plotting | Debugger and implicit plotting removed | PASS: tracked patch `patches/upret_corrected_v1.patch` |
| Directional inner softmax includes padding and later relies on `nansum` | Mask before softmax; finite affinities; invalid outer positions contribute zero | PASS: padding-invariance and dense/blocked gradient tests |
| Grouped evaluator slices `segment_ids[input_mask,...]` | Slice `input_mask[filter_inds,...]` | PASS: tracked upstream patch; new evaluator avoids the tuple path |
| Custom gather backward returns only the local slice | Sum full output gradient across ranks, then return local slice | PASS: two-rank Gloo encoder-gradient fixture matches the single global objective |
| Test is used during training / PH dev loader is incomplete | Complete dev selection only; test locked until configuration selection | Pending trainer/evaluator |
| Tie expansion can emit multiple ranks per query | One stable manifest-order rank per query | PASS: exact-tie fixture yields one rank/query |
| Import-time NLTK downloads and removed `np.long` aliases | Legacy EDA import is lazy; Method 1 tensors use explicit integer dtypes | PARTIAL: import safety passes; legacy loaders are audit-only |
| UPRet imports unpinned `textaugment.EDA` for one random swap | Reproduce the inspected 1.3.4/2.0.0 one-swap algorithm with a stateless per-sample RNG | PASS: deterministic augmentation fixture; upstream package-version ambiguity is retained in the provenance ledger |
