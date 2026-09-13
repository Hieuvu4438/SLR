# `upret_main_corrected_v1` repair ledger

This ledger separates common baseline corrections from the Shared-Support Sign Contrast
intervention. No item below is claimed as Method 1 novelty.

| Upstream behavior | Corrected behavior | Verification status |
|---|---|---|
| NCCL initialized at module import | Distributed initialization occurs only inside CLI runtime setup | Pending implementation test |
| `get_video_feat` feature default follows an invalid MLP path | Feature callers pass `shaped=True, video_frame=1, get_hidden=True` | Pending bridge test |
| `get_similarity_logits` invokes `pdb` and eval plotting | Debugger removed; plotting explicit and off by default | Pending tracked UPRet patch |
| Directional inner softmax includes padding and later relies on `nansum` | Mask before softmax; finite affinities; invalid outer positions contribute zero | Pending scorer tests |
| Grouped evaluator slices `segment_ids[input_mask,...]` | Slice `input_mask[filter_inds,...]` | Pending tracked UPRet patch |
| Custom gather backward returns only the local slice | Sum full output gradient across ranks, then return local slice | Pending two-rank test |
| Test is used during training / PH dev loader is incomplete | Complete dev selection only; test locked until configuration selection | Pending trainer/evaluator |
| Tie expansion can emit multiple ranks per query | One stable manifest-order rank per query | Pending fixture test |
| Import-time NLTK downloads and removed `np.long` aliases | Offline resource validation and explicit integer dtypes | Pending tracked UPRet patch |
