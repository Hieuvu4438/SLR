# Upstream code map (SEDS, derived from CiCo) — verified by reading code 2026-09-23

Paths relative to `third_party/SEDS/`. CiCo (`third_party/SLRT/CiCo/CLCL/`) shares the entrypoint/metrics skeleton.

## Data
| Item | Location | Facts |
|---|---|---|
| Split files | `data_{ph,csl,h2}/{train,test}.pkl` (+ `data_ph/dev.pkl`) | dict sentence_id → entry (PH) or list of entries (CSL/H2S: several videos per sentence). Captions are English (`text`); PH/CSL also keep original German/Chinese in `ori_text`. **No gloss fields.** |
| Loader registry | `dataloaders/data_dataloaders.py::DATALOADER_DICT` | `"dev"` and `"test"` both map to the *test* loader with `subset="test"` hard-coded. |
| Eval loader | `dataloaders/dataloader_ph_retrieval_pose.py::ph_DataLoader_pose` | one item per video; `cut_off_points` = cumulative #videos per sentence (group structure). RGB path = `features_RGB_path/<subset>/<video>.pkl`; pose path = `features_path/<video>.pkl` (flat). |
| Train loader | `dataloader_*_train_pose.py` | one item per *sentence*; a random video of that sentence is drawn; EDA text augmentation (`random_swap`, p=0.5) builds `pairs_text_aug` used on the T2I side. |
| RGB features | `<ds>/I3D_features/<split>/*.pkl` → `item['feature']` [T_clip, 1024] | offline I3D (CiCo domain-aware style), ≤64 clips; zero-padded to `feature_len`=64. |
| Pose | RTMPose 133-kpt wholebody pkl: `keypoints` [F,133,3], `img_list` | hands (21 each, cropped + normalised to 256, left mirrored), body 7 joints; frames subsampled if >300; clips = sliding windows 16, stride 1, ≤64 clips (linspace if more). |
| Masks | `body_mask` [B, 65]: index 0 = CLS slot (0), then **0 = valid, 1 = pad**. `clips_start` [B,64], −1 = pad. |

## Model (`modules/modeling.py::CLIP4Clip`)
- Text: CLIP ViT-B/32 text transformer (`clip.encode_text(return_hidden=True)`), token-level hidden [B, 32, 512], mask from token ids. Max 32 tokens (CLS + linspace subsample + EOS).
- Pose: `signbert.gcn_emb` (ST-GCN per hand/body) → per-frame 1536-d; per clip window → `sign_conv` (temporal conv block with BatchNorm, applied to B·64 windows incl. zero pads) → mean over 16 frames → [B,1536,64,1] → `clip.encode_image` (CLIP visual transformer, 1×1 conv patch embed 1536→768, 12 layers, 65 positions) → token hidden [B,65,512].
- RGB: `clip_rgb.encode_image` on I3D [B,1024,64,1] → [B,65,512].
- Fusion: `module_fusionencoder.Gloss_Fusion_Transformer` (CGAF) pose+rgb → fusion tokens [B,65,512]; computed *per rank before all-gather*.
- CLIP weights converted to **fp16** (`convert_weights`), trained in fp16 with BertAdam.

## Similarity (`flip_similarity_softmax`) — for each stream X ∈ {pose, rgb, fusion}
- token sims `s[a,b,i,j] = <v_a,i , t_b,j>` (L2-normalised).
- `I2T[a,b] = τ · mean_{valid i} Σ_j softmax_j(s/0.07) s`   (video tokens attend over text tokens)
- `T2I[a,b] = τ · mean_{valid j} Σ_i softmax_i(s_aug/0.07) s_aug` (with augmented text during training)
- τ = `clip.logit_scale.exp()` clamped ≤ 100 after every step.
- Eval score = `0.5·I2T + 0.5·T2I` of the **fusion** stream (`dual_mix`=0.5), no inference-time normalisation.

## Losses (`CLIP4Clip.forward`, training)
- `CrossEn` = symmetric InfoNCE on each matrix: `L_X = ½[(CE(I2T)+CE(I2Tᵀ))/2 + (CE(T2Iᵀ)+CE(T2I))/2]` for X∈{fusion,pose,rgb}.
- Pose–RGB fine-grained matching (`rgb_pose_match`, weight 0.4): token sim between pose and rgb, diagonal-masked (eye over token index), InfoNCE over videos.
- Total = L_fusion + L_pose + L_rgb + 0.4·L_r2p (KL branch off in released scripts).
- Negatives: in-batch only (global batch 128 after all-gather). No hard-negative mining, no false-negative handling (duplicate captions in a batch are treated as negatives).

## Distributed
- `until_module.AllGather`: forward all_gather, backward returns the local slice only; DDP averages over ranks ⇒ encoder grad = (1/world)·dL/dθ; `logit_scale` grad = full dL/dτ. Per-rank BN statistics (no SyncBN). Emulated exactly by `methods/sota_slret/src/gradcache.py`.

## Training / selection
- `main_task_retrieval.py::main`: BertAdam, warmup_cosine (10% warmup), lr 1e-5 (CLIP parts), 1e-4 (signbert), others lr=sign_lr 1e-4, wd 1e-3, grad-clip 1.0, 200 epochs, batch 128.
- **Checkpoint selection: after every epoch, eval on the TEST loader, keep best fusion T2V R@1** (`save_best_model`). Released checkpoints are therefore test-selected. Our protocol selects on a train-carved validation split instead.

## Metrics (`metrics.py`)
- Sim is [N_video, N_text_groups]; upstream pads groups with −inf into [G, max_group, N_text].
- `tensor_text_to_video_metrics(stacked)` → reported as **Video-to-Text** (each video ranks unique texts).
- `compute_metrics(tensor_video_to_text_sim(stacked))` → reported as **Text-to-Video** (each text ranks sentence-groups, group score = max over its videos).
- Tie behaviour: V2T ties broken by argsort index order; T2V `compute_metrics` appends extra entries for tied positions. V2T MedR uses `torch.median` (lower middle), T2V MedR uses `np.median`.
