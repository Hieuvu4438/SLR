# ELSC — End-to-end implementation specification

**Evidence-Localized Sign Contrast cho sentence-level Sign Language Retrieval**
Phiên bản: 1.0 · Ngày đối chiếu code: 06/09/2026 · Ngôn ngữ tài liệu: tiếng Việt.

Tài liệu này chuyển phương pháp ELSC trong báo cáo nghiên cứu trước thành một đặc tả để AI coding agent triển khai. Phạm vi chính là **CiCo + ELSC trên feature I3D có sẵn**, tiếp theo là kiểm chứng trên backbone mạnh. Đây là thiết kế nghiên cứu, chưa phải implementation đã huấn luyện hoặc kết quả vượt SOTA.

**Cách đọc trạng thái:** `VERIFIED` là tên file, hàm hoặc hành vi đã đọc từ code tại commit ghi dưới đây; `NEW` là module/API/CLI phải viết; `PROPOSED` là quyết định thiết kế hoặc default chưa được tuning. Các lệnh `python -m elsc...` trong tài liệu là interface cần triển khai, chưa tồn tại trong repository upstream.

## 1. Mục tiêu và contract cho AI coding agent

Xây dựng một phương pháp auxiliary training giúp model phân biệt từ đúng `w` và từ dễ nhầm `u` bằng raw visual features ở vùng thời gian có bằng chứng cho `w`. Adapter được chia sẻ với nhánh retrieval, vì vậy việc học local phải ảnh hưởng được representation dùng khi inference.

Ba giả thuyết cần kiểm chứng:

1. Cùng negative và budget, supervision tại support được teacher chọn tốt hơn random support.
2. Phân biệt lexical pair tại support hữu ích hơn chỉ phân biệt positive/negative toàn câu.
3. Margin positive–negative phụ thuộc vào vùng bằng chứng nhiều hơn vùng control, đồng thời retrieval trên gallery gốc cải thiện hoặc có trade-off được đo rõ.

**Phải giữ:** dataset, split chính thức, gallery, positive mapping và metric tương ứng. Cache training không phải dataset/benchmark mới. Eligibility chỉ lọc auxiliary supervision; tuyệt đối không lọc query/gallery test theo eligibility.

**Phải bàn giao khi implementation hoàn thành:** code ELSC, compatibility patches có lý do, environment lock, manifest và hash, config từng run, checkpoint, evaluation outputs, per-query ranks, ablations và log tài nguyên. Không tự điền kết quả chưa chạy.

**Thứ tự:** compatibility/parity → baseline chọn bằng dev → cache teacher → ELSC-Min → controls → ELSC-Full nếu có tín hiệu → mở rộng dataset/backbone. Không triển khai pose, OT, Gaussian, LLM reranker cùng lúc.

## 2. Repository: clone gì, tái sử dụng gì

### 2.1. CiCo: dependency chính

- Repository: [FangyunWei/SLRT](https://github.com/FangyunWei/SLRT).
- Thư mục chính thức: [CiCo](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo).
- Commit đã kiểm tra: `38a4f7b00da7a858d59b7fabe5093876a84db8e0`.
- Paper: [CiCo, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Cheng_CiCo_Domain-Aware_Sign_Language_Retrieval_via_Cross-Lingual_Contrastive_Learning_CVPR_2023_paper.html).

```bash
mkdir -p third_party
git clone --filter=blob:none --no-checkout https://github.com/FangyunWei/SLRT.git third_party/SLRT
git -C third_party/SLRT sparse-checkout init --cone
git -C third_party/SLRT sparse-checkout set CiCo
git -C third_party/SLRT checkout --detach 38a4f7b00da7a858d59b7fabe5093876a84db8e0
git -C third_party/SLRT rev-parse HEAD
```

Sparse checkout toàn bộ CiCo vẫn có thể tải binary trong thư mục. Với đường truyền hạn chế, đọc từng file tại pinned commit trước; không cần tải checkpoint/dataset để bắt đầu viết wrapper. `--filter=blob:none` không bảo đảm checkout nhỏ.

| File upstream đã xác minh | Thành phần dùng lại | Cách dùng |
|---|---|---|
| `CiCo/CLCL/modules/modeling.py` | `CLIP4Clip`, `from_pretrained`, `get_sequence_output`, `get_visual_output`, `get_similarity_logits`, `flip_similarity_softmax` | Dùng encoder và scorer; không thay bằng cosine của pooled embedding. |
| `CiCo/CLCL/modules/module_clip.py` | `FeatureTransformer`, `CLIP.encode_image`, `CLIP.encode_text` | Chèn adapter vào input raw feature trước `FeatureTransformer`; lấy token output teacher cho mining. |
| `CiCo/CLCL/modules/tokenization_clip.py` | `SimpleTokenizer`, normalization, byte encoder, BPE | Giữ token IDs gốc; bổ sung offset mapping. |
| `CiCo/CLCL/metrics.py` | `compute_metrics`, `tensor_text_to_video_metrics`, `tensor_video_to_text_sim` | Reuse metric kernels nguyên bản; wrapper chuẩn hóa matrix orientation và ID mapping. |
| `CiCo/CLCL/main_task_retrieval.py` | `_run_on_single_gpu_new_mix`, training loss logic | Reuse/block-copy có attribution cho score mixing; viết trainer mới chọn checkpoint bằng dev. |
| `CiCo/CLCL/dataloaders/data_dataloaders.py` | Dataset factory | Tham chiếu; cần sửa routing dev. |
| `CiCo/CLCL/dataloaders/dataloader_ph_retrieval.py` | PH evaluation preprocessing | Reuse behavior tokenize/sample/fusion, sửa lỗi integration có ghi log. |
| `CiCo/CLCL/dataloaders/dataloader_ph_retrieval_train.py` | PH training preprocessing, text augmentation | Adapter dataset mới trả thêm IDs và dense indices; kiểm tra đường dẫn feature. |
| `CiCo/I3D_feature_extractor/get_features.py` | Wrapper trích feature | Chỉ dùng khi thiếu feature; đọc cấu hình và `extract_sign_features.py` trước khi chạy. |

Các đường dẫn còn lại cho How2Sign/CSL có trong `data_dataloaders.py`: `dataloader_H2_retrieval.py`, `dataloader_H2_retrieval_train.py`, `dataloader_csl_retrieval.py`, `dataloader_csl_retrieval_train.py`. Chúng cần audit riêng trước khi chuyển dataset; không suy rằng mọi bug PH đều tồn tại hoặc đã được sửa ở chúng. [CiCo CLCL tại commit](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL).

### 2.2. SAN: dependency tham chiếu, không phải pipeline đã hoàn chỉnh

- Repository: [joonmy/SAN](https://github.com/joonmy/SAN).
- Commit đã kiểm tra qua Git: `82aba9cbc1beb403abef6e9a3875ca52479805c8`.
- Paper: [Semantic Hardness Is Not Visual Hardness, ACL 2026](https://aclanthology.org/2026.acl-long.1302/).

```bash
git clone --filter=blob:none https://github.com/joonmy/SAN.git third_party/SAN
git -C third_party/SAN checkout --detach 82aba9cbc1beb403abef6e9a3875ca52479805c8
```

**Cập nhật so với báo cáo trước:** repository truy cập được qua Git trong lần audit này. Tuy nhiên README đang là `Coming Soon`, phần hướng dẫn nằm trong HTML comment. Tree có `datasets.py`, `models.py`, `train_vlp_v2.py`, `train.bash`, `configs/config_gloss_free.yaml`, các label files và `requirement.txt`. Không thấy pipeline mining hay release fine-grained evaluator/candidate manifest hoàn chỉnh trong tree tại commit đã kiểm tra.

`VERIFIED`:

- `datasets.py:S2T_Dataset.generate_hard_negatives` nhận negative table từ bên ngoài; dùng `sentence.split()` và thay tối đa hai từ. Hàm này không xây negative table.
- `train.bash` đặt `--num_hard 5`, `--loss_lambda 0.4`, nhưng `--neg_table_name` còn là placeholder.
- `models.py` có coarse/fine similarity và hard-caption logits để tham khảo control toàn câu.
- `train_vlp_v2.py:evaluate` trong code đã đọc là full-gallery evaluation, không phải đủ bằng chứng rằng official 40-negative stress test đã được phát hành.
- SAN dùng tokenizer BERT tiếng Đức trong trainer đã đọc. Không mang nguyên negative table tiếng Đức ghép trực tiếp với caption/tokenizer tiếng Anh của một cấu hình CiCo.

**Reuse thực tế:** dùng schema/ý tưởng negative table và loss làm tham chiếu. Nếu lấy được official table có cùng ngôn ngữ, import và ghi hash. Không clone SAN rồi giả định tồn tại `mine_negatives.py` hoặc `eval_finegrained.py`.

### 2.3. SEDS và các dependency không cần clone ban đầu

[SEDS chính thức](https://github.com/longtaojiang/SEDS) chỉ cần cho giai đoạn kiểm chứng trên dual-stream backbone. Chưa pin commit SEDS trong đặc tả này; khi bắt đầu, ghi SHA và audit lại vị trí RGB/pose fusion. Không đưa SEDS vào dependency bắt buộc của ELSC-Min.

Không cần clone CLIP4Clip riêng: CiCo đã có bản module được sửa cho retrieval sign. Không cần tự viết lại I3D hay pretrain toàn bộ sign encoder nếu dùng được feature/checkpoint release. Không có URL code C²RL đã xác minh trong tài liệu này; không tự tạo URL dự đoán.

## 3. Compatibility audit bắt buộc trước khi thêm ELSC

Các điểm sau là code-level findings, không chứng minh tác giả đã dùng chính xác trạng thái code này để tạo mọi số trong paper.

| Điểm đã thấy trong pinned CiCo | Hành động triển khai |
|---|---|
| `DATALOADER_DICT['ph']['dev']` trỏ sang H2 loader; PH loader chưa khai báo `dev.pkl` | Sửa mapping và đọc đúng official dev annotation. Không đổi tên test thành dev. |
| PH train loader tạo `video_path_retrain` bằng `self.features_path` | Bổ sung cấu hình `feature_path_mode`; corrected mode dùng `self.features_path_retrain`. Đo/check hash hai input stream. |
| Main train loop gọi `eval_epoch(...test_dataloader...)` rồi cập nhật `best_score` | Trainer ELSC chỉ tạo dev evaluator trong training. Test là lệnh riêng sau khi lock config/checkpoint. |
| `eval_epoch` có indexing `segment_ids[input_mask, ...]` và nhánh unpack không khớp API hiện tại | Viết wrapper feature extraction theo API thực tế; giữ score/metric kernels. Lưu diff compatibility. |
| `get_visual_output(video_frame=-1)` có nhánh không phù hợp feature path chính, biến `visual_cls` không được gán trong nhánh đó | Với tensor feature CiCo, gọi `video_frame=1` như đường chạy chính đã đọc. Không suy tên tham số nghĩa là phải nhập video RGB. |
| `get_loss` có call/transpose đáng nghi, không phải đường `forward` chính | Trích training loss từ `forward`; không chọn `get_loss` chỉ vì tên thuận tiện. |
| `sim_header` default là `Filip`, nhưng choices không liệt kê `Filip` | Wrapper validate riêng; nếu sửa parser upstream thì ghi compatibility patch. |
| Text quá dài được subsample BPE bằng `linspace`, không đơn thuần cắt prefix | Giữ hành vi này trong parity; span mapper phải theo selected BPE indices. |
| Video mask có `0=valid`, `1=masked`; CLS ở index 0 cũng đặt `1` | Bridge phải chuyển quy ước có chủ đích. Không áp dụng mask của HuggingFace trực tiếp. |
| Similarity softmax không loại mọi padding ở mọi trục trước reduction | Không âm thầm “sửa attention mask” trong ELSC rồi quy gain cho method. Nếu thay, tạo baseline patched riêng. |

Nguồn: [data factory](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/data_dataloaders.py), [PH train loader](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/dataloader_ph_retrieval_train.py), [main](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/main_task_retrieval.py), [modeling](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/modeling.py).

Tách rõ hai loại baseline:

1. **Release parity:** checkpoint release + preprocessing/scorer tương ứng, chỉ sửa những lỗi làm không thể chạy. Mục đích xác minh bridge.
2. **Controlled training baseline:** code đã sửa dev routing/feature path, cùng training protocol sẽ dùng cho ELSC. Mọi hàng ablation dùng cùng patches này.

Không cần tái lập toàn bộ historical training để thử ELSC. Tuy nhiên nếu không biết release checkpoint được chọn bằng dữ liệu nào, nó chỉ là initialization/parity artifact có provenance chưa đầy đủ; không được gọi teacher đó là “dev-selected” nếu chưa có bằng chứng. Teacher chính nên là baseline do dự án chọn bằng dev từ protocol đã khóa. Giữ initialization giống nhau giữa controls.

## 4. Cấu trúc code cần tạo

Tất cả file trong `elsc/`, `configs/`, `tests/` dưới đây là `NEW`.

```text
elsc-project/
  third_party/SLRT/
  third_party/SAN/                   # optional reference/import
  shared/slr_common/
    upstream/cico_bridge.py
    data/manifest.py
    data/cico_dataset.py
    data/word_offsets.py
    data/views.py
    features/i3d.py
    evaluation/cico_eval.py
    evaluation/san_eval.py
    resources.py
    utils.py
  methods/elsc/
    elsc/
      data/cache_dataset.py
      config.py
      mining/teacher_align.py
      mining/negative_graph.py
      mining/build_cache.py
      models/adapter.py
      models/local_head.py
      models/retriever.py
      losses/coarse.py
      losses/lexical.py
      losses/evidence.py
      losses/distillation.py
      train.py
      evaluate.py
      prepare.py
      audit.py
      export.py
    configs/ph_min.yaml
    configs/ph_full.yaml
    configs/ablation_*.yaml
    scripts/
    tests/
  configs/ph_base.yaml
  tests/test_package_boundaries.py
  tests/test_word_offsets.py
  tests/test_support_mapping.py
  tests/test_evaluation_contract.py
  elsc/                              # legacy import compatibility only
  patches/cico_compat.patch
  artifacts/manifests/
  artifacts/cache/
  artifacts/parity/
  runs/
```

Mọi generated cache là artifact có schema/version/hash. Dataset gốc và pretrained checkpoints đặt ngoài source tree hoặc bị ignore. Đọc và giữ attribution/license của từng repository; không tự suy mọi repo có cùng license.

`elsc/config.py` triển khai `extends` theo đường dẫn tương đối với YAML hiện tại, recursive dictionary merge; list/scalar override toàn bộ, phát hiện vòng lặp. Resolve thành một YAML đầy đủ và hash nó trước mỗi run. Đây là tính năng loader phải viết, không dựa vào việc YAML tự hỗ trợ inheritance.

Environment: README CiCo mô tả Python 3.7, PyTorch 1.7.1, CUDA 11.0 và các dependency bổ sung. Đây là môi trường tác giả ghi, không phải cam kết chạy trên GPU/driver hiện tại. AI triển khai chọn một môi trường thực tế, ghi exact versions và chạy parity. Nếu port sang PyTorch mới, sửa `np.long`, AMP, distributed launch theo patch riêng; không thay tokenizer/scorer cùng lúc. Khóa dependency sau smoke test, không đưa ra một version matrix “đã kiểm chứng” khi chưa cài/chạy.

## 5. Dữ liệu, feature và provenance

### 5.1. Ưu tiên dùng release assets

Liên kết dưới đây được lấy từ [README CiCo tại commit](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/README.md). Đã xác minh README chứa link; chưa tải và kiểm tra nội dung archive trong phiên viết đặc tả.

- [Sign features](https://drive.google.com/file/d/1Vb-HFZd-rhjN49sB5WwLRpIbyhiC6xTy/view).
- [CLCL checkpoints](https://drive.google.com/file/d/1Hpcn5obCcG5JHa3nLvHqX9pfrp7g6wDu/view).
- [Domain-aware I3D](https://drive.google.com/file/d/1TbX3UjaUvXhsXSQX2UAm81nitt8tsgLq/view).
- [How2Sign official download](https://how2sign.github.io/#download).

Khi feature train/dev/test và checkpoint phù hợp đã có, không chạy pseudo-labeling, I3D pretraining hoặc trích RGB lại. Khi thiếu dev feature, có thể trích trên **dev chính thức bằng frozen sign encoder**, giữ cùng preprocessing; đây không phải training bằng dev.

`get_features.py` upstream là wrapper có đường dẫn và vòng lặp shard hard-coded. Không chạy mù `python get_features.py` trên đường dẫn mới. Kiểm tra `extract_sign_features.py`, data loader clip sampling và checkpoint trước; viết config rõ split/shards/device/output. Feature đã bị temporal pooling toàn video không phù hợp local branch.

### 5.2. Manifest nội bộ

`prepare.py` xuất một record mỗi pair, giữ ID và thứ tự gốc:

```json
{
  "schema_version": 1,
  "dataset": "ph",
  "split": "train",
  "pair_id": "dataset-native-pair-id",
  "video_id": "dataset-native-video-id",
  "caption_id": "dataset-native-caption-id",
  "caption_original": "exact annotation text",
  "caption_model": "exact text consumed by the chosen baseline",
  "caption_language": "verified-from-artifact",
  "feature_agnostic": "/data/ph_domain_agnostic/train/video.pkl",
  "feature_aware": "/data/ph_domain_aware/train/video.pkl",
  "dense_length": 123,
  "feature_dim": 1024,
  "temporal_metadata": "/data/metadata/video.json"
}
```

Các giá trị ID/text trong JSON là placeholder schema, không phải sample thật. `dense_length=123` chỉ minh họa. Ghi dataset version, annotation checksum, ordered ID checksum, checkpoint checksum, feature extraction recipe, actual counts và missing IDs vào `manifest_meta.json`.

`prepare.py` đọc đường dẫn **source annotation và feature roots** từ block `sources` trong config, không cố tạo manifest mới từ chính đường dẫn manifest output. Nếu CiCo archive thiếu dev annotation ở schema `{'text','video_name'}`, viết conversion từ official dev annotations về cùng schema, giữ IDs và caption language của baseline. Nếu cần bản dịch dev mà artifact không cung cấp, ghi thiếu asset và bổ sung recipe dịch được công khai/cố định cho cả baseline và method; không tự coi text của một ngôn ngữ khác là equivalent input. Không dùng test annotations làm nguồn cho dev.

PHOENIX thường báo cáo 7,096/519/642 train/dev/test; dùng số này làm diagnostic, không ép manifest đạt số đó bằng cách loại/thêm sample. How2Sign có count khác nhau giữa papers; kiểm tra actual release. Không lấy intersection các dataset release để tạo một test set “sạch hơn” rồi so với số cũ.

Caption ngôn ngữ nào phải xác định từ artifact và code, không từ tên dataset. Nếu baseline sử dụng bản dịch English có sẵn thì dùng đúng bản đó. Không tự dịch lại test, không thay tiếng Đức bằng English chỉ để negative table dễ dùng.

### 5.3. Canonical tensors và temporal coordinates

| Tensor | Shape | Quy ước |
|---|---|---|
| `dense_h` | `[L, 1024]` | Feature theo timeline trước subsampling của CLCL. |
| `h` | `[B, F, 1024]` | Feature đã chọn `F=feature_len`, mặc định 64. |
| `valid` | `[B, F]`, bool | `True` là clip thật, không gồm CLS. |
| `dense_index` | `[B, F]`, int64 | Index ở `dense_h`; padding là `-1`. |
| `rf_start`, `rf_end` | `[B, F]` | Interval input frame/time thật, cùng hệ tọa độ, half-open `[start,end)`. |
| `upstream_video` | `[B, 1024, F, 1]` | `h.transpose(1,2).unsqueeze(-1)` cho mode `sum`. |
| `upstream_video_mask` | `[B,F+1]` | CLS `1`, real clip `0`, padding `1`. |
| `visual_tokens` | `[B,F+1,d]` | CiCo trả cả CLS; bỏ CLS ở mining/local association. |
| `text_tokens` | `[B,T,d]` | Dùng mask mà `get_sequence_output` trả, không chỉ input mask. |

PH `combine_type='sum'` tại commit tính **`(1-alpha)*aware + alpha*agnostic`** theo đường dẫn evaluation. Với `alpha=0.9`, không diễn giải là 90% aware. ELSC-MVP dùng mode sum và giữ chính xác trọng số/normalization của baseline.

Mode `cat` có input `[B,1024,2F,1]`, sau đó `conv2_trans` có thể trộn trục token. Không hỗ trợ mode này trong MVP. Nếu checkpoint cần cat thì tạo nhánh hỗ trợ riêng, đặt adapter trước mọi temporal mixing và audit locality; không tự reshape cat thành sum.

## 6. Cầu nối CiCo và định nghĩa similarity chính xác

### 6.1. API bridge mới

```python
# NEW interfaces. Không phải tên hàm có sẵn của upstream.
class CiCoBridge:
    def encode_video(self, h, valid):
        # h: [B,F,1024]; giữ autograd nếu đây là student.
        video = h.transpose(1, 2).unsqueeze(-1).contiguous()
        mask = torch.ones(h.size(0), h.size(1) + 1,
                          dtype=torch.long, device=h.device)
        mask[:, 1:] = (~valid).long()
        # index 0 luôn 1 để giữ upstream CLS behavior.
        return self.core.get_visual_output(
            video, mask, shaped=True, video_frame=1, get_hidden=True)

    def encode_text(self, ids, segments, input_mask):
        return self.core.get_sequence_output(
            ids, segments, input_mask, shaped=False, get_hidden=True)
```

`from_pretrained` phải nhận config/checkpoint đúng như `init_model` upstream; không gọi `CLIP4Clip()` không tham số và kỳ vọng load được. Missing/unexpected checkpoint keys phải được giải thích theo allowlist, không dùng `strict=False` rồi bỏ qua toàn bộ.

### 6.2. Hai matrix và mixed score

`get_similarity_logits(..., sim_header='Filip')` trả `I2T_sim`, `T2I_sim`, auxiliary tuple. **Cả hai matrix đều có shape `[N_video, N_text]`** trong hàm CiCo đã đọc; tên `T2I` không có nghĩa tensor đã transpose.

Ở evaluation, `_run_on_single_gpu_new_mix` tạo:

\[
S = \eta I2T + (1-\eta)T2I,\qquad \eta=\texttt{dual\_mix}.
\]

`S[v,t]` là score video–text. V2T rank theo row của `S`; T2V rank theo row của `S.T`. Không thay score bằng trung bình embedding; không softmax lại trước ranking. Temperature token interaction `0.07` trong code khác `logit_scale.exp()` và khác temperature lexical loss.

Trong MVP một positive caption/video, dùng chính các metric kernels gốc qua layout tương đương:

```python
S_grouped = S[:, None, :]  # [N_video,1,N_text], chỉ khi ID mapping 1:1.
v2t = tensor_text_to_video_metrics(S_grouped)
t2v = compute_metrics(tensor_video_to_text_sim(S_grouped))
```

Tên helper lịch sử gây khó đọc; kiểm tra hướng bằng IDs và fixture, không bằng tên hàm. Nếu manifest có many-to-one/multiple positives, grouping phải theo official protocol và cut-off mapping; không dùng đoạn singleton trên để âm thầm bỏ caption. Dự án dừng ở gate evaluator cho dataset đó cho đến khi mapping đúng.

`compute_metrics` dùng tìm mọi vị trí score bằng diagonal sau sort; tie có thể tạo nhiều rank entries. Giữ nguyên official metric khi so sánh paper, lưu tie rate và per-query scores; có thể báo thêm diagnostic tie policy nhưng không thay metric chính lặng lẽ. [Metrics source](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/metrics.py).

### 6.3. Coarse training loss

Với `mix_design='balance'`, đặt `A=I2T`, `B=T2I`, `CE(M)` là cross-entropy với diagonal targets:

\[
L_{CLCL}=\frac12\{\eta CE(A)+(1-\eta)CE(A^T)
+\eta CE(B^T)+(1-\eta)CE(B)\}.
\]

Giữ original/augmented text branch như upstream. Với `dual_mix=1`, công thức thu về `0.5*(CE(A)+CE(B.T))`. Mode `depart` khác hệ số; triển khai đúng nhánh nếu dùng, không gọi chung là equivalent. Kiểm tra scalar loss và gradient trên cùng logits trước training.

### 6.4. Golden parity

Trước ELSC, lưu fixture gồm checkpoint hash, batch IDs, original feature tensors, token IDs, masks, hidden states, `I2T`, `T2I`, mixed scores và ranks. So sánh bridge với upstream cùng runtime/dtype và `eval()`.

Mục tiêu ban đầu: `atol=1e-6, rtol=1e-5` cho FP32 nếu kernel cho phép; FP16 bắt đầu `atol=rtol=1e-3`, luôn xem rank flip ở near ties. Đây là tolerance kiểm tra, không chứng nhận trước rằng mọi CUDA kernel đạt được. Nếu full pipeline upstream lỗi, tạo reference từ các hàm encoder/scorer gốc với batch đã chuẩn hóa; ghi rõ parity ở component level và phần wrapper đã sửa.

Adapter zero-init phải giữ output ở parity với baseline trong `eval()`. Chỉ sau gate này mới bật lexical loss.

## 7. Kiến trúc ELSC-Min

### 7.1. Vị trí adapter

Pipeline student:

```text
frozen I3D features → fixed feature fusion → sample F clips
                   → local residual adapter → CiCo FeatureTransformer → CLCL score
                                            → local projection → lexical loss
```

Local projection đọc **output adapter trước FeatureTransformer**, không đọc `visual_tokens` sau self-attention. Teacher có thể dùng contextual tokens để đề xuất support, nhưng điều đó không biến support thành ground-truth sign annotation.

Default `r=0`: pointwise residual MLP, không nhận caption, word ID, position toàn câu hoặc pooled video. Một raw I3D clip đã chứa một cửa sổ thời gian; pointwise ở đây không có nghĩa chỉ nhìn một frame.

\[
h'_n=h_n+A_\theta(h_n),\qquad
A_\theta(h)=W_2\,GELU(W_1\,LN(h)),\quad
z_n=\operatorname{norm}(P_\psi h'_n).
\]

Adapter hidden 256 là default đề xuất. Chỉ `W2` và bias output khởi tạo zero; không zero toàn bộ mạng.

```python
# NEW: elsc/models/adapter.py — reference implementation
import torch
from torch import nn
from torch.nn import functional as F

class LocalResidualAdapter(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h, valid):
        delta = self.up(F.gelu(self.down(self.norm(h))))
        delta = delta * valid.unsqueeze(-1).to(delta.dtype)
        return h + delta

class LocalHead(nn.Module):
    def __init__(self, input_dim, text_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, text_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, h_prime):
        return F.normalize(self.proj(h_prime).float(), dim=-1, eps=1e-6)
```

`text_dim` đọc từ teacher output/checkpoint, không hard-code 512 cho mọi backbone. Một lựa chọn initialization khác là tích linear `conv1` và `visual.proj` của CiCo nếu dimensions khớp; vì bỏ Transformer/LN, đó chỉ là initialization heuristic, không tương đương encoder. Dùng một cách khởi tạo cố định ở tất cả controls.

### 7.2. Radius lớn hơn 0

Không dùng `Conv1d(kernel=3)` trên 64 uniformly sampled tokens rồi khẳng định receptive field là ba clip sát nhau: các token có thể cách xa trên timeline gốc.

Nếu bật `r>0`, áp adapter temporal trên dense sequence trước sampling, với kernel `2r+1`, không dilation, padding có mask; hoặc xây neighborhood theo dense index/time thật. Inference cũng phải làm như vậy. Khi mask evidence, feature bị can thiệp phải bị mask trước temporal adapter. Kiểm tra Jacobian/locality và mở rộng receptive field metadata theo neighborhood. Đây là extension; default spec chỉ triển khai `r=0`.

### 7.3. Frozen encoder và gradient

Teacher: `.eval()`, tất cả parameters `requires_grad=False`, forward trong `torch.no_grad()`.

Student warmup: CiCo parameters frozen; adapter và local head trainable. **Không đặt student visual forward trong `no_grad()`**, bởi vẫn cần backprop qua frozen Transformer tới adapter. Giữ frozen backbone ở `eval()` để dropout/batch-stat không vô tình thay đổi teacher–student comparison. `.eval()` không tắt autograd.

Nếu `retriever.train()` gọi đệ quy, override hoặc gọi lại `core.eval()` sau đó trong warmup. Khi mở block cuối, cấu hình train/eval từng module có chủ đích. Không gọi `model.eval()` tạm thời chỉ để né distributed all-gather rồi quên restore; bridge cần flag gather riêng.

Ở bước đầu, gradient `down.weight` có thể bằng zero do `up.weight=0`. Test phải kiểm tra `up.weight` nhận gradient ở bước đầu và `down.weight` sau một vài optimizer steps; không kết luận đứt gradient chỉ từ bước zero-init.

## 8. Token–word mapping và lexical bank

### 8.1. Canonical word units

MVP ưu tiên lexical units alphabetic không chứa khoảng trắng, xuất hiện đúng một lần trong caption. Loại BOS/EOS/punctuation và stopwords trong danh sách cố định có version. Numbers, named entities, multiword phrases, contractions phức tạp và từ lặp để extension/diagnostic riêng. Lựa chọn này giảm coverage; phải báo coverage, không viết method đã giải quyết mọi sign.

Nếu caption thật là tiếng Trung không tách từ, không sử dụng whitespace parser. Cần tokenizer/segmenter phù hợp được pin version và kiểm tra mapping; hoặc dùng chính caption translated đã có của baseline nếu protocol đó vốn dùng nó. Thay ngôn ngữ caption là thay setting.

### 8.2. Mapping đúng BPE

`word_offsets.py` phải:

1. Tạo canonical string bằng **đúng** `whitespace_clean(basic_clean(text)).lower()` của CiCo.
2. Dùng `SimpleTokenizer.pat.finditer(canonical)` để có regex span, rồi byte encoding và `bpe()` của upstream trên từng match.
3. Ghi mapping `canonical_char_span → full_bpe_indices` và IDs. Việc này chạy đúng pipeline tokenizer, không tokenize từng whitespace word rồi giả định offset tự khớp.
4. Thêm BOS/EOS, áp **đúng BPE subsampling/truncation của loader cấu hình đang dùng**. Ghi `selected_full_bpe_indices` và map sang positions trong encoded sequence.
5. Assert token IDs bằng upstream cho cùng caption và `max_words`.
6. Chỉ giữ target có toàn bộ BPE pieces còn tồn tại. Nếu một phần word bị subsample, bỏ auxiliary record, giữ sample trong coarse training.
7. Lưu offset trên canonical string. Nếu normalization đổi độ dài Unicode/HTML, không áp canonical offset vào raw string; giữ raw string chỉ để audit hoặc xây mapping riêng.

Negative caption được tạo bằng thay **một canonical character span** đã xác định; không dùng `str.replace()` toàn câu. Tokenize lại và kiểm tra target thực sự thay đổi. Với nhánh evidence, MVP bỏ các cặp positive/negative cần BPE subsampling: thay một word có thể làm `linspace` chọn lại các từ khác và phá tính chất minimal pair. Coarse branch vẫn xử lý mọi câu theo baseline.

Các tests cần có: từ nhiều BPE, chữ Unicode, HTML entity, punctuation, từ lặp, BOS/EOS, caption quá dài và replacement làm tăng số BPE. Không dùng equality của số từ whitespace để chứng minh chỉ một lexical unit thay đổi.

### 8.3. Lexical text bank cố định

Từ vocabulary train đủ điều kiện, tạo input chỉ chứa surface word với cùng tokenizer/BOS/EOS/padding. Lấy teacher text hidden states, trung bình **pieces thuộc word**, loại special/pad rồi L2 normalize:

\[
e(w)=\operatorname{norm}\left(\frac{1}{|B(w)|}\sum_{m\in B(w)}T^{teacher}(w)_m\right).
\]

Cache `lexical_bank.npy` `[V,d]` float32 và `vocab.json` `word_id→surface, language, occurrence_count, BPE IDs`. Lexical bank không được update theo student trong MVP và không cần khi inference. Không dùng full negative caption contextual embedding làm `e(u)` ở nhánh local.

Mean-BPE embedding là một lựa chọn representation, chưa chắc tốt hơn EOS embedding. Nếu thử EOS/fixed template thì coi là ablation, dùng cùng lựa chọn cho positives/negatives, không chọn từng word theo test.

## 9. Teacher support mining chỉ trên train

### 9.1. Hai temporal views

Teacher là snapshot baseline cố định, không EMA trong MVP. View A dùng deterministic sampling giống evaluation. View B dùng cùng số clips, mỗi selected dense index được jitter tối đa một dense step bằng RNG seed từ `(global_seed, video_id, 'view_b')`; clip index phải hợp lệ, tăng dần, không trùng. Với clip ngắn hoặc không tạo được view khác, đánh dấu `view_independent=false` và bỏ consistency-qualified auxiliary record, không gán agreement=1 một cách giả tạo.

Hai view cùng caption sạch, cùng feature fusion, không random word swap, không horizontal flip/reverse hoặc augmentation có thể đổi nghĩa sign. Đối với rất ngắn, vẫn coarse-train sample đó.

### 9.2. Score cho mỗi word

Lấy teacher visual tokens không CLS `y_n`, word embeddings contextual trong caption `c_k=mean_BPE(text_tokens[k])`, L2 normalize. Dùng riêng cho selector:

\[
M_{nk}=y_n^\top c_k,\quad
p_{nk}=\operatorname{softmax}_{k}(M_{nk}/\tau_a),\quad
q_{nk}=\operatorname{softmax}_{n}(M_{nk}/\tau_s).
\]

Chỉ normalize qua eligible word units hoặc valid temporal tokens tương ứng, không qua padding. Ghi tập word used để confidence có nghĩa xác định. `p` là proxy competition giữa từ; `q` là temporal support distribution; không gọi chúng là alignment posterior đã calibration. Teacher visual tokens vẫn contextual, do đó consistency không loại hết bias của teacher.

### 9.3. Quy tắc chọn support hoàn toàn xác định

`PROPOSED` defaults để chạy pilot, không phải SAN hyperparameters:

- `tau_a=tau_s=0.07`.
- Với mỗi word, tìm shortest contiguous interval trong selected timeline có tổng `q >= 0.60`.
- Tie-break: mass cao hơn, rồi start sớm hơn. Enumerate `O(F²)` được với `F=64`; không cần solver.
- Interval phải có ít nhất hai observed tokens và duration không quá 35% video. Nếu không có candidate hợp lệ thì abstain.
- Trong support, ít nhất một clip có word này là `argmax_k p[n,k]`.
- `confidence = sum_n a_n*p[n,k] >= 0.45`, với `a=q/sum_W q` trên support.
- Chọn độc lập ở view A/B; map intervals về timeline gốc. IoU thời gian tối thiểu 0.5.
- Span canonical `W` lấy từ view A; ghi cả hai spans để audit. Trọng số `a` được cache ở original dense indices view A, không ở index 0..63 không có provenance.
- Một caption có target word lặp hoặc nhiều support modes xa nhau: MVP abstain. Detect mode thứ hai bằng connected interval riêng có ≥30% mass ngoài W; ngưỡng này cần frozen config.

Khi feature RF metadata chưa có, có thể dùng dense index intervals cho ELSC-Min và ghi `coordinate_system='dense_index'`. Khi đó **không** báo alignment IoU bằng giây/frame hoặc chạy ELSC-Full như thể receptive field đã biết. Locality chỉ được khẳng định ở cấp raw input feature.

### 9.4. Reliability và coverage

Trước khi nhân confidence, eligibility là hard gate: word/BPE hợp lệ, đủ support, cross-view agreement, không lặp, không all-padding. Sau khi đếm training occurrences `c_w`, dùng:

\[
\rho=\operatorname{clip}_{[0,1]}\left(
IoU(W_A,W_B)\cdot confidence\cdot\min(1,c_w/5)\right).
\]

MVP yêu cầu ít nhất 3 **video IDs khác nhau** có reliable occurrence để word tham gia negative graph. `rho` detach/fixed. Thresholds có thể screening bằng dev nhưng không tối ưu theo test. Log số records trước/sau từng gate, số word được giữ, số word/video và phân bố signer nếu metadata sẵn có.

Nếu coverage thấp: kiểm tra token/mask/language bug trước. Không tự giảm mọi threshold cho tới khi đủ cache; thay một config, rebuild cache và chạy controls tương ứng. Một empty auxiliary batch trả differentiable zero, không tạo negative giả.

## 10. Negative candidates: hai chế độ rõ ràng

### 10.1. Chế độ A — official SAN artifact

Nếu nhận được negative table chính thức có provenance, cùng ngôn ngữ và lexical units, import vào schema dự án. Giữ source ID, source hash, mining checkpoint và filtering decisions. Nếu table dùng subword tokens, không tự nối thành word mà không kiểm tra.

ELSC và whole-caption control phải dùng **cùng final candidate cache sau filtering**, cùng target positions và seed. Dù source là SAN, subset sau ELSC reliability filtering vẫn cần nêu rõ. Thêm một hàng SAN original configuration độc lập nếu reproducing đầy đủ.

### 10.2. Chế độ B — train-only visual-neighbor fallback

Khi official artifact không có, implement miner sau để dự án vẫn thực thi được. Gọi là `train_visual_neighbors_v1`, **không** gọi đây là SAN official reproduction hoặc so trực tiếp với SAN stress-test score.

1. Từ reliable occurrence `(video,w,W,a)`, lấy raw **fused pre-adapter** feature prototype:
   `r_occ = normalize(sum_n a_n * normalize(h_n))`.
2. Mỗi word lấy tối đa 32 occurrence từ video IDs khác nhau, deterministic sampling. Tách ít nhất một occurrence/video, không để một video dài lấn át.
3. Tính cosine giữa occurrence của hai word khác nhau ở **khác video ID**. Với candidate word pair `(w,u)`, score bằng mean của 3 pair cosine lớn nhất có distinct source video IDs cho mỗi phía; nếu không đủ 3 pair thì bỏ.
4. Chọn tối đa 10 candidate words/word vượt `visual_cosine_min=0.70`; ngưỡng là proposed cho feature này, không thừa hưởng ý nghĩa calibration từ SAN.
5. Ưu tiên mutual top-10 để giảm isolated noisy matches. Nếu quá ít candidates, báo coverage; không fill bằng random trong chính mode này.
6. Optional POS/lemma filter dùng tagger có model/version/hash riêng. Nếu không có, đặt `pos_filter=false` và ghi rõ; không nói đã POS-match. Cache luôn ghi filter mode.

Để không tạo tensor `N_occ²` lớn, tính từng block cosine hoặc top-k exact search theo block, sau đó aggregate word pairs. Không cần FAISS dependency cho pilot PH. Nếu dùng ANN sau này, ghi index/search recall đối với exact subset và giữ same graph cho controls.

Định nghĩa lựa chọn ba pairs ở bước 3: sort decreasing cosine, tie-break bằng `(video_id_left, video_id_right)`; greedily nhận pair chỉ khi cả left/right video ID chưa được dùng trong danh sách đã chọn của word pair đó; dừng ở ba. Đây là thuật toán greedy được cố định, không gọi là tối ưu matching toàn cục.

**Per-occurrence negative filtering:**

- `u != w`; không cùng normalized surface; nếu có lemmatizer thì không cùng lemma.
- `u` không xuất hiện ở nơi khác trong caption gốc; replacement tạo một caption khác thật sự.
- Không cùng known equivalence/synonym group nếu có tài nguyên cố định được công khai. Không giả định khác word ID chắc chắn là semantic negative.
- Word u có đủ independent visual support trong train.
- Cap 5 negatives/target, tối đa 2 targets/video, ưu tiên rho cao rồi tie-break word position.
- Nếu chỉ có 1–4 negative hợp lệ, dùng số đó và mask phần padding, không duplicate để đủ 5.
- Với evidence loss, teacher phải xếp positive trước negative; điều kiện này nằm ở §12, không bắt buộc để local loss được chạy.

Đây là weak supervision có false-negative risk còn lại do polysemy/synonyms. Training exclusion không chứng minh candidate luôn sai. Report diagnostic samples và random-negative control; không dùng oracle gloss hoặc test annotations để sửa graph.

### 10.3. Cache schema cuối

```json
{
  "schema_version": 1,
  "split": "train",
  "pair_id": "...",
  "word_id": 17,
  "word_surface": "...",
  "canonical_char_span": [10, 15],
  "encoded_bpe_positions": [4, 5],
  "support_dense_indices": [21, 23, 25],
  "support_weights": [0.2, 0.5, 0.3],
  "support_interval": [20, 27],
  "coordinate_system": "dense_index",
  "view_iou": 0.75,
  "alignment_confidence": 0.6,
  "rho": 0.45,
  "negative_word_ids": [33, 64],
  "negative_captions": ["...", "..."],
  "negative_source": "train_visual_neighbors_v1",
  "evidence_eligible": false
}
```

Ví dụ số ở trên chỉ minh họa schema. `cache_meta.json` phải chứa teacher/checkpoint hash, config hash, manifest hash, tokenizer/BPE hash, language, feature fusion, view sampling, mining version, negative table hash và lexical bank hash. Loader từ chối cache khi metadata không khớp; không “best effort” map theo thứ tự sample.

## 11. Local lexical loss

Với occurrence `o=(v,w,W)` và negative `u`, tính:

\[
\ell_W(v,w)=\sum_{n\in W}a_n z_n^\top e(w),\quad
\delta_{ou}=\ell_W(v,w)-\ell_W(v,u),
\]

\[
L_{lex}=\frac1{N_o}\sum_o\rho_o\log\left[
1+\sum_{u\in H(o)}\exp\left(\frac{m_{lex}-\delta_{ou}}{\tau_{lex}}\right)\right].
\]

`N_o` là số eligible occurrences trong global batch; không phải số video toàn dataset, cũng không phải `sum(rho)`. Nhân rho rồi chia số occurrence để absolute reliability có tác động. Nếu đổi sang normalize theo `sum(rho)`, coi là objective khác và ghi config.

```python
# NEW: inputs đã gather theo occurrence; score accumulations float32.
def lexical_loss(z, weights, pos_e, neg_e, neg_valid, rho,
                 margin=0.1, temperature=0.07):
    # z [O,F,d], weights [O,F], pos_e [O,d]
    # neg_e [O,K,d], neg_valid [O,K], rho [O]
    pooled = (weights.detach().float().unsqueeze(-1) * z.float()).sum(1)
    # Không L2-normalize pooled: công thức là weighted sum của dot products.
    positive = (pooled * pos_e.detach().float()).sum(-1)
    negative = torch.einsum('od,okd->ok', pooled, neg_e.detach().float())
    logits = (margin - positive[:, None] + negative) / temperature
    logits = logits.masked_fill(~neg_valid, float('-inf'))
    zeros = logits.new_zeros((logits.size(0), 1))
    per_occ = torch.logsumexp(torch.cat([zeros, logits], dim=1), dim=1)
    eligible = neg_valid.any(-1)
    numerator = (rho.detach() * per_occ * eligible).sum()
    denominator = eligible.sum()
    # Trainer chịu trách nhiệm global normalization khi DDP.
    return numerator, denominator
```

Empty occurrence case phải được xử lý trước tensor construction: trả `0.0 * h_prime.sum()` và count=0 để graph hợp lệ. Padding weights bằng zero, weights của mỗi eligible occurrence sum=1, mọi selected dense indices có trong view.

Default local view là canonical deterministic view A đã dùng cache. Coarse branch vẫn dùng preprocessing/augmentation baseline. Reuse forward khi input features/mask thật sự giống nhau; text augmentation có thể khiến text forward khác nhưng video forward vẫn reuse được. Không map cache sang randomly sampled view bằng cách giữ nguyên position 0..63.

Nếu cho phép remap sang view mới, chỉ lấy intersection dense IDs, yêu cầu retained support mass ≥0.8, renormalize retained weights rồi ghi coverage; mặc định tắt để tránh một biến chưa kiểm chứng.

## 12. ELSC-Full: evidence dependence và control invariance

Chỉ bật sau khi ELSC-Min vượt kiểm tra local-vs-random và có dev signal. Phiên bản đầy đủ vẫn inference bằng CiCo score như cũ.

### 12.1. Định nghĩa margin và scale

Với positive caption `t`, negative chỉ thay một target word `t_minus`, dùng **mixed CiCo score của clean text**. Để margin có đơn vị ổn định, đặt:

\[
\bar S(v,t)=S(v,t)/\exp(\texttt{logit\_scale}),\qquad
\Delta(v)=\bar S(v,t)-\bar S(v,t^-).
\]

Giữ `logit_scale` frozen trong MVP/Full pilot. Không chia cho temperature lần nữa. Token-interaction temperature 0.07 vẫn nằm trong scorer. Coarse loss dùng scaled logits như upstream, không thay bằng `bar S`.

Teacher evidence gate: `Delta_teacher(clean) >= 0.02`, rho hợp lệ và cả positive/negative không bị BPE subsampling. `0.02` là default đề xuất; threshold được cố định trong cache/config. Teacher gate detach, không recompute theo student để student điều khiển sample selection.

### 12.2. Mask raw feature đúng receptive field

ELSC-Full cần temporal metadata từ extraction recipe hoặc archive đã xác minh. Với mỗi raw clip token `n`, biết input frame interval `R_n`. Không suy stride/window size từ `L` hoặc `F=64`. Lưu preprocessing FPS, crop/re-alignment mapping và padding/clamp ở đầu/cuối.

Chuyển support W sang frame/time interval thật. Mask mọi input feature có `R_n ∩ W != empty`. Nếu hai stream I3D có receptive fields khác nhau, dùng union metadata trước fusion hoặc mask cả hai stream tương ứng. Không chỉ mask những tokens teacher đã chọn mà bỏ clip chồng lấn chứa cùng evidence.

Mask thực hiện **trước adapter và Transformer**, rồi chạy forward lại:

```python
def apply_input_intervention(h, remove, valid, fill):
    assert not bool((remove & ~valid).any())
    # fill: [D] từ train mean hoặc scalar zero; cố định, không trainable.
    return torch.where(remove.unsqueeze(-1), fill.to(h).view(1, 1, -1), h)
```

Nếu dùng zero scalar, chuẩn hóa nó thành vector `[D]` trước hàm trên. `valid` và upstream attention/pooling mask **không đổi** giữa clean/W/C: đây là can thiệp giá trị feature, không giảm sequence length. Như vậy không đổi denominator và vị trí positional embeddings. Padding vốn có vẫn là padding. Một mask token trainable có thể tạo shortcut; không dùng trong MVP.

Default fill=`zero`; alternative train-feature mean tính trên train valid tokens trước adapter, cùng fusion space, là một ablation. Mỗi operator được dùng giống nhau cho evidence và control. Không mask hidden state sau Transformer: attention đã truyền thông tin sang tokens khác.

### 12.3. Chọn control C

Cho mỗi evidence window W, enumerate contiguous control intervals trên cùng timeline:

- Không giao W và không giao support high-confidence của các target/content words khác mà teacher phát hiện.
- Sau receptive-field closure, số input tokens bị mask **bằng** số tokens bị mask cho W.
- Duration control lệch tối đa 10% so với evidence interval; có thể tăng tolerance trong một config riêng, không âm thầm thay mỗi batch.
- Control không all-padding, không chứa replacement word support khác nếu có.
- Nếu nhiều candidate, chọn bằng deterministic hash `(seed,pair_id,target_id)`; không chọn nơi làm student loss nhỏ nhất.
- Nếu không có control thỏa điều kiện, bỏ `L_dep/L_inv` cho record, vẫn giữ `L_lex` nếu hợp lệ.

Việc teacher không phát hiện support ở C không chứng minh C vô nghĩa. Đây là matched intervention control dựa trên pseudo-support, không phải bằng chứng nhân quả hoàn chỉnh. Log số tokens, duration, location và eligibility rate của W/C.

### 12.4. Hai losses

\[
L_{dep}=\operatorname{mean}_o\rho_o
\max\{0,m_{dep}-[\Delta(v_o)-\Delta(v_o\setminus W_o)]\},
\]

\[
L_{inv}=\operatorname{mean}_o\rho_o\,
Huber_{\delta_h}[\Delta(v_o)-\Delta(v_o\setminus C_o)].
\]

Với `Huber_delta(x) = 0.5*x²` nếu `|x|<=delta`, ngược lại `delta*(|x|-0.5*delta)`. PyTorch `smooth_l1_loss(beta=delta)` chia Huber cho delta; **không** dùng thay thế mà giữ cùng lambda như thể scale giống nhau. Có thể dùng `F.huber_loss(..., delta=delta, reduction='none')` trong runtime hỗ trợ.

`PROPOSED`: `m_dep=0.02`, `delta_h=0.05`, `lambda_dep=0.1`, `lambda_inv=0.1`. Vì Huber có scale nhỏ hơn hinge, theo dõi cả giá trị và gradient contributions; không suy hệ số bằng nhau nghĩa là ảnh hưởng bằng nhau.

Default tối đa 1 evidence pair/video và 25% video trong batch có đủ điều kiện, chọn deterministic RNG. Tất cả lexical negatives vẫn có thể dùng ở `L_lex`; nhánh evidence chỉ chọn một negative theo cùng cache/seed. Khi clean video giống nhau có thể reuse output; mỗi W/C variant phải encode lại. Không tạo tensor all-pairs giữa mọi masked video và mọi caption khi chỉ cần score own positive/negative.

### 12.5. L_keep tùy chọn

Dùng cùng **clean training batch candidates** cho teacher/student. Từ mixed scaled score `S`, tạo softmax theo text candidates cho V2T và theo video candidates cho T2V:

\[
L_{keep}=\frac{T_d^2}{2}\left[
\operatorname{mean}_v KL(p_T(t|v)\Vert p_S(t|v))+
\operatorname{mean}_t KL(p_T(v|t)\Vert p_S(v|t))\right].
\]

`p=softmax(S/T_d)`, default `T_d=1.0`; teacher distribution detach. Với `F.kl_div`, input phải là student `log_softmax`, target là teacher probabilities, direction `teacher || student`. Không dùng test-gallery teacher predictions hoặc GT-informed test features.

`lambda_keep=0` ở Min. Chỉ thử `0.05` rồi `0.1` nếu có CG degradation và so với giảm learning rate/early stopping. KL không bảo đảm bảo toàn R@1 ở near ties. Khi DDP, teacher/student phải có cùng global candidate IDs/order.

### 12.6. Objective cuối và cấu hình method

\[
L=L_{CLCL}+\lambda_{lex}L_{lex}
+\lambda_{dep}L_{dep}+\lambda_{inv}L_{inv}
+\lambda_{keep}L_{keep}.
\]

ELSC-Min dùng `CLCL + L_lex`; ELSC-Full thêm `L_dep/L_inv`, thử keep riêng. Mặc định **không cộng thêm whole-caption SAN fine loss** vào ELSC, để đo đóng góp thay supervision. Có thể thêm hàng `SAN + ELSC` sau đó, ghi thành một ablation khác. Tất cả có cùng coarse objective.

## 13. Whole-caption negative control cần implement

Để so trực tiếp local objective với sentence objective, tạo control nhận cùng `t_minus`, cùng count, cùng reliable target subset và cùng adapter capacity:

\[
L_{caption}=\frac1{N_o}\sum_o\rho_o
\log\left[1+\sum_{u\in H(o)}
\exp((m_{cap}-[\bar S(v,t)-\bar S(v,t^-_u)])/\tau_{cap})\right].
\]

MVP `m_cap=0`, `tau_cap=0.07`. Đây là **matched whole-caption control**, không tự nhận chính xác SAN Eq.10 reproduction vì đã match reliability/single-substitution/mixed-score choices của ELSC. Hàng SAN gốc phải dùng original scoring/loss settings và hai substitutions, năm negatives theo paper/config, được report riêng.

Có ba mức so sánh khác nhau phải ghi đúng tên:

1. `SAN official/reproduction`: chỉ khi đủ official artifact/protocol và fidelity đã kiểm tra.
2. `Matched caption contrast`: cùng negative cache, sampler, adapter và rho với ELSC.
3. `Random lexical negatives`: cùng số negative, filter ngôn ngữ/word-frequency khi có thể, nhưng random graph; dùng để kiểm tra sign-aware candidate có vai trò hay không.

Không giả định chỉ thắng matched control thì đã thắng toàn bộ SAN reported results.

## 14. Training pipeline và API cần viết

### 14.1. Các stage

| Stage | Trainable | Loss | Kết quả cần có |
|---|---|---|---|
| P0 parity | Không train | Không | Bridge tương đương baseline cho inputs giống nhau. |
| B0 baseline | Theo baseline config đã khóa | CLCL | Checkpoint chọn bằng dev; teacher provenance. |
| C0 cache | Không train | Không | Lexical bank, support cache, negative graph train-only. |
| M0 warmup | Adapter + local head | CLCL + ramped lexical | Gradient flow đúng; local loss không chỉ giảm ở head. |
| M1 ELSC-Min | Adapter + head; backbone frozen mặc định | CLCL + lexical | Dev retrieval, controls, 3 seeds khi có signal. |
| F0 ELSC-Full | Như M1 | Thêm dependence/invariance | Chỉ bật khi có RF metadata và M1 có signal. |
| F1 optional unfreeze | Adapter/head + last visual blocks | Cùng objective | Matched baseline cũng được unfreeze bằng cùng schedule. |

Default pilot ELSC adapter training: tối đa 20 epochs, AdamW, adapter LR `1e-4`, head LR `1e-4`, weight decay `1e-2`; warmup LR 10% optimizer steps; cosine decay; gradient clip norm 1.0. Lambda lexical ramp tuyến tính từ 0 đến 0.1 trong epoch đầu. Đây là **starting recipe**, không phải recipe đã đo GPU hours hoặc thắng paper.

ELSC-Full bắt đầu từ checkpoint Min đã chọn dev; để tách continuation effect, baseline/Min control cũng được train tiếp cùng số steps. Không so Full sau thêm 10 epochs với Min đã dừng sớm rồi quy toàn bộ gain cho evidence losses.

### 14.2. Một training step

```python
# Pseudocode NEW; các helper phải được implement theo contracts bên trên.
batch = next(train_loader)
assert batch.split == 'train'
optimizer.zero_grad(set_to_none=True)  # nếu accumulation, ở đầu window

h_prime = adapter(batch.h, batch.valid)
# Student visual forward phải giữ autograd dù core frozen.
v = bridge.encode_video(h_prime, batch.valid)
t_clean = bridge.encode_text(*batch.clean_text)
t_aug = bridge.encode_text(*batch.aug_text)

A, B = score_coarse_with_exact_upstream_semantics(v, t_clean, t_aug)
L_coarse = upstream_balanced_clcl(A, B, dual_mix=cfg.dual_mix)

aux = cache.lookup(batch.pair_ids, view_hash=batch.view_hash)
# Nếu batch visual view khác canonical cache: encode canonical aux view riêng.
z = local_head(h_prime_for_aux)
lex_sum, lex_count = lexical_loss_from_cache(z, aux)
L_lex = globally_normalized_auxiliary(lex_sum, lex_count)

L_dep = L_inv = L_keep = h_prime.sum() * 0.0
if cfg.evidence.enabled:
    selected = select_evidence_records(aux, fixed_seed_state)
    h_W = apply_input_intervention(h_aux, selected.W, valid_aux, fill)
    h_C = apply_input_intervention(h_aux, selected.C, valid_aux, fill)
    v_W = bridge.encode_video(adapter(h_W, valid_aux), valid_aux)
    v_C = bridge.encode_video(adapter(h_C, valid_aux), valid_aux)
    # paired_score dùng cùng token reductions/mix; không all-gather mỗi pair.
    delta_clean = paired_margin(v_clean_aux, t_pos, t_neg)
    delta_W = paired_margin(v_W, t_pos, t_neg)
    delta_C = paired_margin(v_C, t_pos, t_neg)
    L_dep, L_inv = evidence_losses(delta_clean, delta_W, delta_C, selected)

if cfg.keep.weight > 0:
    with torch.no_grad():
        teacher_scores = teacher_clean_batch_scores(batch)
    student_scores = student_clean_batch_scores_from_cached_tokens(v, t_clean)
    L_keep = bidirectional_kl(teacher_scores, student_scores)

loss = (L_coarse + w_lex * L_lex + w_dep * L_dep
        + w_inv * L_inv + w_keep * L_keep)
backward_with_amp_if_enabled(loss)
# unscale gradients trước clip; step/update scaler ở cuối accumulation window.
optimizer_step_and_log()
```

Đoạn này mô tả flow; `h_prime_for_aux`, `h_aux`, `v_clean_aux`, `t_pos`, `t_neg` phải được lấy từ canonical aux-view batch do `cache_dataset.py` dựng, không là biến global hoặc đọc test. `paired_score` phải có parity với diagonal/selected entries của all-pairs scorer, gồm masks, mix và logit scaling.

### 14.3. DDP và effective batch

MVP bắt đầu một GPU để xác minh math; chuyển DDP sau khi correctness ổn. Original coarse scorer có all-gather phụ thuộc `self.training and self.distributed`. Bridge cần gọi scorer với chế độ gather rõ ràng cho coarse, paired auxiliary và evaluation; không để một masked forward vô tình all-gather không đều rồi deadlock.

Nếu coarse loss đã gather global logits, không gather lần hai. Kiểm tra upstream `allgather` autograd và objective scaling với một-GPU reference có cùng global batch. Gradient accumulation tăng effective optimizer batch nhưng **không tự tăng contrastive negative pool**. Không viết batch 512 nếu mỗi forward chỉ contrast 64 samples rồi accumulate tám lần mà không giải thích.

Vì frozen core được giữ `eval()`, original automatic gather có thể không chạy. Bridge phải gather explicit theo `training_objective=true` ở wrapper hoặc tách score kernel ra khỏi orchestration, giữ nguyên math và có parity test; không dựa vào `core.training` làm signal duy nhất. Single-GPU và paired auxiliary đặt gather=false.

Với auxiliary records không đều giữa ranks: global mean không phải mean của local means. Nếu DDP mặc định average parameter gradients, mỗi rank dùng:

```text
N_global = all_reduce_sum(local_eligible_count)  # detached count
local_backward_loss = world_size * local_weighted_loss_sum / max(N_global,1)
```

DDP average gradients sau đó tạo gradient của global sum/global count. `world_size` factor này dành cho local-only auxiliary loss, không tự áp vào coarse loss đã global-gather. Rank không có record vẫn tham gia collective và có differentiable zero qua parameters cần thiết; kiểm tra `find_unused_parameters`/dummy graph theo runtime, tránh training bị treo.

AMP: token similarities, normalization, logsumexp, KL và Huber accumulate float32. Khi upstream cố cast theo CLIP dtype, giữ dtype parity có chủ đích; không half-cast lexical bank rồi hy vọng không underflow. GradScaler, clip, optimizer và scheduler chạy theo optimizer steps, không minibatch count khi accumulation.

### 14.4. Checkpoint và log

Checkpoint chứa adapter/head/core state cần thiết, optimizer, scheduler, AMP scaler, epoch/step, RNG Python/NumPy/Torch/CUDA, sampler state, config/manifest/cache hashes, upstream SHA, initialization/teacher hash, trainable parameter names và dev selection rule. Final inference export bỏ teacher/head/cache nhưng giữ adapter và required core weights.

Log ít nhất: coarse/lex/dep/inv/keep riêng; eligible counts; rho; margin sạch/W/C; gradient norms adapter output/lower layer/local head; LR; feature residual norm `||A(h)||/||h||`; dev T2V/V2T R@1/5/10; elapsed time; peak memory; actual global negative pool.

Gradient diagnostic để phát hiện head hấp thụ loss: đo norm từ riêng `L_lex` vào adapter ở một diagnostic batch và so với head. Không chạy `autograd.grad` mọi step nếu tốn; chạy định kỳ. Adapter frozen + head trainable là control bắt buộc để kiểm tra khả năng loss giảm mà retrieval không đổi.

## 15. Config mẫu có validation

Các path dưới đây phải thay bằng artifact thật. `required` có nghĩa script phải fail-fast nếu chưa được cung cấp, không tự tải dữ liệu khác.

```yaml
# NEW: methods/elsc/configs/ph_min.yaml — proposed defaults, chưa được train/tune
schema_version: 1
experiment: ph_elsc_min
seed: 42

upstream:
  cico_root: third_party/SLRT/CiCo/CLCL
  cico_commit: 38a4f7b00da7a858d59b7fabe5093876a84db8e0
  compatibility_patch: patches/cico_compat.patch

sources:
  annotation_format: cico_pickle_or_explicit_official_conversion
  train_annotation: required_source_train_annotation
  dev_annotation: required_source_dev_annotation
  test_annotation: required_source_test_annotation
  feature_agnostic_root: required_feature_agnostic_root
  feature_aware_root: required_feature_aware_root
  temporal_metadata_root: null  # Min có thể dùng dense indices; Full không được null

data:
  dataset: ph
  train_manifest: artifacts/manifests/ph_train.jsonl
  dev_manifest: artifacts/manifests/ph_dev.jsonl
  test_manifest: artifacts/manifests/ph_test.jsonl
  caption_language: required
  feature_dim: 1024
  feature_len: 64
  max_words: required_from_baseline_config
  combine_type: sum
  alpha: 0.9                 # weight of AGNOSTIC in audited PH eval loader
  feature_path_mode: corrected
  sampling: upstream_uniform
  text_augmentation: cico_random_swap_v1

model:
  init_checkpoint: required
  teacher_checkpoint: required
  teacher_selection_provenance: required
  sim_header: Filip
  dual_mix: 0.5
  mix_design: balance
  backbone_frozen: true
  logit_scale_frozen: true
  adapter:
    enabled: true
    radius: 0
    hidden_dim: 256
    zero_init_output: true
  lexical_head:
    output_dim: infer_from_teacher
    initialization: orthogonal

cache:
  path: artifacts/cache/ph_train_elsc_v1
  require_hash_match: true
  require_train_only: true
  local_view: canonical
  allow_support_remap: false
  negatives_source: train_visual_neighbors_v1
  max_targets_per_video: 2
  max_negatives_per_target: 5

mining:
  tau_word: 0.07
  tau_time: 0.07
  mass_min: 0.60
  min_support_tokens: 2
  max_duration_fraction: 0.35
  confidence_min: 0.45
  view_iou_min: 0.50
  min_distinct_videos_per_word: 3
  max_occurrences_per_word: 32
  visual_cosine_min: 0.70
  graph_top_k: 10
  mutual_neighbors: true
  pos_filter: false

loss:
  lexical_weight: 0.1
  lexical_margin: 0.1
  lexical_temperature: 0.07
  lexical_weight_ramp_epochs: 1

evidence:
  enabled: false
  require_verified_rf_metadata: true
  max_pairs_per_video: 1
  batch_fraction: 0.25
  teacher_margin_min: 0.02
  dependence_margin: 0.02
  dependence_weight: 0.1
  invariance_weight: 0.1
  huber_delta: 0.05
  mask_fill: zero
  control_same_token_count: true
  control_duration_tolerance: 0.10

keep:
  weight: 0.0
  temperature: 1.0

train:
  epochs: 20
  per_device_batch: required_from_memory_and_baseline
  accumulation_steps: 1
  optimizer: adamw
  adapter_lr: 0.0001
  head_lr: 0.0001
  weight_decay: 0.01
  warmup_ratio: 0.10
  schedule: cosine
  grad_clip_norm: 1.0
  precision: fp32_initial_parity_then_audited_amp
  eval_split: dev
  checkpoint_metric: mean_t2v_v2t_r1

evaluation:
  scorer: cico_mixed_token_interaction
  metrics: upstream_cico
  full_gallery: true
  filter_by_aux_eligibility: false
  fine_grained_san: disabled_until_official_artifacts
```

Config validator phải reject: `test` làm training selection, radius>0 với sampled-index neighborhood chưa khai báo, Full thiếu RF metadata, unresolved `required*`, negative language mismatch, all trainable parameters frozen, cache teacher hash khác checkpoint.

Không đưa `max_words`, batch size hoặc backbone fine-tuning LR giả là “config paper”. Đọc run config/checkpoint artifacts thực tế; nếu không có thì đặt proposed value và report khác biệt. Đối với B0, config baseline gốc và budget phải được ghi riêng; config trên là adapter fine-tuning pilot.

Tạo `ph_base.yaml` từ cùng resolved data/scorer configuration, đặt `adapter.enabled=false`, tất cả auxiliary weights=0, không yêu cầu teacher/cache; thay optimizer/trainable parameters/schedule bằng baseline training recipe đã chọn và ghi provenance. `audit assets` chỉ validate nguồn dữ liệu/checkpoint cần cho stage đó; không bắt cache tồn tại trước `build_cache`.

Các YAML dẫn xuất cần viết, ví dụ:

```yaml
# NEW: methods/elsc/configs/ph_full.yaml
extends: ph_min.yaml
experiment: ph_elsc_full
model:
  init_checkpoint: required_selected_min_checkpoint
evidence:
  enabled: true
sources:
  temporal_metadata_root: required_verified_temporal_metadata
```

Full khởi tạo student từ Min nhưng **teacher/cache vẫn giữ baseline cố định**, không tự thay teacher bằng Min vì `init_checkpoint` đã đổi. Cache metadata tách `teacher_hash` với `student_initialization_hash`.

Trong `ablation_caption.yaml`, đặt `method: matched_caption`, `loss.lexical_weight: 0`, thêm `caption.weight: 0.1`, `caption.margin: 0`, `caption.temperature: 0.07`; giữ adapter và cache. Trong `ablation_random_span.yaml`, giữ lexical loss, đặt `aux_support_mode: random_matched`. Trainer phải dispatch các mode này rõ ràng và validate field; không chỉ đổi filename YAML mà chạy cùng method.

## 16. CLI end-to-end AI cần triển khai

Mỗi module CLI dùng `argparse` hoặc framework nhất quán, có `--help`, exit code khác 0 khi gate fail. Đây là các **lệnh mới sau khi code đã được viết**, không phải lệnh có sẵn của CiCo/SAN.

```bash
# 1. Audit dependency, checkpoint và dataset artifacts; chưa train.
python -m elsc.audit --config configs/ph_base.yaml --stage assets

# 2. Xuất manifest giữ nguyên official IDs/splits và validation reports.
python -m elsc.prepare --config configs/ph_base.yaml --splits train dev test

# 3. Kiểm tra bridge/metric parity trên deterministic fixture, ưu tiên dev.
python -m elsc.audit --config configs/ph_base.yaml --stage parity

# 4. Baseline controlled training, chọn bằng dev.
python -m elsc.train --config configs/ph_base.yaml --run-dir runs/ph_base_s42

# 5. Khóa teacher từ checkpoint có provenance; cập nhật YAML path/hash.
python -m elsc.audit --config methods/elsc/configs/ph_min.yaml --stage teacher

# 6. Tạo support cache, lexical bank và negative graph chỉ trên TRAIN.
python -m elsc.mining.build_cache --config methods/elsc/configs/ph_min.yaml --split train

# 7. Structural, gradient và score correctness checks.
python -m pytest tests/test_word_offsets.py tests/test_support_mapping.py
python -m pytest tests/test_losses.py tests/test_gradient_flow.py
python -m pytest tests/test_bridge_parity.py tests/test_evaluation_contract.py

# 8. Pilot ELSC-Min và matched controls.
python -m elsc.train --config methods/elsc/configs/ph_min.yaml --run-dir runs/ph_min_s42
python -m elsc.train --config methods/elsc/configs/ablation_caption.yaml --run-dir runs/ph_caption_s42
python -m elsc.train --config methods/elsc/configs/ablation_random_span.yaml --run-dir runs/ph_random_span_s42

# 9. Full chỉ sau gate cơ chế và RF metadata.
python -m elsc.train --config methods/elsc/configs/ph_full.yaml --run-dir runs/ph_full_s42

# 10. Chọn checkpoint theo dev; đánh giá test bằng lệnh tách biệt.
python -m elsc.evaluate --run-dir runs/ph_min_s42 --split test --checkpoint best_dev

# 11. Export inference: core + adapter + tokenizer/config, bỏ training-only heads.
python -m elsc.export --run-dir runs/ph_min_s42 --checkpoint best_dev --output exports/ph_min
```

`prepare --splits train dev test` chỉ xây manifest/check integrity, không mining hay chọn hyperparameters từ test. Training/cache code chỉ nhận train records. Test lock là artifact kỹ thuật của workflow, không phải yêu cầu người dùng xác nhận mỗi lần: evaluator xác minh checkpoint/config selection đã đóng băng và ghi run metadata.

CLI contracts quan trọng:

- `audit --stage assets`: xuất `asset_audit.json`, liệt kê hash, missing files, shapes, language và status; không tải asset không rõ nguồn để lấp chỗ trống.
- `prepare`: xuất manifests và ordered ID mapping; sample missing feature làm fail/explicit report, không im lặng bỏ query.
- `build_cache`: refuse split khác train; artifact reuse chỉ khi toàn bộ hashes khớp.
- `train`: đọc cache/config; resume khôi phục cả RNG/schedule; chỉ dev evaluator được gọi.
- `evaluate`: lưu score matrix hoặc block-backed file, query/candidate IDs, metrics, per-query ranks và tie stats. Không synthesize SAN candidates ở đây.
- `export`: reload exported model và kiểm tra score parity với training checkpoint khi inference-only, không cần teacher/cache.

Nếu cần dùng lệnh evaluation CiCo gốc để tham chiếu, xem README §Testing, ví dụ PH:

```bash
# UPSTREAM command; cần assets, environment và các compatibility fixes đã audit.
# Chạy trong third_party/SLRT/CiCo/CLCL.
python -m torch.distributed.launch --nproc_per_node=1 main_task_retrieval.py \
  --do_eval --init_model chpt/ph_sota.pth --data_path data_ph \
  --alpha 0.9 --datatype ph \
  --features_path sign_feature/ph_domain_agnostic \
  --features_path_retrain sign_feature/ph_domain_aware
```

Không chạy lệnh historical test này lặp lại để tune ELSC. Dùng dev fixture/component parity trước, và giữ test final riêng.

## 17. Evaluation, chọn mô hình và tiêu chí go/no-go

### 17.1. Full-gallery retrieval là kết quả chính

Ở inference:

1. Load core + adapter; không load teacher, lexical bank, negative table, support cache hoặc ground-truth spans.
2. Encode mọi video candidate/query độc lập bằng feature pipeline cố định và adapter.
3. Encode mọi text bằng tokenizer/text encoder của baseline.
4. Tính token-interaction score CiCo theo block, mix bằng `dual_mix` đã khóa.
5. Rank toàn gallery gốc cho cả T2V/V2T; dùng positive mapping đúng release.

Không cho selector nhìn caption GT để chọn W ở test. Không shortlist chứa đáp án được oracle chọn. Không đổi thành nearest neighbor của pooled features để tăng tốc mà vẫn gọi score CiCo. Feature caching theo video/text độc lập được phép và giống baseline.

Block scoring: tránh materialize tensor `[N_v,N_t,F,T]` cho toàn gallery. Chia `video_block`, `text_block`; normalize/reduce giống upstream trong từng block rồi lưu `[N_v,N_t]`. Block size là quyết định memory, phải có parity test với một batch không chia block.

Giữ R@1/5/10, MedR/MeanR theo official evaluator; dùng MRR chỉ nếu protocol/analysis ghi rõ. R@1 tính theo phần trăm hay fraction phải ghi trong schema; SAN code và CiCo code có thể khác đơn vị. Không trộn 0.70 với 70.0.

### 17.2. Dev selection cố định

Proposed primary selection metric: trung bình `(R1_T2V+R1_V2T)/2` trên dev. Tie-break: mean R@5, rồi epoch sớm hơn. Nếu muốn dùng T2V R@1 giống một baseline thì chọn trước và dùng cho mọi control. Không mỗi method chọn hướng thuận lợi riêng.

Screen với seed 42 trước; chỉ khi correctness/initial signal đạt mới chạy cùng shortlist config với seeds `42,1337,2026`. Không chạy mọi tổ hợp margin/threshold/lambda/backbone. Sau dev selection, lock toàn bộ config, caches và checkpoint rồi chạy test cho bộ run đã chọn. Báo mean±std giữa seeds; không chọn seed tốt nhất test.

Go/no-go đề xuất cho pilot:

- **Gate P:** bridge/metric/mask/gradient tests đạt. Nếu fail, không diễn giải retrieval changes là gain của ELSC.
- **Gate M:** true support tốt hơn random support và matched caption control trên dev ở ít nhất hai seeds khi mở rộng; nếu chỉ local loss giảm, chưa đạt.
- **Gate G:** mục tiêu thực dụng là mean dev R@1 tăng khoảng ≥0.5 điểm phần trăm, không một chiều giảm quá 0.5 điểm; threshold này chỉ giúp quyết định đầu tư, không phải significance test hoặc bảo đảm test gain.
- **Gate F:** chỉ thêm Full khi RF metadata đúng và đủ eligible controls. Nếu Full chỉ làm tăng synthetic margin nhưng không giữ/improve gallery retrieval, dừng Full hoặc report trade-off.
- **Gate X:** kiểm tra How2Sign sớm sau PH để tránh tối ưu weather-domain. CSL-Daily là kiểm chứng tiếp theo; không tự nhận generality từ một dataset.

Paired bootstrap theo query/video IDs để ước lượng CI chênh lệch với baseline, giữ pairing giữa methods. Nếu một video có nhiều captions, resample theo group video; không giả độc lập từng caption. Seed std và bootstrap CI đo các nguồn variability khác nhau, báo rõ. Không dùng CI test làm feedback tuning.

### 17.3. SAN fine-grained evaluation

Paper SAN đánh giá caption gốc với 40 synthetic negatives từ bốn nguồn, khác full-gallery test. [SAN paper](https://aclanthology.org/2026.acl-long.1302/).

Chỉ bật `evaluation/san_eval.py` khi có official candidate manifest/evaluator hoặc reproduction được đối chiếu đầy đủ: query IDs, original caption, ordered negatives, substitution policy, eligibility, duplicate/tie handling, generator/model versions, seed và candidate count.

`san_eval.py` trước hết là **adapter đọc protocol artifact**, không phải nhiệm vụ tự tạo một stress-test mới. Nếu artifact chưa có, trả status `official_artifact_missing` và chỉ chạy CG. Không điền FG=0, không so custom dev negatives với SAN R@1=39.4/49.1. Việc tạo training negatives ở §10 không giải quyết thiếu FG evaluation artifact.

## 18. Ablations tối thiểu để bảo vệ contribution

| ID | Thí nghiệm | Yếu tố phải match | Câu hỏi |
|---|---|---|---|
| A0 | Corrected baseline CLCL | Assets, patches, split, init | Baseline có tái lập ổn không? |
| A1 | Adapter + CLCL, không auxiliary | Adapter, steps, optimizer | Gain có chỉ từ capacity/fine-tuning? |
| A2 | Matched whole-caption contrast | Negatives, targets, rho, adapter, steps | Local placement có giá trị hơn sentence loss? |
| A3 | ELSC-Min true support | Same base/negative cache | Method tối thiểu. |
| A4 | ELSC-Min random support | Support count/duration/weights distribution, same targets | Teacher localization có thực sự hữu ích? |
| A5 | ELSC-Min shuffled lexical association | Same features/capacity/count; shuffle target assignment trong train | Có học đúng visual–word association? |
| A6 | ELSC với random lexical neighbors | Same K, filtering và budget | Visual confusability có vai trò? |
| A7 | Head-only, adapter frozen | Same local loss/data | Local loss có giảm mà retrieval không đổi? |
| A8 | Full vs continued Min | Same initialization và extra steps/forward budget | Evidence constraints thêm gì? |
| A9 | Full chỉ dep / chỉ inv | Same masks/eligible records | Thành phần nào cần thiết? |
| A10 | Keep vs giảm LR/early stopping | Comparable dev search budget | KL có tốt hơn regularization đơn giản? |

A4 không chọn random span làm thay đổi toàn bộ distribution: match số tokens/duration; chuyển weight vector theo thứ tự tương đối và renormalize, giữ rho để kiểm tra vị trí support. Có thể thêm random subset control để kiểm tra riêng reliability selection; không gọi nó là cùng ablation A4.

A5 là diagnostic, không phải dataset mới. Nếu shuffled targets không làm performance khác đáng kể, giả thuyết lexical supervision yếu; phải kiểm tra head absorption và caption artifacts trước khi viết novelty claim.

Để so generic local alignment, thêm một control local word–video contrast không dùng confusing-pair margin, cùng support/adapter/train budget. Nếu không hơn, contribution có thể chỉ là áp dụng local alignment đã có; cần đối chiếu prior art như [SPARC](https://arxiv.org/abs/2401.09865) và [SignCL](https://github.com/JinhuiYE/SignCL), không coi từng primitive là mới.

“Cùng budget” phải ghi rõ optimizer steps, effective contrastive candidates, số teacher/video/text forwards, trainable parameters và wall-clock. Chỉ match epochs không đủ khi Full thêm masked forwards. Có thể báo cả step-matched và compute-matched rows nếu cost khác đáng kể; không cần chạy toàn bộ hai grid ngay pilot.

## 19. Correctness tests có ý nghĩa

Đây là tests cho rủi ro implementation, không phải substitute cho retrieval experiments.

| Test | Fixture/assertion | Lỗi bị chặn |
|---|---|---|
| Feature bridge | `[B,F,1024]↔[B,1024,F,1]`, mask CLS/pad chính xác | Đảo chiều tensor hoặc đảo valid mask. |
| Zero-init parity | Same checkpoint/input, adapter zero → same token scores/ranks | Gain/loss do integration patch. |
| Score orientation | Asymmetric fixture; raw `I2T`,`T2I` đều video×text | Báo T2V thành V2T. |
| Paired scorer | Own `(v,t)` score bằng entry tương ứng all-pairs | Auxiliary dùng scorer khác inference. |
| Token identity | Offsets implementation tạo IDs bằng upstream | Ghép BPE sai hoặc thay nhầm word. |
| Truncation | Word mất một BPE → aux excluded, coarse retained | Giám sát vào word không còn trong input. |
| Canonical mapping | Sampling view hash khác → refuse/remap có kiểm tra | Cached span lệch timeline. |
| Padding locality | Local weights ở padding=0; padding residual=0 | Học zero-padding artifacts ở local head. |
| Raw locality | Với r=0, thay `h[j]` không đổi local `z[i]`, i≠j | Local branch vô tình nhận global context. |
| RF closure | Clip RF giao W bị mask dù center ngoài W | Evidence còn rò qua clip chồng lấn. |
| Control match | W/C cùng count, valid mask/positions giữ nguyên | Loss dựa vào sequence length. |
| Gradient | Teacher không grad; adapter output layer nhận lexical/coarse grad | `no_grad()` cắt student path. |
| Zero-init gradient | Lower adapter grad xuất hiện sau update đầu | Test sai do W2 khởi tạo zero. |
| Lexical math | Equal pos/neg, K hợp lệ → `rho*log(1+K*exp(m/tau))` | Dấu margin/temperature sai. |
| Masked negatives | Padded negative không đóng góp loss/gradient | K-padding đổi objective. |
| Evidence sign | `Delta_clean−Delta_W >= m_dep` → dep=0 | Hinge bị đảo dấu. |
| Huber scale | So giá trị tại 0, delta/2, 2delta với định nghĩa §12 | Nhầm SmoothL1 và Huber. |
| KL direction | Teacher=student → near zero; teacher detach | Đảo KL hoặc distill ngược. |
| Empty auxiliary | O=0 trên một/all ranks vẫn finite, không deadlock | Training crash ở batch coverage thấp. |
| DDP normalization | Unequal eligible counts, compare one-GPU global batch | Mean-of-means sai. |
| Full gallery | Count/ID hash bằng manifest gốc, không lọc eligibility | Tăng R@1 do giảm gallery. |
| Test isolation | Training evaluation logger chỉ có split=dev | Test checkpoint selection. |
| Export parity | Export không có teacher/head nhưng score như checkpoint | Deploy sai kiến trúc. |

Ví dụ fixture orientation: `S=[[9,8,0],[10,7,0],[0,0,9]]`, diagonal là positive, cho V2T R@1 = 2/3 và T2V R@1 = 1/3. Kiểm tra thêm **per-query ranked IDs và matrix transpose**, vì ở các fixture khác scalar R@1 hai chiều có thể tình cờ bằng nhau. Thêm tie fixtures riêng để kiểm tra rank policy.

Không áp generic assertion “đổi padding feature không được đổi CiCo score” lên baseline nếu scorer gốc còn padding behavior chưa mask hết. Test local branch theo contract, và test global scorer theo **parity với upstream**. Nếu sửa global masking, đánh giá baseline-fixed riêng trước ELSC.

Trong phiên viết tài liệu này chưa có PyTorch/checkpoint/dataset runtime để chạy các tests model trên. Những đoạn code được cung cấp là reference kernels và pseudocode; AI triển khai phải chạy gate tương ứng trước khi báo method hoạt động.

## 20. Chi phí và phương án triển khai khi thiếu tài nguyên

Với D=1024, H=256, pointwise adapter có khoảng 0.527 triệu parameters gồm LayerNorm và hai Linear; local linear head 1024→512 thêm 0.524 triệu **training-only** parameters khi text dimension là 512. Con số này là tính từ kiến trúc, không phải đo latency/GPU memory.

Adapter inference cost xấp xỉ `2*F*D*H` multiply-accumulate/video, ngoài CiCo Transformer. ELSC-Min có thể reuse video forward của coarse branch khi views trùng. Full thêm hai video encodings cho mỗi selected evidence instance; ratio 25% với một instance/video không đồng nghĩa chắc chắn chỉ +50% wall-time, vì batching/text/scoring/cache overhead còn khác.

Giảm cost theo thứ tự:

1. Dùng frozen released I3D features và batch cache text bank/support offline.
2. Chạy r=0, frozen core, một target/video trước nếu memory hạn chế.
3. Với paired full losses, tính đúng selected pairs thay vì all-pairs across masks.
4. Chunk score matrix cho evaluation; không đổi scorer để tiết kiệm bộ nhớ.
5. Dùng exact block top-k cho graph mining; cap occurrences/word trước khi cần ANN.
6. Chỉ thử unfreeze/backbone mới khi auxiliary contribution đã được kiểm chứng.

Không ước lượng giờ GPU cố định khi chưa đo một epoch. Log pilot throughput và extrapolate theo actual optimizer steps, ghi giả định. Nếu thiếu dev features/checkpoint, hoàn thành code/schema/tests độc lập trước; tạo `blocked_assets.json` nêu asset cụ thể còn thiếu, không fabricate đường dẫn tải.

## 21. Mở rộng sang How2Sign, CSL-Daily và SEDS

### 21.1. Dataset transfer

Giữ kiến trúc/loss defaults ban đầu, rebuild vocabulary/support/graph chỉ từ train của dataset mới. Không tái dùng word IDs hoặc feature domain-aware checkpoint của PH cho dataset khác mà không ghi rõ transfer setting.

How2Sign: kiểm tra frontal-view/re-alignment và annotation release; giữ video/pair IDs. CSL-Daily: kiểm tra caption language và tokenizer mapping đặc biệt; không suy German/English offsets áp dụng trực tiếp. Mỗi dataset có baseline parity gate, manifests, cache hashes riêng.

Nếu local assumptions không phù hợp nhiều sample (fingerspelling, repeated lexical units, distributed nonmanual signals), báo coverage/abstention. Không ép mọi sample có exactly one sign span. Auxiliary objective là subset supervision; full-gallery evaluation vẫn tất cả sample.

### 21.2. SEDS integration sau pilot

Clone [SEDS](https://github.com/longtaojiang/SEDS), pin SHA thực tế, chạy evaluator/feature parity. Đặt adapter ELSC trên RGB raw temporal feature branch trước global attention/fusion. Retrieval gradient phải đi qua fusion về adapter. Giữ pose branch và semantic module của SEDS theo controlled baseline trước khi thêm thay đổi khác.

Không lấy fused RGB-pose output đã contextual rồi gọi đó là local raw feature. Nếu muốn local multimodal branch, cần RF và alignment của cả RGB/pose, giữ confidence handling của SEDS; đó là extension riêng, không phải việc chỉ đổi một flag `backbone=seds` trong code CiCo.

Chỉ report contribution “backbone-compatible” sau paired controls `SEDS baseline / +matched caption / +ELSC` cùng resources. Thắng CiCo không chứng minh thắng SEDS/C²RL. Số SOTA mới và paper chưa truy cập đầy đủ cần được cập nhật khi chuẩn bị manuscript; tài liệu triển khai này không chứng nhận leaderboard toàn domain.

## 22. Danh sách nghiệm thu implementation

- [ ] Commit upstream đã pin; patch compatibility có diff và lý do.
- [ ] Environment lock và provenance checkpoint/data được ghi.
- [ ] PH dev routing, feature-path mode và fusion alpha đã kiểm tra.
- [ ] Bridge, mixed scorer, two-direction metrics đạt golden parity ở phạm vi đã báo.
- [ ] Test/GT không được dùng để chọn checkpoint hoặc tạo train cache.
- [ ] BPE mapping bằng upstream, replacement đúng một canonical word span.
- [ ] Teacher frozen, support/negative graph train-only, cache hash mismatch bị reject.
- [ ] Local head đọc feature trước contextual Transformer; default r=0.
- [ ] Lexical gradients vào adapter, không chỉ head.
- [ ] ELSC-Min và same-negative/same-capacity controls đã chạy.
- [ ] Full chỉ chạy khi RF metadata/control matching đúng; masked encodings recomputed.
- [ ] DDP/AMP math và empty auxiliary batches có tests.
- [ ] Evaluation giữ toàn gallery và đúng query/candidate IDs.
- [ ] SAN FG được ghi disabled nếu chưa có official protocol artifacts.
- [ ] Export inference bỏ teacher/lexical head và vẫn score parity.
- [ ] Results tách reported baseline scores khỏi measured local runs; không có placeholder metrics bị coi là thật.

**Định nghĩa hoàn thành:** implementation có thể tạo đúng training artifacts, train được ELSC-Min, chạy evaluator chuẩn và xuất evidence/ablation đủ kiểm tra giả thuyết. “Method đã vượt SOTA” là một kết luận thực nghiệm riêng, cần dữ liệu đo và so sánh công bằng; không phải điều kiện có thể bảo đảm chỉ bằng thiết kế code.

## 23. Nguồn để AI triển khai mở trực tiếp

1. [CiCo README — pinned assets và historical commands](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/README.md).
2. [CiCo modeling — encoder APIs, training objective và token scorer](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/modeling.py).
3. [CiCo module_clip — FeatureTransformer và returned tokens/masks](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/module_clip.py).
4. [CiCo tokenizer — normalization, regex và BPE](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/tokenization_clip.py).
5. [CiCo metrics — official kernels](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/metrics.py).
6. [CiCo main — score mixing/evaluation orchestration cần audit](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/main_task_retrieval.py).
7. [CiCo PH eval loader — feature fusion/sample/text behavior](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/dataloader_ph_retrieval.py).
8. [CiCo PH train loader — augmentation và path audit](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/dataloader_ph_retrieval_train.py).
9. [SAN dataset code — negative table consumption và substitutions](https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/datasets.py).
10. [SAN model code — coarse/hard-negative score construction](https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/models.py).
11. [SAN trainer — actual public evaluation path](https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/train_vlp_v2.py).
12. [SAN paper — method và official FG setting](https://aclanthology.org/2026.acl-long.1302/).
13. [SEDS repository — extension sang dual-stream](https://github.com/longtaojiang/SEDS).
14. [C²RL paper — đối thủ cần tính đến khi đánh giá khả năng cạnh tranh](https://arxiv.org/abs/2408.09949).

Các công thức ELSC, thresholds selector, fallback miner, API `elsc.*`, config và gates trong tài liệu là thiết kế được đề xuất cho dự án. Chúng không được trình bày như code/tuyên bố sẵn có của các tác giả CiCo hoặc SAN.
