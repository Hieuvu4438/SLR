# DIVE-SLR v2 — Đặc tả triển khai end-to-end cho AI Agent

**Phiên bản:** Implementation Spec 1.0 — 08/09/2026.  
**Phương pháp nguồn:** `DIVE_SLR_Reviewed_Proposal.md`, Reviewed v2, 07/09/2026.  
**SHA-256 của proposal nguồn:** `0fa349240568f7dd92cbb189e684bd353424676c3e1cab3542e4082c043bfd7c`.  
**Đầu ra cần xây dựng:** một repository có pipeline dữ liệu → baseline → local reference → contrast bank → DIVE student → chọn cấu hình trên dev → truy hồi toàn gallery → đánh giá và ablation.  
**Ngôn ngữ triển khai:** Python/PyTorch. Tài liệu dùng tiếng Việt, giữ tên API, tensor và config bằng tiếng Anh để tránh lệch tên khi code.  
**Trạng thái:** đây là đặc tả để implement, chưa phải một repository DIVE đã huấn luyện. Code mẫu là reference logic cần đưa vào test; không phải báo cáo đã chạy PyTorch hoặc benchmark.

## 0. Hướng dẫn bắt đầu dành cho AI Agent

Đọc hết §0–§4 trước khi tạo module. Sau đó thực hiện milestone ở §19 theo dependency. Dùng tài liệu này làm hợp đồng triển khai; proposal nguồn cung cấp động cơ nghiên cứu và literature review. Các quyết định engineering bổ sung được ghi rõ ở §2, không được mô tả thành phát hiện thực nghiệm.

Khi nhận task, agent cần:

1. Kiểm tra repository hiện tại, các hướng dẫn áp dụng trong workspace và tài nguyên thực sự có. Giữ các thay đổi đang có của người dùng.
2. Tạo `IMPLEMENTATION_STATUS.md`, ghi từng milestone là `not_started`, `in_progress`, `passed` hoặc `blocked`, kèm bằng chứng và blocker cụ thể.
3. Xây vertical slice bằng synthetic fixtures trước, rồi nối adapter SEDS thật. Fixtures không được tính là replication hoặc sign grounding.
4. Hoàn tất những phần không phụ thuộc tài nguyên bị thiếu. Ví dụ thiếu checkpoint không ngăn viết scorer, losses, sampler, evaluator và cache validator.
5. Mỗi milestone phải có command, output artifact và acceptance check. Không đánh dấu hoàn thành chỉ vì có stub, TODO hoặc mock trả tensor đúng shape.
6. Nếu gặp xung đột với phương pháp, ghi `DESIGN_DEVIATIONS.md`: điều khoản, lý do, tác động, phương án. Routine engineering đã được đặc tả thì tự thực hiện; không cần hỏi lại từng lựa chọn.

**Agent không tự thay objective, tăng nguồn supervision, thêm module, đổi test relevance hoặc đổi benchmark để làm kết quả đẹp hơn.** Nếu phải thay để chạy được, đặt tên variant riêng và giữ bản chuẩn có trạng thái rõ ràng.

### 0.1. Phạm vi bắt buộc và phần mở rộng

| Mức | Phải có |
|---|---|
| Core end-to-end | How2Sign + SEDS adapter, một process/GPU; toàn bộ DIVE v2 và A0–A6 |
| CPU correctness | Synthetic fixtures, scoring/loss tests, mining/support fixtures, evaluator và checkpoint round-trip |
| Sau khi core đúng | CiCo/backbone transfer; PHOENIX-2014T hoặc CSL-Daily; phân tích cơ chế |
| Tùy tài nguyên | DDP, AMP, activation checkpointing, ANN shortlist, top-K deployment |
| Ngoài main method | Generation loss, OT, Gaussian uncertainty, EMA teacher, synthetic negative captions, global Transformer trong evidence encoder |

“End-to-end” ở đây là toàn bộ pipeline có thể vận hành từ dữ liệu đến đánh giá. Nó không có nghĩa tất cả encoders đều nhận gradient: B0, text encoder và reference bắt buộc frozen ở stage student.

## 1. Định nghĩa thành công và các bất biến

Một implementation đạt yêu cầu khi đúng cả **ngữ nghĩa phương pháp**, **gradient/data flow**, **protocol đánh giá** và **khả năng tái chạy**. Cải thiện recall là kết quả nghiên cứu cần đo, không phải điều được giả định để pass test.

| ID | Bất biến phải giữ |
|---|---|
| INV-01 | Ma trận score có hàng = video, cột = text; V2T dùng hàng, T2V dùng cột |
| INV-02 | Baseline B0 được chọn bằng dev và frozen trước warm-up evidence/student |
| INV-03 | Mỗi score inference chỉ phụ thuộc video ứng viên và text query tương ứng |
| INV-04 | Evidence video encoder không nhận caption, rival unit, support hoặc sample ID |
| INV-05 | Evidence features lấy trước global temporal Transformer; RF truy ngược tới thời gian gốc được ghi lại |
| INV-06 | Text units độc lập với rival; local target lấy đúng unit vector đang được dùng trong evidence score |
| INV-07 | Mọi mask bên trong dự án là boolean, `True = valid`; mask trước softmax |
| INV-08 | Student và reference cùng kiến trúc, preprocessing, grid và precision contract; khởi tạo bằng copy thật, không chung parameter storage |
| INV-09 | Score là `S0 + gamma * (E_student - E_reference) / 2`, gamma không âm |
| INV-10 | `q`, `g`, text features, baseline và reference đều stop-gradient |
| INV-11 | Local score là weighted mean của cosine, không normalize vector đã pooling |
| INV-12 | Local/pair losses chia cho số contrasts đã lấy mẫu trước support rejection, không chia cho tổng reliability |
| INV-13 | Contrast bank, schema calibration và support targets chỉ dùng train |
| INV-14 | Checkpoint, gamma và threshold được chọn trước test; test không nằm trong train/selection loop |
| INV-15 | A3/A4/A5/A6 dùng cùng cặp, batch schedule, weights, budget, B0/reference và seed tương ứng |
| INV-16 | Cache sai fingerprint phải bị từ chối, không âm thầm dùng tiếp |
| INV-17 | R@K tính theo số query và relevance IDs; ties không được sinh thêm query |
| INV-18 | Không có benchmark result khi chỉ chạy fixtures hoặc chưa có tài nguyên thật |

## 2. Quyết định engineering đã chốt

Các mục sau làm rõ phần proposal chưa quy định đủ để viết code. Chúng không thay cơ chế chính.

| ID | Quyết định | Lý do và phạm vi |
|---|---|---|
| E-01 | Canonical grid cố định cho evidence ở warm-up/student/eval; tối đa 64 clips | Offline support phải trỏ đúng student feature, không trỏ vào random crop mới mỗi epoch |
| E-02 | Hai shifted views chỉ tạo support offline; student forward chính dùng canonical view | Cho phép cache reference chính xác, tránh phải tính lại q mỗi bước |
| E-03 | FP32 là correctness profile; AMP/FP16 cache là optimization profile được kiểm riêng | Reference quantization không được làm mất identity initialization mà không ghi nhận |
| E-04 | RF 24 được hiểu trên trục frame đưa vào pose GCN; luôn map về raw timestamps | Nếu loader downsample, 24 input steps không đồng nghĩa 24 frame gốc liên tiếp |
| E-05 | B0 SEDS dùng fusion score và directional mix đã khóa; evidence giữ mean hai hướng như proposal | Không cộng nhầm scaled logits với một cosine residual |
| E-06 | BN running statistics cố định; BN affine cũng frozen trong main profile | Proposal cho phép affine train nếu khai báo; ở đây chọn cấu hình đơn giản, nhất quán |
| E-07 | Chọn checkpoint bằng gamma_train cố định; sau đó mới sweep gamma trên dev một lần | Tránh mơ hồ giữa joint epoch×gamma search và post-training calibration |
| E-08 | Checkpoint khởi tạo student là một candidate hợp lệ | Nếu training không hơn baseline trên dev, không buộc chọn checkpoint đã làm kém đi |
| E-09 | Mẫu số H tính trên pre-support bank samples; support-failed records vẫn lưu với g=0 | Giữ ý nghĩa giảm supervision tuyệt đối và khả năng audit acceptance rate |
| E-10 | Projector hidden dimension = 1024; output = 512; input dimensions được probe | Không hard-code sai RGB/pose feature width từ tên backbone |
| E-11 | Pilot student 10 epochs, full 30, reference warm-up 5; lịch LR quy định ở §14 | Đây là điểm khởi đầu, chưa được tối ưu thực nghiệm |
| E-12 | Main mining bắt đầu bằng strict numeric-slot schema; lexical schemas cần audit artifact | Không biến một semantic parser chưa kiểm tra thành nguồn nhãn âm mạnh |
| E-13 | Từ chối config key lạ và missing required resources ở stage tương ứng | Phát hiện typo config và ngăn fallback âm thầm |

Nếu `E-01` không khớp grid của RGB artifact đã có, tạo grid từ timestamp của artifact đó khi nó đáp ứng contract. Không tự gán dense timestamps cho sparse features. Nếu không có hai shifted views thật, core code vẫn triển khai được, nhưng main support generation là `blocked`; variant single-view phải có tên riêng.

## 3. Repository, môi trường và interfaces

### 3.1. Cấu trúc module cần tạo

Các đường dưới đây là đường tương đối trong repository sẽ xây, không phải các file đã tồn tại trong workspace này.

| Đường | Trách nhiệm chính |
|---|---|
| `pyproject.toml`, `requirements.lock` | Package, entry point `dive`, dependency versions thực sự đã kiểm |
| `configs/` | Base, pilot, full, fixture, ablation và dataset overrides |
| `src/dive/config.py` | Typed config, validation, resolved config và hashing |
| `src/dive/data/manifest.py` | Sample schema, split IDs, provenance, kiểm tra tài nguyên |
| `src/dive/data/text_units.py` | Normalization/offsets, BPE mapping, atomic units, truncation |
| `src/dive/data/temporal.py` | Frame map, canonical grid, shifted views, RF intervals |
| `src/dive/data/pose.py` | Chuẩn hóa pose và missing-joint policy có locality |
| `src/dive/data/relations.py` | Known positives, excluded negatives, duplicates/source overlap |
| `src/dive/data/collate.py` | Padding, bool masks, batch ID maps |
| `src/dive/adapters/base.py`, `seds.py` | Hợp đồng B0, feature taps, native score và checkpoint import |
| `src/dive/models/evidence.py` | Local pose encoder, RGB concatenation, projector, BN policy |
| `src/dive/models/scoring.py` | Masked late interaction, centered residual, chunking |
| `src/dive/losses.py` | Retrieval, local, full-score pair, optional preservation |
| `src/dive/mining/neighbors.py` | Pooled shortlist và S0 reranking |
| `src/dive/mining/slots.py` | Atomic-slot validation và schema provenance |
| `src/dive/mining/support.py` | Differential support, absolute gate, rebin/JSD/RF filter |
| `src/dive/mining/bank.py` | Bank serialization, g=0 records, counters và fingerprints |
| `src/dive/cache.py` | Feature/index manifests, deterministic cache keys |
| `src/dive/training/sampler.py` | Contrast sampling, endpoint quotas, reproducible batch plans |
| `src/dive/training/runner.py` | Baseline/warm-up/student stages, optimizer, dev selection |
| `src/dive/training/state.py` | Model/optimizer/RNG/sampler checkpoint và resume |
| `src/dive/eval/scoring.py` | Pair blocks, gallery caches, two retrieval directions |
| `src/dive/eval/metrics.py` | Independent evaluator dùng IDs, stable ties và multi-positive |
| `src/dive/eval/calibration.py`, `analysis.py` | Gamma, opportunity bound, bootstrap, error transitions |
| `src/dive/cli.py` | Commands ở §18; exit codes và machine-readable outputs |
| `tests/` | Tests bảo vệ bất biến và scientific validity, §20 |
| `docs/` | Data contract, decisions, source adaptation và reproduction notes |

Không import training script của upstream ngay tại module import: các side effects, parser và global device setup phải nằm sau entry point. Vendor/pin phần cần dùng, giữ attribution/license của source.

### 3.2. Environment contract

- Phát hiện Python, PyTorch/CUDA, driver, GPU/VRAM, NumPy, tokenizer và các upstream dependencies đang có trước khi chọn lockfile.
- Không mặc định bản PyTorch mới nhất tương thích checkpoint/SEDS. Chọn một environment chạy adapter thật, ghi exact versions và lock dependencies. Các tài liệu PyTorch được tham chiếu ở §23 giải thích API; chúng không phải yêu cầu cài đúng version của trang docs.
- Correctness profile dùng FP32, một process và seed cố định. Đặt seed cho Python, NumPy, torch CPU/CUDA, DataLoader workers; lưu sampler RNG độc lập. Bật deterministic algorithms khi backend hỗ trợ và log mọi ngoại lệ, không chỉ gọi `manual_seed` rồi claim bitwise reproducibility. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
- Nếu thiếu torch hoặc GPU, `doctor` báo khả năng theo stage. Không cần cài CUDA toolkit tùy tiện để chạy CPU fixtures; không ghi GPU integration test là passed khi chưa chạy.

### 3.3. Adapter API phải hiện thực

```python
from typing import Protocol

class BaselineAdapter(Protocol):
    def load_and_validate(self, checkpoint_path, resolved_config): ...
    def encode_video_native(self, video_batch): ...
    def encode_text_native(self, text_batch): ...
    def score_prelogit(self, video_features, text_features): ...
    def encode_text_units(self, text_batch, unit_mapping): ...
    def rgb_local_features(self, video_batch, grid_id): ...
    def clone_local_pose_encoder(self): ...
    def describe_preprocessing(self): ...
    def describe_receptive_field(self, video_batch, grid_id): ...
```

Đây là interface sketch; mỗi method thật phải có typed return object, shape validation, ID mapping và lỗi cụ thể. `score_prelogit` trả `[Bv, Bt]`; không nhận ground-truth caption của một video từ gallery loader.

## 4. Dữ liệu và hợp đồng tensor

### 4.1. Manifest một sample

Lưu JSONL/Parquet metadata và tensor shards tách riêng. Ví dụ dưới đây chỉ mô tả schema; các giá trị `fixture_*` không phải dữ liệu How2Sign thật.

```json
{
  "schema_version": "sample.v1",
  "sample_id": "fixture_train_0001",
  "video_id": "fixture_video_0001",
  "text_id": "fixture_text_0001",
  "split": "train",
  "sign_language": "ASL",
  "text_language_original": "en",
  "text_language_model": "en",
  "text_original": "the value is 12 meters",
  "text_model": "the value is 12 meters",
  "source_video_id": null,
  "signer_id": null,
  "source_start_sec": null,
  "source_end_sec": null,
  "duration_sec": 4.0,
  "video_path": "fixtures/video_0001.mp4",
  "pose_path": "fixtures/pose_0001.npz",
  "rgb_feature_key": "fixture_rgb_0001",
  "translation_artifact_hash": null,
  "frame_map_key": "fixture_frame_map_0001",
  "annotation_provenance": "synthetic_fixture"
}
```

Required: unique `sample_id`, split, các ID dùng cho relevance, caption model, thời gian hợp lệ và tài nguyên stage cần. Nullable metadata giữ `null`; không tạo signer/source giả để đủ field. Với main sentence-pair profile, mỗi sample là một annotated video–text pair; duplicated text strings vẫn có text IDs riêng khi benchmark định nghĩa như vậy.

`prepare-data` tạo:

- `manifests/train.jsonl`, `dev.jsonl`, `test.jsonl` và hash của từng file.
- `relevance/{split}.jsonl`: các video/text ID positives theo annotation.
- `relations/train_excluded_negatives.jsonl` với reason/provenance.
- `data_audit.json`: counts, missing assets, overlapping IDs, duplicate groups, language/token coverage, filtered IDs và reasons.

Không tạo dev bằng cách trỏ tới test. Nếu feature files chưa có dev, phải tạo features cho dev thật hoặc ghi blocker. Chênh lệch split giữa các paper đã được nêu trong proposal; exact IDs và caption artifacts là nguồn chuẩn của experiment, không phải một con số tổng được hard-code.

### 4.2. Shapes, dtype và gradient ownership

Ký hiệu: `Bv/Bt` là số video/text trong block; `Nv` số canonical clips đã padding; `Nt` số text units; `F` số pose input steps; `d=512`; `H` số contrasts lấy mẫu.

| Tensor | Shape | Dtype main | Gradient |
|---|---|---|---|
| RGB local `r` | `[Bv, Nv, Dr]` | float32 sau load | Không |
| Pose inputs | Adapter-specific, có explicit frame/joint/channel axes | float32 | Input không cần; pose parameters có |
| Local pose `p` | `[Bv, Nv, Dp]` | float32 | Student: có |
| Evidence `u` | `[Bv, Nv, 512]` | float32 | Student: có |
| Reference `u_ref` | `[Bv, Nv, 512]` | float32 | Không |
| Text units `e` | `[Bt, Nt, 512]` | float32 | Không |
| Video validity `vm` | `[Bv, Nv]` | bool | Không |
| Text validity `tm` | `[Bt, Nt]` | bool | Không |
| Pair scores `S0, E, S` | `[Bv, Bt]` | float32 | Chỉ phần student của E/S |
| Interaction `M` | `[Bv, Bt, Nv, Nt]` trong mỗi block | float32 | Có |
| Support `q` | `[H, 2, Nv]` hoặc ragged + offsets | float32 | Không |
| Unit indices | `[H, 2]` trên text units của hai endpoints | int64 | Không |
| Reliability `g`, text distance `h` | `[H]` | float32 | Không |
| Local quartet `Z`, full quartet `Q` | `[H_active, 2, 2]` | float32 | Có |
| Bridge feasibility `omega` | `[H_active, 4]` | bool | Không |
| Known positives `P`, eligible candidates `C` | `[Bv, Bt]` | bool | Không |

`Nv` không gồm CLS của upstream. `Nt` không gồm BOS/EOS/padding/punctuation-only units. B0 giữ special-token policy đã khóa khi tái lập; không âm thầm đổi B0 sang evidence-unit policy.

Tensors padding được zero hóa bằng mask từ dữ liệu trước scoring. Valid vectors có NaN/Inf hoặc norm quá nhỏ là lỗi cần điều tra. Không dùng `nan_to_num` toàn pipeline để giấu lỗi trên valid features.

### 4.3. Text units, BPE và truncation

1. `text_model` là chuỗi thật sự đi vào encoder sau normalization xác định trước. Lưu version của Unicode/whitespace normalization và tokenizer. Không stemming/xóa stopwords/phủ định để tạo cặp dễ.
2. Unitizer tạo spans trên chính chuỗi đó. Với main English profile, bắt đầu từ word units; numeric expression hợp lệ có thể thành một atomic unit theo rule cố định.
3. Lưu `unit_char_start/end`, `subword_indices`, `token_ids`, `unit_kind`, `normalized_value`, `complete_after_truncation`.
4. Lấy contextual projected token features từ **cùng frozen text encoder của B0**. Với mỗi unit, mean các raw projected subword features hợp lệ rồi normalize một lần thành `e[m]`. Không normalize từng subword rồi đổi pooling mà không đặt variant riêng.
5. Mining, support và inference cùng dùng unit sequence ấy. Target `d_i` là `e_i[m_i]`, không lấy vector word embedding trước Transformer hoặc gọi một encoder khác.
6. Nếu một unit bị cắt mất một phần subwords, không dùng nó làm target. Evidence scoring chỉ dùng complete units; ghi tỷ lệ truncated/removed. Không clamp out-of-range index về token cuối.
7. Loader SEDS có logic giới hạn/sampling text riêng; adapter phải ghi lại mapping qua mọi phép chọn token. Không suy offset từ bản full text rồi áp lên sequence đã bị chọn lại. Nếu thay text policy, tái lập B0 và toàn bộ controls theo policy mới, ghi là protocol adaptation.
8. Nếu captions được dịch sang English, lưu artifact dịch có hash. Mining làm trên `text_model`; đối chiếu nghĩa với caption gốc ở semantic validation. Không gọi dịch online mỗi epoch. [CiCo paper](https://arxiv.org/pdf/2303.12793).

CLIP tokenizer cũ có thể không trả offsets như một tokenizer khác. Khi đó instrument đúng tokenizer/pre-tokenization/BPE path và kiểm tra token IDs round-trip với native encoder. Không thay tokenizer bằng một thư viện khác chỉ vì tên model giống nhau.

### 4.4. Relevance và false negatives

Trong training, xây `P` từ annotation đã biết; `C` gồm positives và các negatives được phép. Nếu cặp bị excluded vì exact duplicate text, known equivalence hoặc source overlap thì loại nó khỏi negatives, không tự biến thành positive. Luôn có `P ⊆ C`.

Không deduplicate text IDs ở evaluator benchmark trừ khi protocol chính thức yêu cầu. Không dùng semantic similarity model để tự gán multi-positive test labels. Với source metadata thiếu, ghi thiếu; không giả định hai video khác IDs chắc chắn không overlap.

## 5. Temporal grid, pose preprocessing và locality

### 5.1. Ba trục thời gian phải phân biệt

| Trục | Ý nghĩa |
|---|---|
| Raw time | Timestamp/frame index trong video gốc |
| Pose input steps | Chuỗi thật sự đi vào GCN, có thể đã sampling |
| Canonical clip index | Vị trí `n` của evidence vector và support q |

Lưu mapping raw timestamp cho mọi pose input step và raw interval cho mỗi RGB clip. Nếu mỗi input step cách nhau `s` raw frames, một RF 24 input steps có span `(24 - 1) * s + 1` raw frames trong trường hợp đều; sampling không đều phải dùng timestamps thật. Không dùng con số 24 trực tiếp để tính tỷ lệ video bị che.

Preprocessing evidence phải có dependency hữu hạn đã biết: normalize coordinates theo frame hoặc bằng train statistics cố định; missing joints có validity/confidence rõ. Không nội suy từ toàn video rồi claim RF ngắn. Nếu dùng phép điền theo cửa sổ, cộng cửa sổ ấy vào RF. Không loại riêng frame của tay phải/trái làm ba streams mất đồng bộ mà vẫn dùng cùng index.

### 5.2. Canonical clip grid

Ưu tiên import `clip_start`, frame map và RGB timestamps của artifact chuẩn đã xác minh. Khi phải tạo mới:

1. Xây dense list cửa sổ 16 pose input steps, stride cấu hình đã khóa; bổ sung end window khi cần, loại duplicated starts.
2. Nếu quá 64 windows, lấy tối đa 64 indices phân bố đều, deterministic. Nếu ít hơn, giữ số thật và pad bằng invalid slots, không lặp window để giả đủ 64.
3. Video ngắn hơn 16 steps: có thể repeat boundary frame cho forward nếu encoder cần, nhưng RF/raw duration chỉ tính phần video thật. Thông thường sample này không qua support concentration gate.
4. RGB và pose của canonical clip phải tương ứng thời gian. Không nội suy một contextual RGB embedding để giả feature của window mới.
5. Tạo `grid_id = hash(frame_map, starts, window, stride, max_clips, alignment_policy)`; grid cố định qua epochs.

Feature extraction policy này phải được so với adapter B0; không giả định hai loader có cùng grid chỉ vì cùng `feature_len=64`.

### 5.3. Shifted views cho support

Main pilot dùng offsets `[-1, +1]` pose input step cho start, giới hạn trong miền hợp lệ. Recompute reference features và RGB window features thật, hoặc đọc đúng dense-window cache. Việc shift làm hai view trùng nhau ở biên được ghi nhận; nếu cả hai grids giống hệt, đánh dấu `no_distinct_views`.

Thay đổi timestamp phải kéo theo RGB/pose alignment tương ứng. Copy lại tensor canonical rồi đổi tên `view_id` là không hợp lệ. Student/reference score chính vẫn dùng canonical view; các views này chỉ dùng để sinh support offline.

### 5.4. RF intervals

Với một canonical clip, lấy hợp dependency của RGB path, pose GCN, `sign_conv` và preprocessing. Lưu intervals nửa mở `[start_sec, end_sec)`; không tính phần padded ngoài video. Nếu dùng một bounding interval thay cho hợp nhiều đoạn rời, ghi đó là bound bảo thủ.

Kiểm tra locality ở raw inputs: sửa frame/pose ngoài RF không đổi output clip trong tolerance; gradient ra ngoài RF bằng 0 trong training profile khi input được bật `requires_grad`. Kiểm lại khi thay preprocessing, BN policy hoặc backbone.

## 6. Adapter SEDS và baseline B0

### 6.1. Snapshot và feature taps

Pin SEDS tại `434e3f714fcb6a7d1f4001fb9a246bbd93ec0246`. Các điểm đã đọc trong lượt review:

| File/hàm upstream | Cách dùng |
|---|---|
| `modules/modeling.py:get_sign_output` | RGB cache và pose clip outputs trước `clip.encode_image` |
| `signbert.gcn_emb(pose_all)` | Chạy GCN trước khi slicing windows; phải tính dependency mở rộng |
| `signbert.sign_conv` | Conv trong clip rồi mean temporal; clone vào local pose path |
| `get_visual_output` / `clip.encode_image` | Global/contextual video representation cho B0; không dùng output này làm local evidence |
| `get_sequence_output` / `clip.encode_text(return_hidden=True)` | Nguồn frozen contextual text features, adapter phải xác minh projection/shape |
| `flip_similarity_softmax` | Native fusion/pose/RGB directional scores có logit scale |
| `_run_on_single_gpu_new_mix` trong training file | Native evaluation trộn hai directional fusion scores bằng `dual_mix` |

Nguồn: [modeling.py](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/modules/modeling.py), [SignBERT](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/modules/modeling_signbert.py), [training/evaluation](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/main_task_retrieval.py). Snapshot không bảo đảm tái tạo paper khi thiếu checkpoint/data artifacts.

Pose output upstream có thể là `[B, Dp, Nv, 1]`; adapter phải chuyển rõ về `[B, Nv, Dp]`. Upstream video mask có CLS đầu và `0 = valid`: evidence dùng `(legacy_mask[:, 1:] == 0)` **chỉ sau khi assert** đúng một CLS và số vị trí khớp pose/RGB outputs. Text mask normalize riêng. Không áp `[:, 1:]` lên mọi tensor theo thói quen.

Trong snapshot này, `get_similarity_logits(..., is_train=False)` rơi vào route trả `None`; native evaluator gọi route `is_train=True` khi model đã eval. Agent không được đồng nhất route flag này với `module.training`. Viết pure scoring wrapper hoặc gọi đúng route dưới `model.eval()`/no-grad rồi kiểm parity; không bật train mode của B0 chỉ để có scores. [SEDS score dispatcher](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/modules/modeling.py).

### 6.2. Baseline fixes trước thí nghiệm DIVE

Phải sửa dev mapping/routing/training selection; mask candidate padding trước softmax; evaluator orientation/ties. Ghi từng patch trong `BASELINE_ADAPTATIONS.md`. Test `dev_loader.sample_ids == declared_dev_ids` và không có test ID trong selection path.

Stage B0 tái lập objective/backbone nguồn và resource budget đã khóa. Không gọi một model random-init với tên SEDS là baseline tái lập. Checkpoint phát hành có lịch sử selection chưa rõ có thể dùng để debug adapter, nhưng controlled result cần B0 được huấn luyện/chọn theo protocol đúng hoặc có provenance đủ xác nhận.

### 6.3. Score trước logit scale

Với native fusion directional logits `L_vt`, `L_tv` cùng shape `[Bv, Bt]` và scalar `a = exp(logit_scale)`:

$$
S_0 = \frac{\beta L_{vt} + (1-\beta)L_{tv}}{a},\qquad \beta=0.5.
$$

Hai tensor trên đều có rows video; tên `tv` không tự có nghĩa phải transpose. Assert bằng fixture IDs. Không cộng pose/RGB branch scores thêm một lần nếu native selected endpoint là fusion. Nếu adapter lấy trực tiếp unscaled cosine aggregation thì không chia lần hai.

Log min/max/quantiles của `S0` và `a`. Nếu `S0` còn có độ lớn như logits hàng chục/hàng trăm, chặn stage trước khi dùng bound `2*gamma`. Không normalize từng hàng/cột của S0 vì sẽ làm đổi score function và thứ hạng chiều còn lại.

### 6.4. B0 đã khóa

Save model weights/buffers, native config, best dev metric, selection epoch, split hashes, tokenizer/translation/frame-grid metadata và checkpoint hash. Sau đó `.eval()` và `requires_grad_(False)`; optimizer evidence tuyệt đối không chứa parameters của B0.

## 7. Local evidence encoder, warm-up và reference

### 7.1. Forward

$$
p_n^\psi=F_P^\psi(\text{pose};\mathcal I_n),\quad
x_n=[r_n;p_n^\psi],\quad
u_n^\psi=\operatorname{normalize}\left(W_2\operatorname{GELU}(W_1\operatorname{LN}(x_n))\right).
$$

`LN` chỉ theo last/channel dimension. `W1: (Dr+Dp)→1024`, `W2: 1024→512`; initialize linear layers theo framework default đã seeded. Không thêm bias theo video ID, positional embedding toàn video, attention text→video hoặc temporal normalization làm tăng RF.

Với valid feature có pre-normalization norm `<= 1e-6`, raise diagnostic error ở correctness profile. Nếu deployment gặp dữ liệu không hợp lệ do đầu vào, dùng fallback đã định nghĩa ở §8; không cho trainable feature collapse tự tạo validity mask.

### 7.2. Train/eval và freeze policy

`eval()` và việc tắt gradient là hai cơ chế khác nhau. Student có thể học parameters trong khi BN vẫn dùng running statistics cố định. [PyTorch Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html).

- Main profile: dropout của evidence = 0; BN `track_running_stats=True`, running mean/var cố định, affine frozen. Nếu upstream BN không có stored statistics, stage phải xử lý provenance/initialization trước, không coi `.eval()` là đủ.
- Override `EvidenceEncoder.train(mode)` để gọi `super().train(mode)` rồi đặt mọi BN trở lại eval. Assert sau mỗi `student.train()`; outer module `.train()` thường bật lại mode của submodule.
- Freeze B0/reference/text bằng `requires_grad_(False)` **và** eval; cache/frozen forward dùng `torch.no_grad()`. Chỉ bọc frozen branches; không bọc student scoring/loss trong no-grad. [PyTorch no_grad](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html).
- Không dùng `inference_mode()` cho tensors sẽ được tái dùng trực tiếp trong autograd nếu chưa kiểm khả năng tương thích; main implementation dùng no-grad.
- Kiểm optimizer parameter IDs: đúng projector + các pose parameters được cho phép; không trùng nhóm, không chứa B0/ref/text.

### 7.3. Warm-up

Warm-up local encoder bằng symmetric retrieval loss **trên E_local**, ordinary train pairs, 5 epochs. B0/text/RGB extractor frozen. Không dùng hard bank, q, pair loss hoặc local loss ở warm-up.

Chọn checkpoint reference bằng mean(T2V R@1, V2T R@1) trên dev của `E_local`, tie chọn epoch sớm hơn. Báo teacher-only retrieval metrics. Không chọn reference bằng test hoặc support agreement với chính model.

### 7.4. Tạo reference và student

1. Deep-copy warm-up winner thành reference frozen.
2. Deep-copy cùng winner thành student trainable theo policy trên.
3. Hash weights **và buffers**, kiểm equality lúc khởi tạo và kiểm parameter storage không chung.
4. Cache reference canonical/shifted features bằng đúng precision/grid/token contracts.
5. Trước optimizer step đầu, xác minh `E_student ≈ E_reference` và `S ≈ S0`; numerical tolerance FP32 khởi đầu `atol=1e-6, rtol=1e-5`.

Nếu reference cache FP16 còn student FP32, equality chỉ xấp xỉ và có thể lệch. Main correctness profile lưu reference FP32. Compression profile phải đo sai lệch score/rank, ghi cache dtype và không dùng kết quả ấy để tuyên bố identity chính xác.

## 8. Scorer duy nhất dùng trong training và inference

### 8.1. Công thức

Với các tập clip/unit hợp lệ và $M_{nm}=u_n^\top e_m$:

$$
E(v,t)=\frac12\left[
\frac1{N_v}\sum_n\sum_m\operatorname{softmax}_m(M_{nm}/\tau_a)M_{nm}
+\frac1{N_t}\sum_m\sum_n\operatorname{softmax}_n(M_{nm}/\tau_a)M_{nm}
\right],\qquad \tau_a=0.07.
$$

Hai mẫu số là số vị trí **valid**, không phải padded lengths. Đây là expected cosine dưới softmax, không thay bằng max pooling, logsumexp score hoặc một attention value projection.

$$
C(v,t)=\tfrac12(E_\psi(v,t)-E_{ref}(v,t)),\qquad
S(v,t)=S_0(v,t)+\gamma C(v,t).
$$

Valid unit vectors cho `E ∈ [-1,1]`, `C ∈ [-1,1]`; một candidate margin chỉ đổi tối đa `2*gamma`. `gamma_train=0.1` cố định, không tạo trainable parameter cho gamma.

### 8.2. Reference code cho masked late interaction

```python
import torch

def masked_softmax(x, keep, dim):
    keep = keep.expand_as(x)
    has_any = keep.any(dim=dim, keepdim=True)
    z = x.masked_fill(~keep, float("-inf"))
    z = torch.where(has_any, z, torch.zeros_like(z))
    return torch.softmax(z, dim=dim).masked_fill(~keep, 0.0)

def evidence_score_block(u, e, vm, tm, tau_a=0.07):
    # Input contracts: normalized valid vectors, finite padded tensors,
    # u[Bv,Nv,d], e[Bt,Nt,d], bool masks with True=valid.
    assert tau_a > 0 and vm.dtype == tm.dtype == torch.bool
    # Keep float64 for gradcheck; otherwise perform scoring in float32.
    dtype = torch.float64 if u.dtype == e.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=u.device.type, enabled=False):
        u = torch.where(vm[..., None], u.to(dtype), 0.0)
        e = torch.where(tm[..., None], e.to(dtype), 0.0)
        m = torch.einsum("and,bmd->abnm", u, e)
        keep = vm[:, None, :, None] & tm[None, :, None, :]
        pw = masked_softmax(m / tau_a, keep, dim=-1)
        pv = masked_softmax(m / tau_a, keep, dim=-2)
        row = (pw * m).sum(-1).sum(-1) / vm.sum(-1).clamp_min(1)[:, None]
        col = (pv * m).sum(-2).sum(-1) / tm.sum(-1).clamp_min(1)[None, :]
        pair_valid = vm.any(-1)[:, None] & tm.any(-1)[None, :]
        score = torch.where(pair_valid, 0.5 * (row + col), 0.0)
    return score, pair_valid

def compose_score(s0, e_student, e_reference, pair_valid, gamma):
    assert gamma >= 0
    correction = torch.where(pair_valid, 0.5 * (e_student - e_reference), 0.0)
    return s0 + gamma * correction
```

Đầu vào valid phải được kiểm finite/norm ở boundary trước hàm này. Hai chiều có thể có `Bv != Bt`; không dựa vào diagonal trừ khi batch relation thực sự là paired square matrix.

### 8.3. Empty evidence và chunking

Nếu một video/text không còn evidence positions do data validity, đặt residual = 0 cho các cặp liên quan, giữ S0 và giữ sample trong evaluator nếu B0 vẫn chấm được. Nếu B0 cũng thiếu dữ liệu, báo lỗi completeness của gallery; không tự bỏ query rồi tính recall trên phần còn lại.

Chấm theo video/text blocks và ghép bằng `torch.cat` hoặc autograd-safe composition. Không `.detach()`, `.numpy()`, ghi disk rồi đọc lại student scores trong training. Không chạy InfoNCE độc lập từng candidate block rồi lấy mean: denominator phải bao phủ toàn effective batch.

Chunking giảm kích thước temporary tensor, nhưng autograd có thể vẫn giữ activation của mọi block. Đo peak memory thật; nếu cần recomputation/activation checkpointing, phải pass forward/gradient parity test. Không coi gradient accumulation trên các batch nhỏ là tương đương batch 128 có 128 negatives trong denominator.

## 9. Contrast mining chỉ trên train

### 9.1. Candidate shortlist và reranking

Mining chạy sau khi B0 đã khóa; không cần student checkpoint. Với main paired dataset, candidate text ID `j` được ánh xạ tới annotated video `v_j` bằng manifest, không suy từ row number của một file khác.

1. Encode train native B0 representations. Lấy normalized pooled video/text vectors chỉ để tìm shortlist; lựa chọn pooling và dimension phải lưu trong mining config.
2. Với mỗi anchor i, lấy top 128 text candidates từ video i và top 128 video candidates từ text i; loại known self/positive IDs trước khi lấy negative shortlist. Union candidate sample IDs.
3. Chấm các cross-pairs cần thiết bằng `S0`; không dùng pooled dot product làm final hardness score.
4. Tạo quartet baseline `Q0 = [[S0(i,i), S0(i,j)], [S0(j,i), S0(j,j)]]` và bốn margins ở §12. Hardness engineering dùng `-min(margins4(Q0))`, lớn hơn là khó hơn. Sort deterministic theo hardness rồi candidate ID; giữ tối đa 16/anchor.
5. Deduplicate unordered pairs `(min(id_i,id_j), max(...))`, nhưng lưu cả các anchor/direction đã đề cử. Sau đó semantic filter ở §9.2. Không âm thầm tăng top-K để bù filter làm coverage thấp.

Pooled shortlist là xấp xỉ. Trên một train audit subset có seed/IDs cố định, chấm full-gallery S0 và đo tỷ lệ top-16 hard neighbors chính xác nằm trong shortlist. Lưu `shortlist_coverage.json`. Exact search chỉ trên audit subset, không tạo tensor N×N×Nv×Nt cho toàn train.

### 9.2. Atomic-slot validator

Mỗi rule trả `{eligible, unit_i, unit_j, category, g_sem, reason, schema_version, audit_id}`. `eligible=True` cần cả hai cross-pairs được rule xem là mâu thuẫn có điều kiện; single lexical mismatch là chưa đủ.

Strict numeric schema khởi đầu:

1. Hai text unit sequences cùng số unit và chỉ khác một atomic numeric unit; mọi unit khác trùng sau normalization đã khóa.
2. Hai unit parse thành hai giá trị khác nhau, cùng numeric type và cùng semantic slot/đơn vị/thời điểm do rule xác nhận. Không coi date, ordinal, clock time, measurement và identifier là cùng loại.
3. Reject approximation, range, conditional, negation/scope chưa phân giải được, nhiều occurrences của slot, phép liệt kê và lựa chọn “or”. Reject khi token truncation làm mất slot hoặc điều kiện cần thiết.
4. Không dùng mọi integer trong câu như numeric fact: số áo, tên sản phẩm, mã tuyến đường cần rule riêng hoặc reject.
5. Caption annotation vẫn được giả định trung thành với sign; ghi giả định này. Rule không tương đương human verification từng cặp.

Ví dụ synthetic để test parser: `the value is 12 meters` / `the value is 15 meters` có cùng một numeric slot; `between 12 and 15 meters`, `about 12 meters`, `12 meters or 15 meters` phải bị rule khởi đầu từ chối. Đây là fixtures về text parsing, chưa chứng minh cross-negativity của sign video thật.

Lexical schema là một allowlist có version gồm language, canonical units, slot pattern và lý do mutual incompatibility. Chỉ bật sau train audit; không gọi LLM tự sinh nhãn rồi ghi `g_sem=1` như nhãn đã kiểm.

### 9.3. Human audit và trạng thái schema

Tạo sample export theo category/confidence, khoảng 100–200 contrasts cho pilot như một kế hoạch ban đầu. File annotation có: pair ID, language, rating hai positives, rating hai cross-negatives, support correctness, uncertainty, rater ID và notes. Không đưa kết quả student vào annotation template để tránh trộn nguồn đánh giá.

Schema có `audit_status: pending | accepted | rejected`, audit artifact hash và phạm vi áp dụng. Main defaults: strict accepted schema `g_sem=1`, audited lexical schema kém chắc hơn `g_sem=0.5`; pending/rejected `g_sem=0` và không vào semantic-eligible bank. Các giá trị là reliability weights, không phải xác suất đã calibration.

Nếu chưa có audit, agent vẫn hoàn tất code và fixture bank. Có thể xuất weakly validated pilot bank với tên variant rõ; không tự điền một audit artifact giả để mở main training.

### 9.4. Bank schema và failed records

Semantic-eligible pairs tạo **pre-support bank**. Support processing bổ sung fields nhưng không xóa record chỉ vì support thất bại. Lưu rejected semantic proposals ở một audit table khác; H ở loss không đếm những proposals chưa từng được đưa vào pre-support bank.

| Field | Nội dung |
|---|---|
| `pair_id`, `sample_i`, `sample_j` | ID ổn định, endpoints đều thuộc train |
| `text_hash_i/j`, `unit_i/j`, `unit_mapping_hash_i/j` | Target mapping tới frozen text units |
| `category`, `schema_id`, `audit_hash`, `g_sem` | Nguồn quyết định semantic |
| `s0_quartet[2,2]`, `margins0[4]`, `hardness` | Score trước scale và thứ tự margins thống nhất |
| `reference_hash`, `grid_i/j`, `views_i/j` | Provenance của support |
| `q_i`, `q_j`, `h`, `stability_i/j`, `g` | Float32; q có thể null khi failed |
| `retained_mass_i/j`, `rf_union_ratio_i/j` | Diagnostics trước normalize q |
| `support_status`, `failure_reason` | Ví dụ `accepted`, `tiny_text_distance`, `no_positive_mass`, `no_distinct_views`, `low_stability`, `diffuse_support`, `wide_rf` |
| `mining_fingerprint` | B0, split, tokenizer, config, scorer version |

Giữ `q` float32 trong main bank; serialized support nén phải được kiểm mass conservation khi load. H và acceptance rate không được suy ngược từ số non-null q sau khi đã lọc.

## 10. Sinh support phân biệt đối thủ

### 10.1. Positive và differential gates

Từ hai unit vectors `d_i`, `d_j` đặt:

$$
h=\|d_i-d_j\|_2,\quad
s_i^+(n)=u^{ref}_{i,n}\!\cdot d_i,\quad
b_i(n)=\frac{u^{ref}_{i,n}\!\cdot(d_i-d_j)}{h}.
$$

Nếu `h < h_min=1e-3`, support fail cả pair. Đây là threshold engineering để tránh chia cho khác biệt gần zero, không xác nhận hai câu đồng nghĩa.

$$
a_i^+(n)=\operatorname{softmax}_{n\ valid}(s_i^+(n)/\tau_{loc}),\qquad
w_i(n)=a_i^+(n)[s_i^+(n)-\eta_p]_+[b_i(n)-\eta_c]_+.
$$

Main: `tau_loc=0.07`, `eta_p=0`, `eta_c=0.05`. Video j đảo own/rival: positive target là `d_j`, differential direction là `d_j-d_i`. Không dùng cùng dấu differential cho hai videos.

Nếu một view của một endpoint có `sum(w) <= 1e-12`, đánh dấu failed; không epsilon-normalize thành phân phối có vẻ hợp lệ và không fallback argmax. Positive gate tồn tại vì softmax một vector toàn score kém vẫn có tổng bằng 1.

### 10.2. Rebin theo thời gian

Mỗi canonical feature n có center `c_n`. Sort centers tăng dần, không duplicated centers. Bin boundaries là 0, các midpoint `(c_n+c_{n+1})/2`, và duration L. Như vậy mỗi bin B_n ánh xạ đúng một student feature n và bins phân hoạch `[0,L]`.

Với view r, normalize `a_view[k] = w_view[k]/sum(w_view)`. Lấy RF intervals `I_k` đã clip về raw duration, không dùng khoảng padded. Rebin:

$$
p_n^{(r)}=\sum_k a_k^{(r)}\frac{|B_n\cap I_k^{(r)}|}{|I_k^{(r)}|}.
$$

Nếu RF được lưu dưới dạng hợp đoạn rời, tử/mẫu số dùng measure của hợp tương ứng; không double-count đoạn overlap. Implementation đầu có thể dùng bounding intervals bảo thủ nhất quán. Mỗi valid interval có độ dài dương và nằm trong timeline bins, nên tổng p bằng 1 trong tolerance. Không normalize lại một p mất nhiều mass để giấu mapping sai; fail geometry validation trước.

### 10.3. Stability và concentration

Đặt `m=(p_view1+p_view2)/2` và dùng natural log:

$$
\operatorname{JSD}(p_1,p_2)=\tfrac12\operatorname{KL}(p_1\|m)+\tfrac12\operatorname{KL}(p_2\|m),\qquad
c=1-\operatorname{JSD}/\log 2.
$$

Hạng có p=0 đóng góp 0; dùng stable `xlogy` hoặc masked logs. Chỉ clamp rounding drift rất nhỏ về `[0,1]`; sai lệch lớn là lỗi. Hai endpoints đều phải có `c >= 0.7`.

Tạo q từ `p_bar=(p_view1+p_view2)/2`:

1. Sort bins theo `(-mass, canonical_index)`; chỉ xét canonical valid clips.
2. `cap = floor(0.4 * Nv_valid)`. Nếu cap < 1 thì support fail.
3. Lấy prefix ngắn nhất đạt mass ≥ 0.60; nếu không đạt trong cap, lấy tối đa cap bins.
4. Nếu retained mass < 0.50 thì fail; giữ giá trị retained mass **trước** normalize để log.
5. Tính hợp **raw RF của canonical clips được chọn**, không dùng tổng độ dài bins. Nếu union duration / video duration > 0.50 thì fail. Không tự thay support khác để vượt gate trong main profile.
6. q bằng p_bar tại các selected bins, 0 ở chỗ khác, rồi normalize tổng 1.

Với cả hai endpoint accepted:

$$
g_{ij}=g_{ij}^{sem}\min(c_i,c_j).
$$

Nếu bất kỳ support gate fail: `g=0`, q có thể null; lưu reason. Các tỷ lệ 0.6/0.4/0.5/0.7 là pilot defaults, cần sensitivity analysis, không phải ngưỡng khoa học phổ quát.

### 10.4. Reference code cho phần chọn q

```python
import numpy as np

def interval_union_length(intervals):
    intervals = sorted((float(a), float(b)) for a, b in intervals)
    if not intervals:
        return 0.0
    total = 0.0
    left, right = intervals[0]
    assert right > left
    for a, b in intervals[1:]:
        assert b > a
        if a <= right:
            right = max(right, b)
        else:
            total += right - left
            left, right = a, b
    return total + right - left

def select_support(p, canonical_rf, duration, target_mass=0.60,
                   clip_fraction=0.40, min_retained_mass=0.50,
                   max_rf_fraction=0.50):
    # p/rf include valid canonical positions only; reinsert padding afterward.
    p = np.asarray(p, dtype=np.float64)
    assert duration > 0 and np.all(p >= 0) and np.isclose(p.sum(), 1.0)
    assert canonical_rf.shape == (len(p), 2)
    cap = int(np.floor(clip_fraction * len(p)))
    if cap < 1:
        return None, "too_few_clips"
    order = np.lexsort((np.arange(len(p)), -p))
    cumulative = np.cumsum(p[order])
    wanted = int(np.searchsorted(cumulative, target_mass, side="left")) + 1
    chosen = order[:min(wanted, cap)]
    retained = float(p[chosen].sum())
    if retained < min_retained_mass:
        return None, "diffuse_support"
    ratio = interval_union_length(canonical_rf[chosen]) / duration
    if ratio > max_rf_fraction:
        return None, "wide_rf"
    q = np.zeros_like(p)
    q[chosen] = p[chosen] / retained
    return {"q": q, "retained_mass": retained, "rf_union_ratio": ratio}, "accepted"
```

Code mẫu giả định interval validation đã thực hiện và RF dùng bounding intervals. Caller phải truyền các thresholds từ `cfg.support`, không chỉ dựa vào defaults khi chạy sensitivity configs. Không chạy trên video đã mất frame-map provenance rồi coi là localization đúng.

## 11. Batch construction và sampling reproducible

### 11.1. Effective batch

Main student batch chứa 128 **unique paired samples**, gồm endpoints của tối đa 16 contrasts và ordinary fill samples. H=16 contrasts có nhiều nhất 32 unique endpoints, nên còn ít nhất 96 slots để fill nếu đủ dữ liệu.

Sampling mỗi epoch:

1. Shuffle pre-support pair IDs bằng dedicated sampler RNG, không lọc theo g trước sampling.
2. Sample không replacement trong cùng step; không duplicate unordered pair. Default endpoint quota = 4 lần xuất hiện trong contrast schedule mỗi epoch; đây là engineering control chống domination, lưu rejection counters.
3. Với mỗi step, lấy tối đa 16 pairs còn khả dụng và không vượt quota; không repeat một cặp chỉ để đủ H.
4. Union endpoints theo ID; fill ordinary samples không trùng các ID đã có tới batch 128.
5. `H_sampled = len(sampled_pairs)` trước khi bỏ g=0 ở loss. Nếu cuối epoch bank/quota hết, H có thể nhỏ hơn hoặc bằng 0; retrieval vẫn chạy, log rõ.
6. Build `video_id→row`, `text_id→column`, `sample_id→paired positions`; pair loss không dựa vào vị trí đứng cạnh nhau trong batch.

Chuẩn hoá một epoch student là `ceil(N_train / 128)` optimizer steps, ordinary pool cycling có RNG đã lưu khi cần. Vì endpoints làm thay đổi exposure, báo optimizer steps và sample exposure, không chỉ “epochs”. Sinh và lưu cùng `batch_plan` cho A3/A4/A5/A6; sampler không đọc loss/model state.

A2 thuần retrieval dùng ordinary batches theo schedule riêng đã khóa. Có thể thêm A2h dùng hard-partner schedule nhưng không pair/local loss để tách hiệu ứng exposure; A2h là control bổ sung, không thay A3.

### 11.2. g=0 và gradient

Pairs support-failed vẫn có endpoints tham gia ordinary global retrieval trong batch plan. Local và pair terms của chúng bằng 0; không truy cập `unit=-1`, q null hoặc chia h=0. Dùng active index set `g>0` trước khi gather tensors cho auxiliary losses.

Partners trong active loss phải được encode bằng student hiện tại ở canonical view. Không dùng stale student features từ bank. B0/reference/text có thể đọc cache vì frozen và fingerprint khớp.

### 11.3. Positives và candidates trong batch

Xây `P/C` bằng ID relations cho mọi row/column, không chỉ exclude hai cross-pairs được sample. Đảm bảo mỗi training query có ít nhất một positive trong candidate set. Query chỉ có positives và không còn negative đóng góp loss 0 một cách tự nhiên; log số đó. Query không có positive là data/sampler error, không tự bỏ khỏi mẫu số.

## 12. Losses: công thức và indexing chính xác

### 12.1. Common four margins

Với quartet `A[...,2,2]`, thứ tự cố định:

$$
\delta(A)=\left[
A_{00}-A_{01},\quad A_{11}-A_{10},\quad
A_{00}-A_{10},\quad A_{11}-A_{01}
\right].
$$

Hai margin đầu V2T, hai margin sau T2V. Không lặp row margins để giả là bốn directional constraints.

```python
import torch
import torch.nn.functional as F

def margins4(a):
    return torch.stack((a[..., 0, 0] - a[..., 0, 1],
                        a[..., 1, 1] - a[..., 1, 0],
                        a[..., 0, 0] - a[..., 1, 0],
                        a[..., 1, 1] - a[..., 0, 1]), dim=-1)

def smooth_margin(delta, margin, tau):
    assert tau > 0
    return tau * F.softplus((margin - delta) / tau)
```

### 12.2. Global retrieval loss

Cho `logits = S / tau_ret`, `tau_ret=0.07` là pilot default, tách biệt `tau_a`. Không nhân lại `exp(B0.logit_scale)` vì S0 đã unscaled và loss có temperature riêng.

$$
\mathcal L_{V2T}=-\frac1{B_v}\sum_i
\left[\operatorname{LSE}_{j:P_{ij}}(S_{ij}/\tau_{ret})-
\operatorname{LSE}_{j:C_{ij}}(S_{ij}/\tau_{ret})\right],\qquad
\mathcal L_{ret}=\tfrac12(\mathcal L_{V2T}+\mathcal L_{T2V}).
$$

T2V transpose cả scores lẫn P/C, rồi áp cùng hàm. Multi-positive numerator là sum-exp; không lấy mean từng positive cross-entropy vì đó là objective khác.

```python
def directional_retrieval(scores, positives, candidates, tau=0.07):
    assert scores.shape == positives.shape == candidates.shape
    assert positives.dtype == candidates.dtype == torch.bool
    assert bool((~positives | candidates).all())
    assert bool(positives.any(dim=1).all())
    logits = scores / tau
    numerator = torch.logsumexp(logits.masked_fill(~positives, float("-inf")), dim=1)
    denominator = torch.logsumexp(logits.masked_fill(~candidates, float("-inf")), dim=1)
    return (denominator - numerator).mean()

def retrieval_loss(scores, positives, candidates, tau=0.07):
    return 0.5 * (directional_retrieval(scores, positives, candidates, tau)
                  + directional_retrieval(scores.T, positives.T, candidates.T, tau))
```

Native B0 objectives chỉ dùng trong baseline training. Khi student stage bắt đầu, không vô tình cộng lại upstream pose/RGB matching loss hoặc caption augmentation loss ngoài công thức DIVE.

### 12.3. Local quartet

Với active pair `(i,j)`, gather:

- `U_pair`: `[H_active, 2, Nv, d]` từ student ở canonical view.
- `q_pair`: `[H_active, 2, Nv]`, tổng q mỗi endpoint bằng 1.
- `D_pair`: `[H_active, 2, d]`, targets của hai captions.

$$
Z_{ab}=\sum_n q_a(n)\,u_{a,n}^{\psi\top}d_b,\qquad
m_{ij}=0.25\|d_i-d_j\|_2.
$$

Không normalize weighted pooled U. Mỗi video dùng cùng q để chấm hai targets.

```python
def local_loss_active(u_pair, q_pair, d_pair, g_active, text_distance, num_sampled_contrasts,
                      tau_delta=0.1, alpha_margin=0.25):
    z = torch.einsum("han,hand,hbd->hab", q_pair, u_pair, d_pair)
    delta = margins4(z)
    target = alpha_margin * text_distance[:, None]
    per_pair = smooth_margin(delta, target, tau_delta).mean(dim=-1)
    loss = (g_active * per_pair).sum() / max(1, num_sampled_contrasts)
    return loss, z, delta
```

`text_distance` là vector khoảng cách h; `num_sampled_contrasts` là integer H trước support rejection. Giữ hai tên tách biệt trong implementation. Khi active count bằng 0, trả graph-compatible zero từ student, không gọi hàm trên tensor indices không hợp lệ.

### 12.4. Full-caption pair loss và feasibility mask

Gather Q và Q0 bằng batch ID maps, **caption đầy đủ**:

$$
Q_{ab}=S(v_a,t_b),\quad Q^0_{ab}=S_0(v_a,t_b),\quad
\omega_r=\mathbf1\{\delta_r(Q^0)\geq-2\gamma_{train}\}.
$$

$$
\mathcal L_{pair}=\frac1{\max(1,H)}\sum_{(i,j)\ active}
\frac{g_{ij}}4\sum_{r=1}^4\omega_r\,
\tau_R\operatorname{softplus}(-\delta_r(Q)/\tau_R),\qquad \tau_R=0.07.
$$

Main target full-score margin = 0. `omega` từ detached Q0 và gamma_train, không từ student scores và không từ gamma đã calibrate sau training. Nếu comparison bị loại vì không thể sửa trong bound, local loss vẫn được giữ khi support hợp lệ.

```python
def pair_loss_active(q_full, q0_full, g_active, num_sampled_contrasts,
                     gamma_train=0.1, tau_r=0.07):
    with torch.no_grad():
        omega = margins4(q0_full) >= -2.0 * gamma_train
    delta = margins4(q_full)
    per_comparison = smooth_margin(delta, 0.0, tau_r)
    per_pair = (omega.to(delta.dtype) * per_comparison).mean(dim=-1)
    loss = (g_active * per_pair).sum() / max(1, num_sampled_contrasts)
    return loss, omega
```

Mẫu số 4 không đổi thành số comparisons có omega=1. Nếu mọi omega=0, loss đúng bằng 0 nhưng denominator H vẫn như cũ. Không chuyển omega thành một weight trainable.

### 12.5. Optional one-sided preservation

Main `lambda_pres=0`. Nếu mở ablation, tạo easy comparisons train-only: B0 rank đúng, baseline margin vượt threshold đã khóa; endpoints có mặt trong batch hoặc có schedule encode bổ sung được tính vào budget. Lưu easy-set hash.

$$
\mathcal L_{pres}=\frac1{\max(1,|\mathcal A_{step}|)}
\sum_{r\in\mathcal A_{step}}[\delta^0_r-\epsilon_A-\delta_r(S)]_+^2.
$$

Không dùng symmetric Huber/MSE khớp margin vì nó phạt cả sự cải thiện. Không thêm endpoints ngoài batch mà quên tăng compute/exposure khi so controls.

### 12.6. Tổng loss và monitoring

$$
\mathcal L=\mathcal L_{ret}+0.1\mathcal L_{pair}+0.1\mathcal L_{loc}+0\mathcal L_{pres}.
$$

Log unweighted và weighted losses, H sampled/active, mean g trên H, sum g, active omega count, local/full margins và gradient norms. Loss nhỏ vì bank gần như trống không được coi là đã học tốt.

Định kỳ đo cosine giữa gradients của local và retrieval/pair terms trên cùng projector parameter subset. Đây là diagnostic; không tự bật PCGrad hoặc đổi loss weighting nếu không tạo variant riêng.

## 13. Cache, provenance và serialization

### 13.1. Cache fingerprints

| Artifact | Fingerprint tối thiểu |
|---|---|
| RGB local features | Dataset/sample content, RGB extractor weights, frame map, grid/view, crop/preprocess, extraction stage và dtype |
| Native B0 video/text | B0 weights/buffers, native preprocessing/tokenizer, sequence selection, masks, split/sample IDs |
| Text unit features | Text encoder hash, text_model hash, token IDs, unit mapping, normalization/epsilon, dtype |
| Reference local features | Reference weights/buffers, RGB/pose preprocessing, canonical/shifted grid, BN policy, dtype |
| Pair score cache | Feature hashes, scorer version, tau_a/native mixing, score scale convention |
| Contrast bank | Train manifest, B0, reference, schema/audit, units, mining/support configs, timestamps |
| Student gallery cache | Student checkpoint hash + reference/B0/text contracts; không tái dùng giữa epochs |

Validation phải so fingerprint đầy đủ; chỉ trùng filename/shape không đủ. Cache load trả tensor kèm ordered IDs, masks và timestamps. Không concat shards rồi giả thứ tự trùng manifest.

### 13.2. Viết và đọc

Tensor shards có schema/version và index file; write temp rồi atomic rename, chỉ đánh dấu shard complete sau khi checksum thành công. Ghi `float32` cho q/reference correctness cache. Checkpoint đầy đủ có cả model buffers; warm-up/student resume không chỉ lưu projector.

Metadata/raw resource import từ upstream và format nội bộ phải tách rõ. Sau khi chuyển đổi một artifact legacy, lưu checksum nguồn và converter version. Không regenerate toàn bộ cache mỗi run nếu fingerprint không đổi.

### 13.3. Chống rò split qua artifacts

Commands tạo bank/support/schema statistics chỉ nhận manifest train trong API của chúng; assert split của mọi endpoint. Dev artifacts được dùng cho selection/opportunity analysis, không đưa vào bank. Test caption/video cache phục vụ final evaluation có namespace riêng và không được dùng bởi training process.

## 14. Training loop, optimizer và lựa chọn trên dev

### 14.1. Cấu hình optimizer

Main student/warm-up dùng AdamW với hai nhóm: projector LR `1e-4`, trainable pose LR `1e-5`; weight decay `0.01`. Bias và normalization parameters được loại weight decay bằng tên/module type đã kiểm, không bằng một heuristic substring quá rộng. BN affine đã frozen nên không nằm trong optimizer.

Engineering default scheduler: linear warm-up 10% tổng optimizer steps rồi cosine decay về 10% initial LR. Clip global gradient norm 1.0. Fix schedule theo planned steps; checkpoint resume khôi phục vị trí scheduler, không warm-up lại. Baseline B0 giữ optimizer/schedule trong reproduction config riêng.

Correctness run không AMP. Optimization profile chỉ bật AMP sau parity checks, score/loss reductions vẫn FP32. Nếu dùng gradient scaler, lưu scaler state và log skipped steps; scheduled batch/step accounting không được thay đổi âm thầm khi overflow.

### 14.2. Một student step

```python
def student_step(system, batch, sampled, optimizer, cfg):
    # All helpers below must obey their contracts in this specification.
    system.student.train()
    system.enforce_frozen_modes()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        frozen = system.load_or_encode_frozen(batch)
        s0 = frozen.s0_prelogit
        e_ref, pair_valid_ref = evidence_score_block(
            frozen.u_ref, frozen.text_units, batch.vm, batch.tm, cfg.evidence.tau_alignment)

    u = system.student(batch.pose, frozen.rgb, batch.grid)
    e_student, pair_valid = evidence_score_block(
        u, frozen.text_units, batch.vm, batch.tm, cfg.evidence.tau_alignment)
    assert torch.equal(pair_valid, pair_valid_ref)
    scores = compose_score(s0, e_student, e_ref, pair_valid, cfg.train.gamma_train)
    loss_ret = retrieval_loss(scores, batch.positives, batch.candidates, cfg.loss.tau_retrieval)

    # Filter before reading nullable q or invalid target indices.
    active = sampled.active_records()
    loss_loc = u.sum() * 0.0
    loss_pair = scores.sum() * 0.0
    if len(active) > 0:
        tensors = system.gather_active_tensors(u, frozen.text_units, active, batch.id_maps)
        if cfg.loss.lambda_local > 0:
            loss_loc, _, _ = local_loss_active(
                tensors.u_pair, tensors.q, tensors.d_pair,
                tensors.g, tensors.text_distance, sampled.num_sampled,
                tau_delta=cfg.loss.tau_local_margin,
                alpha_margin=cfg.loss.local_margin_alpha)
        if cfg.loss.lambda_pair > 0:
            q_full = system.gather_full_quartets(scores, active, batch.id_maps)
            q0_full = system.gather_full_quartets(s0, active, batch.id_maps)
            loss_pair, _ = pair_loss_active(
                q_full, q0_full, tensors.g, sampled.num_sampled,
                cfg.train.gamma_train, cfg.loss.tau_pair)

    loss_pres = system.optional_preservation(scores, s0, batch, cfg)
    loss = (loss_ret + cfg.loss.lambda_pair * loss_pair
            + cfg.loss.lambda_local * loss_loc + cfg.loss.lambda_preservation * loss_pres)
    system.assert_finite_loss(loss)
    loss.backward()
    system.assert_gradient_ownership()
    torch.nn.utils.clip_grad_norm_(system.trainable_parameters(), cfg.train.grad_clip_norm)
    optimizer.step()
    return system.detached_step_metrics(loss, loss_ret, loss_pair, loss_loc, sampled)
```

Code là loop sketch: `system.*`, `batch.*`, `sampled.*` là APIs agent phải viết, không phải functions có sẵn trong SEDS. Config adapters phải truyền đúng temperatures/margins vào helpers; nếu đổi defaults, không để hàm âm thầm dùng giá trị hard-code cũ. Triển khai A2/A3 phải tránh chạy unnecessary local loss/graph khi lambda_local=0, nhưng vẫn giữ common schedule/metadata của group.

### 14.3. Chọn checkpoint rồi calibrate gamma

1. Evaluate student checkpoint 0 trên dev ở `gamma_train`; nó phải cho baseline score trong tolerance.
2. Cuối mỗi epoch, eval full dev gallery tại **gamma_train cố định**. Endpoint = mean hai R@1, dùng fraction 0–1 nội bộ.
3. Save winner khi endpoint tăng; tie chọn checkpoint sớm hơn. Không dùng test, best training loss hoặc local quartet accuracy để chọn.
4. Sau đủ budget, khóa checkpoint winner. Sweep gamma trên dev `{0, 0.05, 0.1, 0.2}`. Tie chọn gamma nhỏ hơn.
5. Lưu `selection.json` với epoch/step, checkpoint hash, dev IDs/hash, metric, gamma candidates và lựa chọn. Không quay lại chọn epoch khác sau khi thấy calibration trừ khi định nghĩa một variant có joint search trước đó.

Nếu gamma=0 hoặc checkpoint 0 thắng, báo method không chứng minh thêm giá trị trên dev. Không loại gamma=0 khỏi grid để buộc kết quả có residual. Warm-up và A1 có selection rules riêng đã nêu; mọi control cùng loại dùng cùng rule.

### 14.4. Checkpoint schema và resume

Save ít nhất:

- Student weights/buffers; refs tới B0/reference weights có hash xác minh; config/schema versions.
- Optimizer, scheduler, scaler nếu có, epoch, global optimizer step, best dev state và checkpoint 0 metadata.
- RNG state của Python, NumPy, torch CPU/CUDA; sampler RNG, batch-plan hash/cursor và quota counters.
- Data/unit/grid/bank/audit/cache fingerprints, seed, precision, software/git revision và optimizer parameter-group manifest.

Resume phải từ đúng split/config/bank/reference. Nếu fingerprint thay, tạo run mới hoặc explicit migration, không gọi là tiếp tục chính xác. Test interrupted run và uninterrupted run trên deterministic fixture trong tolerance; thông báo nếu backend không hỗ trợ bitwise parity.

## 15. Evaluation và inference deployment

### 15.1. Public inference APIs

```python
class RetrievalSystem:
    def build_video_index(self, video_manifest, checkpoint_bundle): ...
    def build_text_index(self, text_manifest, checkpoint_bundle): ...
    def retrieve_videos(self, text_queries, video_index, top_k): ...
    def retrieve_texts(self, video_queries, text_index, top_k): ...
    def score_pairs(self, videos, texts): ...
```

Video gallery index gồm native B0 video features, student/local reference features, masks/timestamps và checkpoint fingerprints. **Nó không cần caption annotation của video ứng viên để tính S.** Text index gồm native B0 text features và unit features. Unitization query giống train, độc lập gallery/rivals; không chạy mining/support generation khi nhận query mới.

T2V: encode text query một lần, score từng video block. V2T: encode video query một lần, score text blocks. Reference là model khác student nên phải có reference gallery features hoặc encode reference online cho query video. Bỏ reference subtraction trong serving là đổi method.

### 15.2. Exact standard evaluation

1. Load locked bundle, manifest và relevance; assert mọi candidate/query ID được cover đúng một lần theo protocol.
2. Score toàn gallery bằng các pair blocks; dùng một convention `[Nvideos,Ntexts]` xuyên suốt.
3. V2T rank rows theo text candidates; T2V rank columns theo video candidates. Không dùng cùng một row-wise metrics call cho hai chiều mà thiếu transpose.
4. Sort key là `(-score, candidate_id)` với ID order xác định trước. Exact numeric ties dùng secondary ID, không thêm jitter, không dùng `isclose` để tạo ties mới.
5. R@K là tỷ lệ query có ít nhất một annotated positive trong top K. MeanR/MedianR dùng best positive rank theo relevance gốc; rank bắt đầu từ 1.
6. Số query có positive rank luôn bằng số eligible queries; query không có relevance là lỗi protocol cần xử lý trước, không tự loại sau scoring.

Evaluator nên dùng NumPy/CPU và IDs độc lập với model code. Với small fixtures, full sort là oracle. Với gallery lớn, có thể stream counts của candidates đứng trước best positive; tie rule phải đúng y hệt. Top-10 list một mình không đủ tính exact MeanR/MedianR.

Nội bộ metrics lưu fraction; report chuyển phần trăm và ghi rõ đơn vị. Ví dụ `0.625` → `62.5%`, không nhân 100 hai lần.

### 15.3. Duplicates và uncertainty

Exact duplicate text queries vẫn được giữ theo benchmark. Nếu g identical queries có g positives riêng biệt, shared deterministic ranking chỉ đưa một trong các positives lên top-1; báo duplicate-group analysis, không đổi relevance để loại giới hạn này.

So sánh cùng seed/B0/reference bằng paired query outcomes; bootstrap 2,000 replicates là engineering default. Nếu clips dùng chung source, ưu tiên source-cluster bootstrap, giữ gallery fixed và nói rõ giới hạn này. Với endpoint hai chiều, resample linked paired sample/source units để giữ tương quan giữa T2V và V2T. Khi thiếu source IDs, báo query-level assumption.

Báo mean/std qua 3 seeds cùng paired experiment design. Bootstrap CI không thay seed variability; không coi 3 seeds tự bảo đảm statistical power. Practical gain và noninferiority tolerance phải được ghi trong experiment plan trước final test, không chốt từ test outcomes.

### 15.4. Opportunity bound

Trên dev, với single-positive query baseline đang sai, positive p và top wrong candidate c0, tính gap `S0(c0)-S0(p)`. Nếu gap > 2*gamma thì bounded residual không sửa được lỗi đó.

$$
U_\gamma=\frac{\#\{q:\operatorname{rank}_0(p)>1,\ S_0(q,c_0)-S_0(q,p)\leq2\gamma\}}{N_{dev}}.
$$

Báo riêng hai chiều trên full dev gallery và các gamma trong grid. U là bound thô trên **mức tăng R@1 tuyệt đối**, không phải accuracy cuối cùng hay gain dự báo. Với multi-positive protocol, triển khai bound tương ứng best annotated positive hoặc đánh dấu single-positive analysis subset rõ ràng; không áp diagonal formula tùy tiện.

### 15.5. Memory/latency và large-gallery mode

64×512 FP16 = 64 KiB cho một evidence stream/video; student + reference ≈128 KiB/video, chưa tính B0. Đây chỉ là công thức dung lượng. Cache main FP32 dùng gấp đôi phần evidence đó. Đo peak RAM/VRAM, cold/warm latency và số queries với hardware metadata.

Top-K deployment mode có thể shortlist bằng pooled/ANN hoặc B0 rồi rerank bằng S, nhưng phải báo candidate recall và K. Đây là approximation profile; standard benchmark vẫn full-gallery. Không dùng top-K reranking score để tự nhận exact corpus retrieval nếu positive chưa từng được vào shortlist.

## 16. Ablation matrix và kiểm chứng cơ chế

### 16.1. Config variants

| Variant | Score/train differences | Mục đích |
|---|---|---|
| A0 | B0 locked | Controlled baseline |
| A1 | `S0 + gamma * E_ref`; reference warm-up đã có, gamma chọn dev | Thêm model/ensemble có đủ tạo gain? |
| A2 | Centered residual, ordinary batches, chỉ L_ret | Capacity và adaptation |
| A2h, optional | A2 với hard-partner batch schedule | Hiệu ứng exposure riêng |
| A3 | Centered residual, hard-partner batches, L_ret + L_pair | Same-pool full-score control quyết định |
| A4 | A3 + L_loc với random support được match budget | Có cần chọn đúng vùng? |
| A5 | A3 + L_loc, positive-only support giữ absolute gate nhưng bỏ differential factor | Rival specificity có ích? |
| A6 | Full main method | Discriminative local supervision |
| B1 | A6 bỏ L_pair | Bridge ablation |
| B2 | A6 đổi **support teacher** sang contextual B0; E_ref của score vẫn là local reference | Tách ảnh hưởng source tạo q khỏi architecture/identity |
| B3 | A6 dùng span-only targets chỉ cho local loss | Intentional target-space ablation; không coi là main INV-06 |
| B4 | Self-normalized g, no-filter hoặc matched-volume variants | Reliability và selection bias |
| B5 | Thêm one-sided preservation | Easy-pair drift |
| B6 | Adapted EqSim, cùng backbone/pair pool/budget | So prior pairwise consistency regularization |
| B7 | Method v1 với baseline/protocol fixes chung | Kiểm những sửa thiết kế thực sự giúp gì |

A1 giữ ordinary uncentered ensemble theo proposal, không dùng bound `2*gamma` của centered residual để giải thích A1; các giới hạn score của nó cần phân tích riêng.

### 16.2. Comparison groups khi support khác nhau

**Primary pair group:** A3 và A6 dùng toàn main bank, cùng g/omega/batch plan. Đây là comparison chính về local supervision thêm vào full-score training.

**Support mechanism group:** tạo common subset mà random và positive-only support có thể xây theo contract. Chạy A3-common, A4-common, A5-common, A6-common với cùng IDs, g, omega, schedule và budgets. Không so A4 trên subset nhỏ với A6 toàn bank rồi quy khác biệt cho localization.

Random support: giữ số nonzero q và các weight values, chuyển chúng tới random positions; match union-RF trong tolerance một raw frame hoặc một timestamp step đã khai báo. Dùng seed cố định và tối đa 100 proposals/endpoint; nếu không có match, ghi unmatched và loại pair khỏi **toàn comparison group**, không chỉ A4. Original support không được dùng như random fallback. Khi khả thi, match thêm pose quality và coarse temporal position; báo phần không match được.

A5 tính lại positive-only weights qua cùng two-view/rebin/selection pipeline. Nếu gate của A5 fail trên một pair A6 pass, đó là khác biệt coverage cần báo; dùng common group cho clean comparison, không âm thầm dùng g khác giữa variants. Sau khi bank group được khóa, g không phụ thuộc model đang train.

Filter ablation phải có matched-count hoặc matched-total-weight controls để tách chất lượng supervision khỏi lượng training signal. B6/B7 là extensions sau pilot; đọc/đặc tả chính xác phương pháp comparator trước khi gán tên, không dùng một loss tự đoán rồi gọi reproduction.

### 16.3. Mechanism outputs

`analyze` tạo ít nhất:

- Accepted/rejected pairs theo reason/category; số independent videos, targets, signers/sources, endpoint frequency.
- Local và full-caption margins trước/sau, query sai→đúng và đúng→sai theo baseline gap và caption/video length.
- Teacher-only versus student correction; scores dưới random/shuffled support controls.
- Gradient conflict diagnostics; g/H/omega distributions, projection/pose gradient norms, gamma choice.
- Human audit agreement/uncertain fraction và support correctness, tách khỏi pseudo-label accuracy.

Raw intervention phải re-encode cả student/reference, và B0 khi đo full system dưới cùng perturbation. So support deletion với random deletion cùng raw-frame budget; so increment `S-S0` bên cạnh absolute scores. Không chỉ zero một cached vector rồi gọi là raw-video ablation. Deletion có distribution shift và không tự xác định causal effect.

## 17. Config chuẩn và validation

### 17.1. YAML template

Config dưới đây là schema target để agent hiện thực. `null` ở resource paths nghĩa là chưa biết tài nguyên, không phải tài nguyên tùy chọn. `doctor --stage ...` phải báo đúng fields stage đó thiếu; agent tiếp tục làm các milestones không phụ thuộc chúng.

```yaml
spec_version: dive_v2_impl_1_0
run:
  seed: 17
  profile: correctness
  output_root: runs/dive_v2
  variant: A6
  comparison_group: primary_pair

data:
  dataset: how2sign
  train_manifest: null
  dev_manifest: null
  test_manifest: null
  train_relations: null
  relevance_dir: null
  video_root: null
  pose_root: null
  rgb_cache_root: null
  translation_artifact: null
  allow_split_overlap: false

baseline:
  family: seds
  upstream_commit: 434e3f714fcb6a7d1f4001fb9a246bbd93ec0246
  reproduction_config: null
  initial_weights: null
  locked_checkpoint: null
  score_branch: fusion
  dual_mix: 0.5
  score_scale: prelogit

text:
  tokenizer_artifact: null
  normalization_version: text_norm_v1
  unitizer_version: word_numeric_v1
  unit_pool: mean_raw_projected_subwords_then_normalize
  target_source: frozen_contextual_units
  truncate_policy: recorded_native_mapping
  reject_partial_targets: true

temporal:
  canonical_policy: validated_artifact_or_deterministic_dense
  clip_steps: 16
  dense_stride_steps: 1
  max_clips: 64
  support_view_offsets_steps: [-1, 1]
  require_distinct_views: true
  record_raw_frame_map: true
  forbid_unrecorded_global_preprocessing: true

evidence:
  output_dim: 512
  hidden_dim: 1024
  dropout: 0.0
  freeze_bn_statistics: true
  train_bn_affine: false
  normalize_epsilon: 0.000001
  tau_alignment: 0.07
  reference_cache_dtype: float32
  teacher_update: frozen_after_warmup

mining:
  shortlist_per_direction: 128
  rerank_topk: 16
  hardness: negative_min_four_baseline_margins
  semantic_rule_set: strict_numeric_v1
  schema_audit_artifact: null
  shortlist_audit_queries: 128
  shortlist_audit_seed: 701

support:
  h_min: 0.001
  mass_epsilon: 0.000000000001
  tau_localization: 0.07
  eta_positive: 0.0
  eta_differential: 0.05
  min_stability: 0.7
  target_mass: 0.6
  max_clip_fraction: 0.4
  min_retained_mass: 0.5
  max_raw_rf_fraction: 0.5
  strict_schema_weight: 1.0
  audited_lexical_weight: 0.5
  failed_record_policy: retain_with_zero_weight

sampler:
  effective_batch_size: 128
  contrasts_per_step: 16
  endpoint_quota_per_epoch: 4
  unique_sample_ids: true
  preserve_pair_schedule_across_controls: true

loss:
  tau_retrieval: 0.07
  tau_pair: 0.07
  tau_local_margin: 0.1
  local_margin_alpha: 0.25
  pair_target_margin: 0.0
  lambda_pair: 0.1
  lambda_local: 0.1
  lambda_preservation: 0.0
  auxiliary_denominator: sampled_contrasts_before_support_rejection
  pair_feasibility_bound: two_gamma_train

train:
  warmup_epochs: 5
  pilot_epochs: 10
  full_epochs: 30
  budget_mode: pilot
  gamma_train: 0.1
  optimizer: adamw
  lr_projector: 0.0001
  lr_pose: 0.00001
  weight_decay: 0.01
  scheduler: warmup_then_cosine
  warmup_fraction: 0.1
  minimum_lr_fraction: 0.1
  grad_clip_norm: 1.0
  amp: false
  world_size: 1
  gradient_accumulation_steps: 1
  checkpoint_every_epoch: true
  include_initial_checkpoint: true

selection:
  split: dev
  metric: mean_t2v_v2t_r1
  checkpoint_gamma: fixed_gamma_train
  checkpoint_tie_break: earliest_step
  gamma_grid: [0.0, 0.05, 0.1, 0.2]
  gamma_tie_break: smallest_gamma

evaluation:
  gallery: full
  query_chunk: 16
  candidate_chunk: 128
  topk: [1, 5, 10]
  score_layout: video_rows_text_columns
  ties: score_desc_then_candidate_id
  bootstrap_replicates: 2000
  bootstrap_seed: 811
  final_seeds: [17, 23, 42]
  final_experiment_plan: null
```

Resource-specific settings như pose detector, RGB extractor checkpoint, joint convention, native max text length và image preprocessing phải nằm trong resolved adapter config. Không bịa từ generic CLIP defaults; probe và lấy từ artifacts/reproduction config.

### 17.2. Semantic validation của config

Validator phải reject ít nhất:

- `gamma_train<=0` cho centered-residual training; negative gamma ở calibration/inference.
- `output_dim` khác text feature dimension khi chưa có declared shared projection.
- `teacher_update!=frozen_after_warmup`, evidence global Transformer/dropout khác 0 hoặc BN policy trái main profile.
- Nonpositive temperatures, fractions ngoài `[0,1]`, `min_retained_mass>target_mass`, incompatible clip/RF geometry.
- `pair_target_margin != 0` trong main profile; muốn thử target khác phải mở named variant và truyền margin vào loss thật, không để config bị bỏ qua.
- `2*contrasts_per_step>effective_batch_size` trong worst-case unique endpoints.
- `selection.split=test`, bank manifest không phải train, tokenizer/grid/reference hash mismatch.
- Single-view resources khi chạy main two-view support; cho phép riêng variant có tên explicit.
- `amp=true` hoặc `world_size>1` khi optimization profile chưa có parity artifact, nếu run gắn nhãn validated main.
- Unknown keys, silent renamed aliases, repeated conflicting settings từ nhiều config layers.

Export toàn bộ merged config trước khi run. Hash dùng canonical key ordering, không hash YAML text có whitespace/comments không ổn định.

## 18. CLI và artifact contract từng stage

### 18.1. Commands cần hiện thực

Đây là commands của repository **sẽ được agent xây dựng**. Không giả định chúng đã cài sẵn.

| Command | Input chính | Output bắt buộc |
|---|---|---|
| `dive doctor --stage STAGE` | Config, environment | `doctor.json`, capabilities và missing resources theo stage |
| `dive prepare-data` | Raw metadata/assets | Manifests, relevance, data audit, frame maps |
| `dive validate-data` | Manifests/relations | Split/ID/language/temporal/asset checks |
| `dive baseline train` | Train/dev, native initial assets | Controlled B0 checkpoints và adaptation notes |
| `dive baseline validate` | Locked/debug B0 | Adapter score/shape/mask parity và dev results |
| `dive evidence warmup` | Locked B0 + train/dev | Warm-up checkpoints, selected local reference |
| `dive cache build --kind KIND` | Frozen model + split/grid | Indexed feature shards + fingerprint report |
| `dive mine --phase propose/finalize` | B0 + train representations; accepted audit để finalize | Candidate proposals trước audit; semantic/pre-support bank sau audit, shortlist audit |
| `dive audit export` | Proposed train contrasts | Annotation template và sampled IDs; không tự điền ratings |
| `dive support build` | Reference + audited bank + real views | q/g bank, failure diagnostics, RF/support audit |
| `dive sampler build` | Bank + train IDs | Reproducible batch plan và exposure report |
| `dive opportunity` | B0 + dev gallery | U_gamma, baseline margin distributions |
| `dive train` | B0/reference/bank/train/dev | Student checkpoints, losses, dev selection records |
| `dive calibrate` | Selected student + dev | Frozen gamma selection artifact |
| `dive evaluate` | Locked bundle + chosen split | Full-gallery metrics, ranks/IDs, cost report |
| `dive analyze` | Scores/ranks/bank/audits | Paired changes, bootstrap, coverage/mechanism diagnostics |
| `dive export` | Locked B0/reference/student/config | Inference bundle manifest và index API metadata |

Doctor và validation trả nonzero exit khi stage prerequisites không đạt; JSON giữ error codes như `MISSING_DEV_ARTIFACT`, `CACHE_HASH_MISMATCH`, `UNVERIFIED_TEXT_MAPPING`, `NO_DISTINCT_SUPPORT_VIEWS`. Không thay exception bằng warning rồi tiếp tục dưới tên main experiment.

Artifact resolver lưu shared B0/reference/bank tại `output_root/shared/seed{seed}/`, và mỗi experiment tại `output_root/{comparison_group}/{variant}/seed{seed}/`. Mỗi stage ghi output IDs/hashes vào `run_state.json`. Resource path null có thể được resolve từ **output đã xác minh của stage cha trong cùng run**, không từ một checkpoint bất kỳ tìm được trên máy. Resource explicit phải match fingerprint; thiếu audit/tokenizer/initial assets thực sự vẫn là blocker. Nhờ đó runbook không đòi người dùng chép tay đường checkpoint sau mỗi command.

### 18.2. Runbook theo dependency

```bash
python -m pip install -e .
dive doctor --config configs/how2sign_pilot.yaml --stage prepare
dive prepare-data --config configs/how2sign_pilot.yaml
dive validate-data --config configs/how2sign_pilot.yaml
dive baseline train --config configs/how2sign_pilot.yaml
dive baseline validate --config configs/how2sign_pilot.yaml --split dev
dive opportunity --config configs/how2sign_pilot.yaml --split dev
dive evidence warmup --config configs/how2sign_pilot.yaml
dive cache build --config configs/how2sign_pilot.yaml --kind frozen_train
dive mine --config configs/how2sign_pilot.yaml --phase propose
dive audit export --config configs/how2sign_pilot.yaml
```

Sau khi audit artifact thực sự có và config được cập nhật, tiếp tục:

```bash
dive mine --config configs/how2sign_pilot.yaml --phase finalize
dive support build --config configs/how2sign_pilot.yaml
dive sampler build --config configs/how2sign_pilot.yaml
dive train --config configs/how2sign_pilot.yaml --variant A3
dive train --config configs/how2sign_pilot.yaml --variant A6
dive calibrate --config configs/how2sign_pilot.yaml --variant A3 --split dev
dive calibrate --config configs/how2sign_pilot.yaml --variant A6 --split dev
dive analyze --config configs/how2sign_pilot.yaml --split dev
```

Commands cho A1/A2/support-common group được sinh từ variant configs; không reuse output directory gây overwrite checkpoint. Sau pilot/ablation và khi experiment plan đã khóa, mới evaluate test:

```bash
dive evaluate --bundle runs/dive_v2/primary_pair/A6/seed17/locked_bundle.json --split test
dive export --bundle runs/dive_v2/primary_pair/A6/seed17/locked_bundle.json
```

Đường bundle ví dụ là path agent cần tạo; config/runner phải lưu output directory riêng theo variant, seed và comparison group để commands resolve đúng artifact. `evaluate --split test` yêu cầu bundle có config/selection hashes và experiment plan đã khóa; đây là protocol check, không phải một yêu cầu xin phép lại người dùng cho từng lần chạy.

## 19. Milestones để agent thực thi theo thứ tự

Các trạng thái dưới đây là acceptance targets, không phải báo cáo đã chạy. Agent điền trạng thái thật vào `IMPLEMENTATION_STATUS.md` và tiếp tục phần không bị blocker.

| Milestone | Công việc | Acceptance và artifact |
|---|---|---|
| M00 — inventory | Đọc spec, xác minh repo/assets/environment | `doctor.json`, resource map, không đoán paths/versions |
| M01 — skeleton | Config, CLI, schemas, fixture dataset | CLI help/doctor hoạt động; unknown config key bị reject |
| M02 — evaluator oracle | ID-based ranking và two directions | Asymmetric/tie/duplicate/multi-positive fixtures đúng |
| M03 — score/loss core | §8 và §12, dense trước chunked | Numerical và gradient tests của critical kernels pass |
| M04 — data contracts | Manifest, text units, frame mapping, P/C | Offset round-trip, split guards, masks/truncation tests pass |
| M05 — SEDS adapter | Native outputs, prelogit scaling, pre-Transformer feature taps | Real checkpoint parity khi có tài nguyên; debug-only phải ghi rõ |
| M06 — B0 reproduction | Sửa dev protocol, train/select/freeze | Checkpoint có provenance, dev/test routing tests, published vs controlled tách riêng |
| M07 — evidence/reference | Locality, BN policy, warm-up và clone | RF test, frozen gradients, teacher cache và identity initialization pass |
| M08 — mining/audit | Shortlist, proposals, atomic schemas, accepted audit | Coverage report và semantic-eligible pair bank; thiếu audit ghi blocker |
| M09 — support | Two views, rebin/JSD/selection, q/g caching | Geometry/mass checks; support bank và reject reasons đầy đủ |
| M10 — sampler/step | Shared plans, global batch, exact H, training step | H=0/g=0/ID remap tests; frozen params không đổi sau step |
| M11 — end-to-end smoke | Fixture run, resume, rồi tiny real integration khi có assets | Loss hữu hạn, outputs có provenance; không gọi fixture recall là benchmark |
| M12 — primary pilot | A0/A1/A2/A3/A6 trên dev cùng controlled setup | A3 vs A6 paired report, opportunity bound, cost và stop/go record |
| M13 — mechanism | Support-common group và sensitivity/audits | A6-common vs A4/A5-common; raw intervention và bias analysis |
| M14 — final protocol | Seeds/transfer khi phù hợp, lock plan, calibrate, test, export | Locked bundle, exact evaluator, stats và full resource accounting |
| M15 — optimization, optional | AMP/DDP/storage/reranking | Parity + profile-specific metrics; không chặn core single-process khi không cần |

Không trì hoãn M02/M03 vì chưa có sign videos: đây là các bước cần tránh xây một model lớn trên evaluator hoặc loss sai. Không chạy expensive sweep trước M06–M10.

### 19.1. Smoke run đủ ý nghĩa

Synthetic fixture phải có planted video/text association, variable valid lengths, ít nhất một contrast có target biết trước, một support-failed pair, một excluded negative và một tie case. Dùng tiny local projector/encoder để kiểm gradient flow; không thay thế SEDS trong main code path bằng fixture encoder.

Với real resources, tiny integration khoảng 8–32 samples chỉ kiểm decode/adapter/forward/backward/cache/resume và khả năng giảm objective trên tập nhỏ. Không dùng tiny-set overfit làm bằng chứng generalization; tách artifact namespace `smoke` khỏi `pilot/full`.

### 19.2. Research gates sau khi code đúng

| Gate | Bằng chứng cần | Nếu không đạt |
|---|---|---|
| Protocol | Split/metrics/prelogit/cache đúng | Sửa implementation trước method experiments |
| Opportunity | U_gamma và coverage có cơ hội đạt practical gain đã định | Đổi pilot config/hướng nghiên cứu có ghi nhận, không chạy dài mù |
| Supervision | Audit cho thấy cross-pairs/support đủ tin cậy trong phạm vi rule | Abstain/sửa schema; không tăng lambda để ép nhãn yếu |
| Added value | A6 hơn A3 và capacity/ensemble controls trên dev | Chưa bảo vệ được đóng góp local |
| Mechanism | Random/positive-only controls và raw interventions phù hợp claim | Thu hẹp hoặc bác localization explanation |
| Generalization | Setting thứ hai với resource control | Thu hẹp claim theo dataset/backbone |
| Final result | Test chuẩn, thống kê và compute được báo đúng | Không tự nhận SOTA/A* acceptance |

Practical gain, validity target và noninferiority tolerance là các quyết định nghiên cứu cần preregister trong experiment plan sau pilot/train audit, trước final test. Tài liệu không áp một ngưỡng 80% hoặc 0.5 điểm recall như chân lý chung.

## 20. Test plan bảo vệ hành vi nghiên cứu

### 20.1. Tests bắt buộc cho data/scoring/losses

| Test ID | Điều kiểm tra | Oracle/acceptance |
|---|---|---|
| D01 | Train/dev/test IDs và loader routes | Exact manifest IDs; test không cấp selection loader |
| D02 | Unit↔subword offsets | Reconstruct đúng model text/token sequence trên fixtures và real samples |
| D03 | Partial/truncated target | Reject local target; không clamp index |
| D04 | Numeric rule exclusions | Range/approximation/or/scope mơ hồ bị reject |
| D05 | P/C construction | Positives là subset candidates, known ambiguous pair không bị ép negative |
| D06 | Frame maps và stream alignment | Mọi local feature có raw interval và đúng video/grid |
| S01 | Score orientation | Asymmetric score fixture cho R@1 hai chiều khác nhau như oracle |
| S02 | Padding invariance | Thêm arbitrary padded clips/units không đổi valid pair score |
| S03 | All-invalid evidence | Residual bằng 0, không NaN; B0 fallback còn hoạt động |
| S04 | Variable block sizes | Dense và chunked scores bằng nhau trong tolerance |
| S05 | Unit cosine bounds | E và C nằm trong [-1,1] trên unit-vector fixtures |
| S06 | Identity initialization | Clone/reference cùng forward cho S≈S0 tại gamma>0 |
| S07 | Nonzero student gradient | Retrieval-only centered score có gradient không zero trên fixture không suy biến |
| S08 | Prelogit scale | Native fusion logits/scalar scale khớp exported S0 |
| L01 | Four margin ordering | Explicit 2×2 matrix cho đúng 2 row + 2 column comparisons |
| L02 | No pooled normalization | Ví dụ cancellation cho raw mean epsilon, không bị nâng thành 1 |
| L03 | Reliability denominator | H gồm failed samples; giảm toàn g bốn lần làm auxiliary loss giảm bốn lần |
| L04 | Bridge feasibility | Chỉ delta0<-2gamma bị omega=0, order khớp margins4 |
| L05 | Empty active pairs | Auxiliary losses zero, global backward vẫn chạy |
| L06 | Multi-positive retrieval | Logsumexp oracle; không bằng mean-positive CE nói chung |
| L07 | Gradient ownership | Student nhận grad; B0/ref/text/q/g không nhận grad hoặc thay weights |
| L08 | Chunked autograd | Gradients của dense và chunked scorer cùng loss gần nhau |
| L09 | Finite-difference/gradcheck | Tiny FP64 inputs; kiểm scorer/local loss tại điểm tránh gate discontinuity |

### 20.2. Tests bắt buộc cho support/training/evaluation

| Test ID | Điều kiểm tra | Oracle/acceptance |
|---|---|---|
| Q01 | Differential direction | Flip own/rival cho endpoint j đúng dấu |
| Q02 | Absolute positive gate | All negative positive-cosines có thể được relative softmax chọn nhưng main gate reject |
| Q03 | Tiny text distance | h<h_min không tạo NaN/forced support |
| Q04 | Rebin conservation | Sum mass=1 trên valid interval geometry, kể cả overlap |
| Q05 | JSD | Identical p cho c=1; disjoint p cho c=0 |
| Q06 | Concentration cap | Dùng floor, retained mass trước normalize, đúng reject path |
| Q07 | RF union | Overlapping intervals không double-count; raw mapping được dùng |
| Q08 | Shift-view validity | Hai identical grids không được giả là two-view stability |
| T01 | BN mode reset | Sau student.train(), BN vẫn eval; buffers không đổi qua optimizer steps |
| T02 | Locality | Raw perturbation/gradient ngoài declared RF không ảnh hưởng local feature |
| T03 | No shared storage | Update student không thay B0/reference parameters/buffers |
| T04 | Sampler | Unique endpoints trong batch, H/quota đúng, partner indices có thật |
| T05 | Common controls | A3/A6 hoặc support-common group dùng cùng schedule/weights fingerprints |
| T06 | Cache invalidation | Đổi grid/tokenizer/reference/precision làm stale cache bị reject |
| T07 | Resume | Interrupted fixture run gần uninterrupted run và giữ sampler cursor |
| T08 | Config propagation | Đổi non-default tau/lambda/support thresholds qua YAML tác động đúng helper, không mắc default cũ |
| E01 | Ties | Một query chỉ có một rank; stable secondary ID đúng |
| E02 | Multi-positive ranks | Best annotated positive rank theo cùng tie order |
| E03 | Duplicate queries | Group ceiling/phân tích đúng, không tự sửa test relevance |
| E04 | Gamma=0 | Tất cả scores/ranks khớp S0 trong same numeric profile |
| E05 | Gamma/epoch selection | Chỉ dev; tie rules deterministic, checkpoint 0 hợp lệ |
| E06 | Full-gallery completeness | Missing block/duplicate ID/NaN score làm evaluator fail rõ |
| E07 | Export parity | Loaded bundle và training evaluator cho cùng pair scores |
| E08 | Opportunity bound | Không đánh dấu lỗi gap>2gamma là có thể sửa |

Tolerances FP32 ban đầu `atol=1e-6, rtol=1e-5`; gradient parity có thể cần tolerance theo backend nhưng phải ghi lý do. Không tăng tolerance chỉ để che một transpose/mask/index lỗi. Gate/index selection là offline discrete operation, không gradcheck qua hard gates.

### 20.3. Điều không cần test theo kiểu giả lập

Không viết test “recall phải vượt SEDS” với số hard-code, test fake labels từ chính model để chứng minh support đúng, hoặc snapshot loss ngẫu nhiên chỉ mirror implementation. Cần numerical oracles độc lập, integration thật khi có assets và human/sign audit cho câu hỏi ngữ nghĩa.

## 21. Resource failures, optimization và xử lý sai lệch

| Tình huống | Agent phải làm |
|---|---|
| Thiếu B0/pretraining checkpoint | Hoàn tất interfaces/fixtures/tests; ghi tên/hash hoặc nguồn checkpoint cần, không tạo random checkpoint dưới nhãn reproduced B0 |
| Thiếu dev features | Materialize/extract từ dev thật khi có nguồn; không alias test |
| Chỉ có contextual video features | Không gọi đó là local RGB/pose; tìm pre-context tap hoặc ghi missing resource |
| Chỉ có sparse canonical RGB cache | Main two-view support blocked; xây extraction/dense cache khi có raw video; single-view là variant riêng |
| Thiếu frame maps | Reconstruct từ preprocessing có provenance; nếu không thể, không claim raw locality/RF intervention |
| Không có accepted semantic audit | Code và fixture tests vẫn tiếp tục; audit export/proposal artifacts sẵn sàng, main supervision status rõ |
| Few/no eligible contrasts | Báo coverage và failure distribution; không sinh negatives hoặc nới rule âm thầm |
| GPU OOM | Profile, giảm temporary blocks, recompute/checkpoint activations; nếu phải giảm effective batch thì config mới cho mọi control, ghi negative-budget thay đổi |
| Loss NaN | Lưu failing IDs/shapes/masks/stats và dừng affected run; không nan_to_num mọi valid feature |
| Gamma chọn 0 | Báo không có added value trong calibration; giữ kết quả, không bỏ zero candidate |
| A6 không hơn A3 | Không tự thêm OT/generation/loss mới để làm kết quả thắng; error analysis rồi tạo proposal variant nếu có lý do |

### 21.1. DDP là extension có contract riêng

Core correctness ưu tiên `world_size=1`. Nếu thêm DDP:

- Effective global batch vẫn 128 gồm partners; không nhân batch theo số GPUs rồi so như cùng budget.
- Gather student video features phải bảo toàn autograd; detached remote features làm sai gradients. Text/reference/B0 có thể gather không gradient.
- Mẫu số retrieval và H tính toàn global batch, không lấy mean các local-batch objectives rồi gọi tương đương.
- Quy định rõ loss ownership và scaling khi distributed reduction; so một optimizer update multi-GPU với single-process global-batch oracle trên tiny fixture.
- Chia cặp/partners và mapping IDs phải đúng trên tất cả ranks; lưu world size, sampler/global step và resume behavior.

Không claim DDP đúng chỉ vì loss finite. Source custom `allgather` phải được audit, không copy nguyên rồi mặc định backward/scaling phù hợp pipeline mới.

### 21.2. Giới hạn tự động hóa trong nghiên cứu

Agent có thể thực hiện engineering choices trong contract, sửa bugs và chạy các stages có đủ tài nguyên. Những thiếu hụt không thể suy luận từ data đang có — ví dụ độ đúng ngữ nghĩa của sign support hoặc practical effect size cuối cùng — phải được ghi thành nghiên cứu cần xác nhận, không điền bằng suy đoán.

Các resource blockers chỉ dừng stages phụ thuộc chúng. Chỉ yêu cầu thêm tài nguyên cụ thể sau khi những phần code/artifact đã có thể làm được đã hoàn tất và review được.

## 22. Checklist bàn giao và prompt thực thi

### 22.1. Definition of done

Agent hoàn thành **implementation** khi có:

- Repository package cài được, config/CLI có schema và diagnostics rõ; không còn stub ở core data/model/loss/eval paths.
- Data/frame/text/relevance contracts có validators và fixtures; upstream adapter/pinned adaptations có tài liệu.
- Toàn bộ main method, A3 control và support-common variants có cấu hình riêng; seed/budget/weight provenance đầy đủ.
- Test suite bảo vệ các bất biến khoa học; test bị skip vì thiếu resources có reason, không bị cộng vào passed integration.
- Runbook, checkpoint/resume, cache invalidation, dev selection/calibration và inference export thực hiện được.
- `IMPLEMENTATION_STATUS.md`, `DESIGN_DEVIATIONS.md`, `BASELINE_ADAPTATIONS.md`, `REPRODUCIBILITY.md` cập nhật đúng thực tế.

Agent hoàn thành **experiment** chỉ khi thêm actual logs/checkpoints/scores/metrics/audits đủ cho stages đã chạy. Implementation completed và experiment pending/blocked là hai trạng thái hợp lệ khác nhau; không gộp chúng thành một tuyên bố “đã tái lập thành công”.

### 22.2. Format báo cáo cuối của agent

```text
Implemented:
- Modules/stages completed, with repository paths and actual commands.

Verified:
- Test names, commands, pass/skip/fail counts, device/precision.
- Real-data/checkpoint integration separately from synthetic fixtures.

Artifacts:
- Resolved config, fingerprints, checkpoints, banks, metrics and inference bundle.

Research outcome:
- Actual measured dev/test numbers only, with protocol and uncertainty.
- If no real run: explicitly state that effectiveness is unmeasured.

Remaining blockers/deviations:
- Exact missing resource or violated contract; affected milestones and next action.
```

### 22.3. Prompt có thể giao trực tiếp cho AI implementation agent

> Triển khai DIVE-SLR v2 theo toàn bộ `DIVE_SLR_End_to_End_Implementation_Spec.md`. Đọc hợp đồng, bất biến và milestone trước khi sửa code. Bắt đầu bằng inventory tài nguyên, config/CLI, evaluator oracle và numerical kernels; sau đó nối SEDS adapter đã pin, tái lập B0 đúng dev protocol, warm-up reference, mining/audit/support, sampler, student training, calibration, evaluation và export. Giữ đúng score `S0 + gamma * (E_student - E_reference) / 2`, four-direction losses, absolute reliability denominator và timestamp/RF contracts. Dùng các lựa chọn engineering đã chốt, ghi mọi deviation. Hoàn thành mọi bước có thể làm với tài nguyên hiện có; blockers chỉ chặn stages phụ thuộc. Kiểm chứng bằng tests có oracle và actual integration khi có dữ liệu; không gọi synthetic fixtures là benchmark replication. Không thêm module, đổi supervision hoặc chỉnh test protocol để ép kết quả thắng. Duy trì status và artifact provenance để có thể tiếp tục qua nhiều phiên làm việc. Kết thúc bằng implementation status, tests thật đã chạy, artifacts và blockers cụ thể.

## 23. Traceability và nguồn triển khai

### 23.1. Mapping về proposal v2

| Proposal nguồn | Đặc tả này | Nội dung |
|---|---|---|
| §3 | §5–§7 | Baseline fixes, adapter, RF/BN, masks |
| §5.2–§5.3 | §4–§7 | Data/text units và evidence encoder |
| §5.4–§5.5 | §7–§8 | Reference warm-up và centered inference score |
| §5.6 | §9, §11 | Real-pair mining và batch construction |
| §5.7 | §10 | Differential/positive gates, views, support/reliability |
| §5.8–§5.10 | §12, §14 | Local/pair/retrieval/preservation losses và training |
| §5.11–§5.12 | §14, §17–§19 | Runbook, config, pipeline execution |
| §6 | §13, §15, §21 | Opportunity, cost, resources và reproducibility |
| §7–§8 | §16, §19–§22 | Controls, gates, statistical analysis và acceptance |

Proposal v2 là nguồn method đã review. E-01–E-13 và một số scheduler/quota/CLI choices là quyết định triển khai bổ sung của tài liệu này. Chúng phải được lưu trong run config, không được mô tả là hyperparameters đã đạt kết quả tốt nhất.

### 23.2. Các nguồn cần giữ trong repository notes

- [SEDS paper](https://arxiv.org/html/2407.16394v1) và [pinned SEDS repository](https://github.com/longtaojiang/SEDS/tree/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246): primary backbone/method context; các patches của dự án được ghi riêng.
- [Pinned SEDS GCN](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/modules/modeling_gcn.py): kiểm dependency của local pose path.
- [Pinned SEDS data routing](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/dataloaders/data_dataloaders.py) và [How2Sign loader](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/dataloaders/dataloader_H2_retrieval_pose.py): provenance split/mask/grid adaptations.
- [CiCo paper](https://arxiv.org/pdf/2303.12793), [pinned CLCL model](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/modeling.py): backbone transfer và text preprocessing reference.
- [EqSim](https://arxiv.org/html/2303.14465v2): prior art cần đọc chính xác trước khi implement B6; không gán tên EqSim cho một generic quartet loss.
- [PyTorch Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html), [no_grad](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html), [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html): đối chiếu mode/gradient và reproducibility semantics. Environment triển khai vẫn phải lock theo compatibility thực tế.

### 23.3. Mức kiểm tra của chính tài liệu này

Tài liệu được đối chiếu với proposal có SHA ở đầu file và các feature/scoring/loader paths SEDS đã lưu từ lượt audit trước. Python code fences được kiểm cú pháp AST; YAML/JSON được parse; bảng và công thức được kiểm bằng Markdown/KaTeX; các numerical reference cases dùng NumPy. Đây là kiểm tra đặc tả, không phải thực thi toàn bộ PyTorch snippets. PyTorch training, actual checkpoint integration, human sign audit và benchmark experiment chưa được thực hiện trong lượt tạo tài liệu này.

**Mục đích cuối:** agent có thể biết phải viết gì, nhận/trả tensor nào, dùng gradient ở đâu, chạy theo thứ tự nào, kiểm đúng bằng gì và phải báo gì khi không đủ bằng chứng. Tài liệu không tạo một bảo đảm cải thiện SOTA; nó tạo một implementation có thể kiểm chứng phương pháp một cách đúng đắn.
