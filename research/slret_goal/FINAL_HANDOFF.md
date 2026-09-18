# Bàn giao nghiên cứu SLRet — 2026-09-19

## Kết luận

**INCONCLUSIVE_OR_BLOCKED** theo §0(3) của goal: hoàn tất đợt nghiên cứu có
đối chứng nhưng **chưa chứng minh cải thiện retrieval/SOTA**, và chưa đủ bằng
chứng cho contribution thay thế có tính mới/khái quát. Đây là thiếu bằng chứng
khoa học, không phải hiện đang thiếu ổ đĩa/GPU.

Sau khi người dùng dọn ổ, đã hoàn tất trích xuất TRAIN **7.096/7.096 video**,
tiếp tục training và kiểm chứng. Không dừng ở proposal. Không còn job nghiên
cứu đang chạy hay thí nghiệm đã đăng ký còn chờ. Xem COMPLETION_AUDIT.md.

## Kết quả mạnh nhất và điều không đạt

Giữ optimizer moments ở FP32 giảm suy giảm của CiCo khi huấn luyện tiếp từ
cùng checkpoint, với fresh optimizer và cùng exposure. Đơn vị dưới là **điểm
phần trăm mean bidirectional R@1**, FP32 trừ đối chứng native. Không gộp hai
dataset thành một điểm và không chọn seed tốt nhất trên TEST.

| Locked TEST | Endpoint cố định: mean ± sample SD, 3 seed | CI 95% có điều kiện | Checkpoint chọn bằng DEV: mean delta |
|---|---:|---:|---:|
| PHOENIX-2014T | +7.113188 ± 2.473074 | [5.120159, 9.202454] | 0.000000 |
| CSL-Daily | +3.557256 ± 1.533727 | [2.299635, 4.810784] | −0.070862 |

CI bootstrap giữ cố định model/gallery, **không** là CI tổng thể training seeds.
PH dùng 331 nhóm suy ra từ video IDs; CSL dùng 798 caption clusters. Không
chứng minh signer independence. CSL có 795/798 caption IDs trùng DEV và được
đăng ký sau kết quả PH: chỉ là kiểm chứng held-out video, không phải query mới.

Tất cả endpoint FP32 TEST đều bằng hoặc thấp hơn khởi tạo về meanR1. Tất cả
selector PH chọn khởi tạo. CSL selected contrast không cho gain; FP32 seed1337
giảm V2T R@1 0.510204pp so khởi tạo, vượt guardrail 0.5pp. Một số R@5/10 giảm;
đọc toàn bộ bảng. Đây **không** là cải thiện mô hình được chọn, optimizer mới,
hay tái lập nguyên bản điểm paper.

SEDS adapted DEV: FP32 control chọn step111 đạt meanR1 77.649326, chỉ
+0.096339pp so khởi tạo, không qua pilot gate. RGB-tail pilot chọn khởi tạo
77.552987, không vượt control; không cứu bằng sweep. Giả thuyết lỗi fusion
layout bị bác bỏ. Pruning không được nhận vì scoring tối ưu đã dưới ngưỡng
bottleneck đăng ký. UPRet chỉ xác minh checkpoint train dở, không dùng làm
baseline yếu để tuyên bố thắng. Không có novel-method candidate được nhận;
điều này không có nghĩa mọi cơ chế chưa xét đều bất khả thi.

## File và bằng chứng

Đường dẫn tính từ `/home/haipd/SLR`:

- `research/slret_goal/RESULTS.md`: số đo từng seed/chiều và các thất bại.
- `research/slret_goal/PAPER_CASE.md`: claim, phản chứng, outline và thiếu sót.
- `research/slret_goal/COMPLETION_AUDIT.md`: audit từng yêu cầu goal.
- `research/slret_goal/BASELINE_AUDIT.md`, `LITERATURE_AND_PROTOCOLS.md`,
  `NO_GO_REGISTRY.md`, `CANDIDATE_CARDS.md`: phạm vi và lý do chọn/đóng nhánh.
- `research/slret_goal/experiments.jsonl`: ledger append-only, gồm failures.
- `artifacts/slret_goal/cico-ph-test-analysis-001/model_metrics.csv`: 14 dòng,
  7 model × 2 chiều, R@1/5/10 và rank metrics.
- `artifacts/slret_goal/cico-csl-test-analysis-001/model_metrics.csv`: 20 dòng,
  10 model × 2 chiều, đầy đủ metrics.
- `artifacts/slret_goal/cico-ph-locked-test-001/` và
  `artifacts/slret_goal/cico-csl-locked-test-001/`: scores/ranks từng model.
- `artifacts/slret_goal/cico-checkpoint-replay-001/run.json`: fresh-process
  reconstruction cả 12 endpoints, DEV score matrices bit-exact.
- `artifacts/slret_goal/campaign-record-audit-002/run.json`: 81 báo cáo trước
  audit, 71 completed/10 failed, không running/status mismatch. Chín cảnh báo
  schema/wording và sáu command thiếu được công khai trong README.

PH lock SHA256:
`9ca08b6a2c79f7211fb6a05b9fa8166c197ae004e12339c9db91cebc4a3d666a`.
CSL lock SHA256:
`88151788467a2130a2b0c2f2b79650dc9e8986d226ec442bcda7704ff25b74c6`.
CSL TEST manifest SHA256:
`68e5b7456d10daec1bb332b7d8b6d22deb44571a588a992e9a6e68f1458078ce`.

## Kiểm tra và tái lập

Đọc README và protocol tương ứng trước khi chạy. Không ghi đè run IDs cũ.
Các lệnh CPU dưới không train hoặc chọn lại trên TEST; audit tạo artifact mới,
phải thay ID nếu đã tồn tại:

```bash
cd /home/haipd/SLR
/home/haipd/miniconda3/bin/python -m pytest research/slret_goal/tests -q
/home/haipd/miniconda3/bin/python research/slret_goal/tools/campaign_record_audit.py --run-id campaign-record-audit-review-001
/home/haipd/miniconda3/bin/python research/slret_goal/tools/cico_ph_test_analysis.py --run-id cico-ph-test-review-001
/home/haipd/miniconda3/bin/python research/slret_goal/tools/cico_csl_test_analysis.py --run-id cico-csl-test-review-001
```

Kiểm tra cuối: **51 tests passed, 1.51s**; `git diff --check` sạch. Tests chỉ
xác minh cơ học; real-artifact audits/model replay là bằng chứng riêng. Lệnh
GPU đã thực hiện nằm trong run.json/launch.json và README; không tự chạy lại
TEST. Cả PH/CSL TEST đều đã mở, không còn là confirmation chưa xem.

Base Python3.13.5/torch2.11.0+cu128 cho CiCo; env `seds`
Python3.10.21/torch2.3.1+cu121 cho SEDS; RGB bridge dùng cả hai. Xem
environment_snapshot.json và inventory ban đầu. Snapshot là danh sách package,
không phải lock đã kiểm chứng cài mới; không hứa portability bit-exact.
HEAD `0f78470097fce2844897bc7e5d622a4acabeea3b`; dirty UPRet changes có trước
được giữ nguyên, không reset worktree để tái lập.

## Ngân sách và lưu giữ

RUN_BUDGET phân loại chi tiết, gồm smoke/failures và extraction. Discovery/
control 9173.071408s/14400s; confirmation 5334.691536s/21600s, phần lớn là
CSL TEST extraction. Đây là wall-time các nhóm ngân sách, không phải tổng
GPU-active time toàn dự án. CPU audits ghi riêng, không cộng trùng.
Artifacts khoảng 41.45GiB/44GiB; filesystem còn khoảng 114GiB lúc kiểm tra
cuối, reserve tối thiểu 15GiB. Không xóa dữ liệu/artifact để kết thúc đợt.

Giữ scores, manifests, hashes, logs, paired checkpoints và final optimizer/RNG
states. Một số selected SEDS/RGB checkpoints chỉ đủ inference; file final
không tự chứng minh resume đã được kiểm thử. Không upload/commit corpus,
checkpoint lớn hay tài sản riêng tư. Dataset catalogue: `docs/proposal1/datasets.md`;
quyền truy cập không đồng nghĩa quyền phân phối. Không đụng tới job người khác.

## Bằng chứng còn thiếu và bước tiếp theo

1. **Accuracy:** cần cơ chế mới có diagnostic hỗ trợ, không trùng NO-GO; đăng
   ký train-internal selection phù hợp metadata, control mạnh và confirmation
   độc lập. Không tiếp tục FP32/RGB sweep theo hai TEST đã mở.
2. **Cơ chế số học:** full-training từ initialization chưa chọn theo DEV, nhiều
   initialization, SEDS nhiều seed; factorial first/second moments nếu muốn
   nhận diện mediator. Giữ control khởi tạo/frozen và exposure/selector. Các
   thử nghiệm này chưa chạy, cần thiết kế prospectively cho đợt mới.
3. **SOTA/công bố:** xác minh recipe/assets/pretraining provenance và supplement
   còn thiếu; baseline UPRet đầy đủ khi cần, prior-art/domain review độc lập.
   FP32 moments vốn đã biết. Theorem quantization được kiểm tra loại trừ
   underflow, không được áp dụng như chứng minh lý thuyết cho kết quả này.

Skill academic-research-suite hỗ trợ đăng ký/đối chứng, kiểm tra thống kê và
claim–counterevidence audit, giữ tách biệt endpoint mitigation với gain của mô
hình được chọn. Đây không phải chứng nhận độc lập hay đảm bảo paper.
