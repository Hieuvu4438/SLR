# DIVE-SLR: bản phản biện và proposal phương pháp đã chỉnh sửa

**Phiên bản:** Reviewed v2, ngày 07/09/2026.  
**Tài liệu đầu vào:** `Sign_Language_Retrieval_DIVE_SLR_Research_Proposal(2).md`, 885 dòng.  
**Phạm vi:** sentence-level text-to-sign-video retrieval (T2V) và sign-video-to-text retrieval (V2T); ưu tiên How2Sign, sau đó PHOENIX-2014T và CSL-Daily. Giữ các benchmark hiện có.  
**Trạng thái:** đã review proposal, đối chiếu paper và mã nguồn công khai, sửa đặc tả và chạy kiểm tra số học tổng hợp. **Chưa huấn luyện DIVE-SLR, chưa chạy checkpoint benchmark, chưa có kết quả chứng minh vượt SOTA.**

## 1. Quyết định sau review

**Đánh giá: cần sửa đáng kể trước khi chạy thí nghiệm chính.** Giữ hướng học bằng chứng thị giác phân biệt từ hai cặp video–text thật, nhưng chưa chấp nhận bản cũ làm phương pháp hoàn chỉnh để viết các claim thực nghiệm. Các vấn đề quan trọng nằm ở cách tạo supervision, khả năng chuyển tín hiệu cục bộ thành thứ hạng, và tính đúng đắn của pipeline.

Hướng này đáng làm pilot vì câu hỏi nghiên cứu rõ và có thể kiểm chứng bằng các control mạnh. Tuy nhiên, hiện chưa có cơ sở xếp nó là phương pháp tốt nhất. Nếu coi proposal là một submission đã hoàn thành, reviewer có thể bác vì thiếu bằng chứng cho cả hiệu quả lẫn cơ chế; nếu coi là kế hoạch nghiên cứu, có thể tiếp tục sau các chỉnh sửa dưới đây.

Phiên bản đề xuất trong tài liệu này vẫn mang tên **DIVE-SLR — Discriminative Visual Evidence Learning for Sign Language Retrieval**. “v2” chỉ là phiên bản thiết kế, không phải tên một phương pháp đã công bố.

### 1.1. Các thay đổi thực chất so với bản gửi

| Vị trí bản cũ | Vấn đề hoặc khoảng trống | Quyết định trong v2 |
|---|---|---|
| §6–7, novelty | Chưa đối chiếu EqSim và TPM-CL, là các prior art gần hơn generic quadruplet | Bổ sung; thu hẹp claim vào supervision phân biệt tại vùng sign; thêm control thích nghi từ EqSim |
| §8.2, nhánh local | Cắt trước Transformer chưa đủ xác định receptive field; BatchNorm còn có thể trộn thông tin toàn chuỗi khi train | Khai báo khoảng frame tác động, đóng băng thống kê BN, kiểm tra locality ở raw input |
| §8.4, text units | Chưa xử lý đầy đủ khác biệt giữa transcript gốc và caption thật sự đi vào encoder | Lưu cả hai; mining và offset mapping phải đúng ngôn ngữ đầu vào model |
| §8.4, negative validity | Hai cặp thật không bảo đảm hai cặp chéo đều sai | Yêu cầu kiểm tra cả hai chiều, cùng semantic slot; không dùng lexical mismatch mơ hồ làm nhãn âm mạnh |
| §8.5, teacher support | Attention chỉ với từ đúng có thể chọn vùng chung cho cả hai câu | Dùng độ tương hợp với **đúng unit trừ rival unit**, kết hợp positive support và abstention |
| §8.5, teacher locality | Teacher contextual có thể phát tán thông tin sang mọi clip | Tạo teacher local bằng một nhánh warm-up riêng; teacher này đồng thời làm mốc tham chiếu cho residual |
| §8.6–8.7, local score | Normalize vector sau pooling có thể khuếch đại một trung bình rất yếu; local auxiliary khác phép chấm điểm suy luận | Dùng trung bình các cosine trực tiếp; thêm loss trên chính score inference cho cùng quartet |
| §8.6, quartet | Loss quartet phụ là ràng buộc chồng lặp; phản ví dụ ma trận cũ không hiện thực được bằng các unit vector như đã định nghĩa | Bỏ quartet khỏi cấu hình chính; dùng bốn directional margins; thay bằng phản ví dụ cosine hợp lệ |
| §8.5–8.6, confidence | Chia cho tổng confidence làm mất tác dụng giảm độ mạnh khi tất cả confidence cùng thấp | Chia cho số contrast được lấy mẫu, giữ ý nghĩa trọng số tuyệt đối |
| §8.7, khởi tạo | Gamma bằng 0 không làm chết local loss, nhưng có thể làm control chỉ có retrieval loss khởi đầu bất lợi | Residual có mốc tham chiếu, score ban đầu bằng baseline với gamma train dương |
| §8.8, anchor | Huber khớp nguyên margin teacher phạt cả những cải thiện trên cặp dễ | Chuyển thành phạt suy giảm một phía, chỉ mở nếu pilot cần |
| §8.8, duplicate texts | Trùng chuỗi không tự xác nhận mọi cross-pair là positive theo annotation benchmark | Mặc định loại khỏi negatives đã biết không đáng tin; chỉ thêm positive khi có relevance xác nhận |
| §9, mining cost | Một lượt encode không đồng nghĩa tìm hard neighbors bằng late interaction là rẻ | Candidate shortlist bằng pooled features, sau đó chấm lại bằng baseline; đo độ phủ shortlist |
| §10–11, thực nghiệm | Chưa đo giới hạn sửa lỗi do residual bị chặn, và còn thiếu control teacher/ensemble | Thêm upper bound trên dev, control warm-up ensemble, same-pool pair loss, support random và positive-only |

Các chỉnh sửa này làm thiết kế rõ hơn và loại một số con đường thất bại. **Chúng vẫn là giả thuyết thiết kế cần so với bản cũ bằng thực nghiệm**, không phải bằng chứng rằng v2 chắc chắn tăng recall.

## 2. Literature review và mức độ bằng chứng

Tra cứu dùng paper gốc, proceedings, repository của tác giả và trang tác giả/đại học. Đã đọc các phần phương pháp, thiết lập, kết quả và phụ lục liên quan của sáu công trình được yêu cầu. Code được audit tĩnh theo đường gọi cụ thể; không chạy code tải từ repo để suy ra hiệu quả.

### 2.1. Sáu công trình cốt lõi

| Công trình | Nội dung đã xác minh | Phản biện và hệ quả cho DIVE-SLR |
|---|---|---|
| **SPOT-ALIGN**, Duarte và cộng sự, CVPR 2022 | Lặp spotting–retraining để cải thiện I3D; retrieval bằng cross-modal ranking, bổ sung score từ nhận dạng sign. [Paper §3](https://arxiv.org/pdf/2201.02495) | Nhánh spotting bị giới hạn bởi lexicon; không suy ra nhánh embedding chỉ hiểu lexicon đó. DIVE cũng phải hơn một phép late fusion thông thường. |
| **CiCo**, Cheng và cộng sự, CVPR 2023 | Domain-aware/agnostic encoder và CLCL với clip–word interaction; supervision retrieval tổng hợp từ tương tác cục bộ. [Paper §3](https://arxiv.org/pdf/2303.12793) | Không được gọi CiCo là global-only. Ma trận alignment chưa chứng minh localization đúng. Paper dùng dịch caption Đức/Trung sang tiếng Anh để dùng CLIP; đây là yêu cầu tái lập quan trọng. |
| **UPRet**, Wu và cộng sự, ECCV 2024 | Gaussian representations, sampling và OT hỗ trợ training; paper trình bày token-level transport. [Paper §3](https://arxiv.org/html/2405.19689v1) | Variance chưa tự chứng minh uncertainty đã hiệu chuẩn. Không nhận novelty từ OT/uncertainty; cần phân biệt công thức paper với implementation được phát hành. |
| **SEDS**, Jiang và cộng sự, ACM MM 2024 | RGB–pose, CGAF và pose–RGB clip matching. [Paper §3](https://arxiv.org/html/2407.16394v1) | Matching hai stream cùng clip không trực tiếp cô lập nội dung phân biệt một đối thủ. RGB vẫn có thể chứa nonmanual cues; không kết luận SEDS bỏ toàn bộ thông tin mặt. |
| **C²RL**, Chen và cộng sự, arXiv 2024/TCSVT 2025 | ICL dựa trên CLCL và ECL sinh câu; downstream retrieval dùng hai mBART encoder độc lập. [Paper §III](https://arxiv.org/html/2408.09949v1), [journal](https://ieeexplore.ieee.org/document/10933970/) | Generation/context không còn là gap mới. So sánh với SEDS phải ghi backbone, pretraining và compute; không quy toàn bộ khác biệt recall cho objective. |
| **SAN**, Lee và cộng sự, ACL 2026 | Mine sign–word pairs, tìm sign gần nhau với token khác, thay từ tạo caption âm; fine loss dùng score video–caption. [Paper §3](https://arxiv.org/html/2607.09263v1), [ACL](https://aclanthology.org/2026.acl-long.1302/) | Visually similar và semantically incompatible là hai điều kiện khác nhau. Cặp thật tránh một số artifact của caption sinh, nhưng vẫn có false negatives. Phải so local supervision với full-caption loss trên cùng pool. |

Những phản biện ở cột cuối là suy luận của bản review, không phải kết luận rằng các phương pháp trên không thể học discrimination hoặc alignment đúng.

### 2.2. Prior art bổ sung ảnh hưởng trực tiếp đến novelty

| Công trình | Phần liên quan | Điều cần thay đổi trong claim |
|---|---|---|
| **EqSim**, Wang và cộng sự, ICCV 2023 | Từ hai matched pairs tính bốn similarity; regularize quan hệ giữa các chênh lệch. [§3](https://arxiv.org/html/2303.14465v2) | “Hai cặp thật + bốn điểm + khác biệt ngữ nghĩa” chưa đủ mới. EqSim ép tính nhất quán của margins; DIVE đề xuất supervision tại vùng sign. Khác biệt này phải được chứng minh. |
| **DMAE/TPM-CL**, Jiang và cộng sự, ACM MM 2023 | Hard negatives, chọn token theo cross-modal weights, masking và partial-margin ranking trong text–video retrieval. [§3.2–3.3](https://arxiv.org/html/2309.11082v3) | Không claim lần đầu kết hợp token selection với hard-negative discrimination. Masked partial samples của TPM-CL khác real contradictory pairs, nhưng đó mới là phân biệt về thiết kế. |
| **A³PRVR**, Chen và cộng sự, AAAI 2026 | Alignment theo action/object, negative captions và các nhánh alignment trong partially relevant video retrieval. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37271) | “Supervision theo thành phần ngữ nghĩa” cũng có tiền lệ. Không gộp metric PRVR vào sentence-level SLRet. |
| **TripletCLIP**, Patel và cộng sự, NeurIPS 2024 | Học compositional discrimination với negative captions và negative images tổng hợp. [Paper](https://arxiv.org/abs/2411.02545) | Cần phân biệt real-pair supervision với synthetic pairs, thay vì claim chung về negative ở hai modality. |
| **VTaMo**, preprint 07/2026; **DualAnchor**, preprint 07/2026 | Null-token OT và partial OT cho SLT. [VTaMo](https://arxiv.org/html/2607.09126v1), [DualAnchor](https://arxiv.org/html/2607.27614v1) | Null/partial alignment không phải hướng trống. Không dùng BLEU của SLT để suy luận recall SLRet. |
| **SignSeek**, preprint 09/2026 | Articulator saliency và masking cho sign dictionary retrieval. [Paper](https://arxiv.org/html/2609.03695v1) | Generic saliency-guided masking đã có tiền lệ; task và supervision khác sentence retrieval. |
| **CBA**, preprint 05/2026 | Counterfactual bi-directional alignment cho text–video retrieval. [Bản v1](https://www.preprints.org/manuscript/202605.1948) | Việc mask một vùng rồi quan sát score không tự là đóng góp mới hoặc causal identification. Đây là preprint. |

[SignCLIP, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.518/) bổ sung góc nhìn multilingual dictionary pretraining. [GTRN, Neurocomputing 2025](https://ro.ecu.edu.au/ecuworks2022-2026/6028/) dùng visual signing queries để truy hồi corpus. Đây là các nhánh retrieval liên quan, không phải các hàng so recall trực tiếp với setting đã chọn. GTRN được kiểm tra ở mức mô tả chính thức; chưa đọc được toàn văn.

**Khoảng trống chưa giải quyết:** [Causality-inspired multi-grained cross-modal sign language retrieval, CVIU 2026](https://doi.org/10.1016/j.cviu.2025.104631) được xác nhận qua metadata/abstract và [trang tác giả](https://homepage.zjut.edu.cn/yxh2/), nhưng chưa truy cập được toàn văn và bảng kết quả. [HNMA, công bố online 19/08/2026](https://www.sciencedirect.com/science/article/abs/pii/S1077314226002857) cũng cần kiểm tra toàn văn về overlap với hard-negative/multi-grained text–video retrieval. Không suy luận chúng khác DIVE chỉ từ tên bài.

Do đó, bản review **không chứng nhận novelty độc nhất hoặc SOTA tuyệt đối đến ngày 07/09/2026**. Đã dừng mở rộng tìm kiếm khi các quyết định thiết kế chính có đủ nguồn; các khoảng trống toàn văn được giữ lại để xử lý trước submission.

### 2.3. Các mốc số liệu đã đối chiếu

Đơn vị: %. Mỗi ô là **T2V R@1 / V2T R@1**, lấy từ paper, chưa tái lập. Các con số không tạo thành leaderboard đã kiểm soát dữ liệu, backbone và compute.

| Method | How2Sign | PHOENIX-2014T | CSL-Daily | Nguồn |
|---|---:|---:|---:|---|
| CiCo | 56.6 / 51.6 | 69.5 / 70.2 | 75.3 / 74.7 | [Tables 1–3](https://arxiv.org/pdf/2303.12793) |
| UPRet | 59.1 / 53.4 | 72.0 / 72.0 | 78.4 / 77.0 | [Tables 1–3](https://arxiv.org/html/2405.19689v1) |
| SEDS | 62.5 / 57.9 | 76.8 / 78.7 | 85.8 / 85.4 | [Tables 1–3](https://arxiv.org/html/2407.16394v1) |
| C²RL | 62.4 / 57.5 | 78.7 / 77.6 | 90.3 / 88.4 | [Table VI, v1](https://arxiv.org/html/2408.09949v1) |

Với How2Sign, SEDS báo T2V R@1/5/10 = 62.5/75.1/80.1 và V2T = 57.9/70.4/74.9. Cần báo cả R@5/10 để thấy đánh đổi, không chỉ săn R@1. Với CSL-Daily, vượt SEDS chưa đủ vượt mốc C²RL trong các bảng trên.

SAN phải được đọc theo hai protocol khác nhau:

| Backbone trong SAN | Standard T2V R@1, trước → sau SAN | Standard V2T R@1 | Fine V2T R@1 |
|---|---:|---:|---:|
| CiCo | 69.2 → 68.1 | 70.1 → 67.8 | 17.9 → 39.4 |
| GFSLT-VLP | 67.9 → 70.2 | 69.4 → 67.4 | 16.8 → 49.1 |

Nguồn: [SAN, Table 1 và §4.1](https://arxiv.org/html/2607.09263v1). Fine evaluation so caption gốc với 40 negatives biến đổi. Kết quả cho thấy một đánh đổi cần nghiên cứu; không cho phép kết luận SAN luôn làm giảm retrieval chuẩn, hoặc xác định nguyên nhân giảm chỉ từ bảng số liệu.

Hai bất nhất được giữ nguyên thay vì tự sửa bằng suy đoán: SPOT-ALIGN PDF truy cập được, Table 6, báo 32.8/23.3 trên How2Sign, trong khi CiCo trích 34.2/23.6; C²RL v1 có OpenASL T2V R@1 khác nhau giữa prose và Table VI. Không dùng OpenASL làm endpoint của proposal này. [SPOT-ALIGN](https://arxiv.org/pdf/2201.02495), [C²RL](https://arxiv.org/html/2408.09949v1).

## 3. Audit source code: các kết luận có thể dùng để triển khai

### 3.1. Phạm vi và snapshot

| Repo | Snapshot đã kiểm tra | Đường code đã đọc |
|---|---|---|
| CiCo, FangyunWei/SLRT | `38a4f7b00da7a858d59b7fabe5093876a84db8e0` | `CiCo/CLCL/modules/modeling.py`, `main_task_retrieval.py` |
| SEDS, longtaojiang/SEDS | `434e3f714fcb6a7d1f4001fb9a246bbd93ec0246` | Modeling, GCN, SignBERT, CLIP, fusion, training, routing và How2Sign loaders |
| UPRet, xua222/UPRet | `046366227417e1d8ec14145965403462df345984` | Modeling, PDE, Sinkhorn, training route, pooling và metrics |
| SAN, joonmy/SAN | `82aba9cbc1beb403abef6e9a3875ca52479805c8` | Models, datasets, training/evaluation |
| C²RL | Chưa tìm được implementation chính chủ đủ để audit | [sltbaselines](https://github.com/ozgemercanoglu/sltbaselines) là tái triển khai SLT độc lập, không thay thế official SLRet code |
| SPOT-ALIGN | [Repo tìm được](https://github.com/imatge-upc/sl_retrieval) là project page | Không tuyên bố đã audit training implementation |

Các quan sát bên dưới nói về snapshot công khai. **Không có experiment log để kết luận tác giả đã dùng chính code đó tạo các bảng trong paper.**

### 3.2. Chọn checkpoint và routing dev

- **CiCo:** `main()` gọi `eval_epoch(...test_dataloader...)` rồi cập nhật `best_score`; val call bị comment. [Code](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/main_task_retrieval.py).
- **SEDS:** training cũng chọn best từ test. `DATALOADER_DICT` trỏ dev tới hàm test; hàm đó truyền `subset="test"`. How2Sign dataset class chấp nhận tên dev nhưng mapping file chỉ có train/test. Phải sửa cả training call, routing và mapping/manifest; đổi tên loader là chưa đủ. [Training](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/main_task_retrieval.py), [routing](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/dataloaders/data_dataloaders.py), [dataset](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/dataloaders/dataloader_H2_retrieval_pose.py).
- **SAN:** `test_label_path` cấp dữ liệu cho loader dùng chọn checkpoint `dev_best_*`. Tên checkpoint không xác minh split. [Training](https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/train_vlp_v2.py).

DIVE và mọi control phải chạy trên cùng bản nền đã sửa protocol. Báo riêng **published numbers** và **controlled reproduction**; không nhận novelty hoặc method gain từ việc sửa lỗi đánh giá.

### 3.3. Receptive field của pose branch: phát hiện cần sửa ngay

Trong SEDS, `get_sign_output()` chạy `gcn_emb` trên chuỗi pose trước khi cắt các cửa sổ. `ST_GCN_Model.forward()` có hai `TemporalConvNetBlock`; mỗi block có đường kernel 5 hoặc hai kernel 3 nối tiếp, tương đương receptive field tối đa 5 frame ở stride 1. [Modeling](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/modules/modeling.py), [GCN](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/modules/modeling_gcn.py).

**Suy ra từ graph code, ở eval với BN cố định:** hai block tạo RF 9; pooling một cửa sổ 16 frame sau đó có thể phụ thuộc tối đa

$$
R_{\rm pose}=16+(9-1)=24\ \text{frame}.
$$

`sign_conv` xử lý cửa sổ đã cắt, nên không tự lấy thêm frame ngoài hợp của các RF đầu vào cửa sổ đó. Biên video/padding làm khoảng thực tế ngắn hơn. Đây là RF danh nghĩa của đường được đọc, chưa phải đo influence của checkpoint. [SignBERT module](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/modules/modeling_signbert.py).

Trong training, BN tính thống kê qua các frame/batch có thể làm dependency rộng hơn. V2 đóng băng running statistics của tất cả BN trong nhánh evidence; baseline B0 vẫn được tái lập riêng. Preprocessing evidence phải dùng phép chuẩn hóa theo frame/clip hoặc thống kê train cố định; nội suy toàn video và crop thích nghi cần được tính vào dependency nếu sử dụng.

### 3.4. Masking, UPRet và SAN

**Mask trước softmax.** Trong các hàm scoring đã đọc, một số softmax bao gồm các vị trí padding rồi mới mask ở bước aggregate. SEDS còn dùng quy ước video mask `0 = valid`, khác text mask. Adapter phải đổi về một quy ước và kiểm tra bất biến khi padding thay đổi. [CiCo scoring](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/modeling.py), [SEDS scoring](https://github.com/longtaojiang/SEDS/blob/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246/modules/modeling.py).

**UPRet:** active route `flip_similarity_softmax` đặt `sample_num=2`, pooling token trước `einsum`, rồi giải Sinkhorn trên tensor có trục cuối 2×2. Sau đó còn max/mean aggregation; eval bỏ phần OT. `get_similarity_logits` có `pdb.set_trace()`. Không mô tả nhánh này như token-OT đang chạy ở inference. [Modeling](https://github.com/xua222/UPRet/blob/046366227417e1d8ec14145965403462df345984/modules/modeling.py). `compute_metrics` lấy mọi vị trí bằng diagonal score, nên số phần tử rank có thể vượt số query khi ties. [Metrics](https://github.com/xua222/UPRet/blob/046366227417e1d8ec14145965403462df345984/metrics.py).

**SAN:** dataset cần `args.neg_table_name`; model aggregate clip–word scores thành caption scores cho hard-negative loss. Các file đã phát hành không đủ để xem việc tái tạo mining resources và stress evaluation là tự động hoàn chỉnh. Nếu tự viết phần còn thiếu, phải gọi đó là adapted implementation. [Datasets](https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/datasets.py), [models](https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/models.py).

## 4. Phản biện phương pháp cũ ở mức công thức

### 4.1. Local auxiliary chưa bảo đảm giúp inference

Bản cũ **đã có** retrieval loss trên score dùng lúc inference; không phải toàn hệ thống train bằng một score rồi test bằng score khác. Vấn đề hẹp hơn: local auxiliary dùng

$$
z=\operatorname{norm}\!\left(\sum_n q_nu_n\right),
$$

trong khi evidence score inference tổng hợp cosine của toàn bộ clip–word. Local loss có thể giảm mà ảnh hưởng hữu ích lên score này rất nhỏ hoặc xung đột với retrieval loss.

Ví dụ unit vectors $u_1=(\epsilon,\sqrt{1-\epsilon^2})$, $u_2=(\epsilon,-\sqrt{1-\epsilon^2})$, $q=(1/2,1/2)$, $d=(1,0)$. Trung bình cosine với $d$ là $\epsilon$, nhưng cosine sau normalize pooled vector là 1 với mọi $\epsilon>0$. Đây là khuếch đại do normalization, không chứng minh mô hình dùng bằng chứng mạnh.

V2 dùng trung bình cosine trực tiếp, đồng thời đặt một loss tập trung trên **cùng full-caption inference score** của quartet. Điều này giảm sự tách rời giữa hai mục tiêu; vẫn phải đo local-loss gain có chuyển thành retrieval gain hay không.

### 4.2. Positive saliency chưa phải discriminative evidence

Một vùng có thể giống cả hai unit cần phân biệt. Nếu teacher chỉ xét $\langle h_n,d_i\rangle$, vùng ấy vẫn được chọn. Stability không phát hiện được lỗi sai ổn định. Bản cũ đã thừa nhận giới hạn này và có concentration filter; vì vậy không nên nói uniform attention sẽ chắc chắn lọt toàn pipeline. Thiếu sót còn lại là **không trực tiếp đo tính đặc hiệu với đối thủ**.

V2 dùng positive support nhân với phần dương của $\langle u_n^{\rm ref},d_i-d_j\rangle$, có điều kiện từ chối khi mọi vị trí thiếu bằng chứng. Không tạo pseudo-support bằng cách luôn lấy argmax, kể cả khi toàn video không hỗ trợ khác biệt.

### 4.3. Quartet, tính khả thi hình học và độ tin cậy

Đẳng thức

$$
\Delta=Z_{ii}+Z_{jj}-Z_{ij}-Z_{ji}
=(z_i-z_j)^\top(d_i-d_j)
$$

là đúng, nhưng $\Delta$ lớn không buộc mọi positive thắng. Một phản ví dụ **hiện thực được bằng unit vectors** là $d_i=(1,0)$, $d_j=(-1,0)$, $z_i=(1,0)$, $z_j=(0.1,\sqrt{0.99})$. Khi đó

$$
Z=\begin{bmatrix}1&-1\\0.1&-0.1\end{bmatrix},\qquad\Delta=1.8,
$$

nhưng hàng thứ hai vẫn chọn negative. Ma trận $\left[\begin{smallmatrix}1&-1\\0.1&0\end{smallmatrix}\right]$ trong bản cũ không thỏa các ràng buộc cosine đó: hai ô đầu buộc $d_j=-d_i$, nên ô cuối phải là $-0.1$.

Ngoài ra, $\Delta\leq2\lVert d_i-d_j\rVert$. Margin cố định có thể không phù hợp khi text targets rất gần. V2 dùng directional margins thích nghi theo khoảng cách text và bỏ quartet phụ khỏi cấu hình chính. Đây là lựa chọn đơn giản hóa, không phải phát minh một ranking loss mới.

Với confidence, nếu mọi $g$ cùng từ 1 giảm xuống 0.25 thì $\sum g\ell/\sum g$ không đổi. Weighted average vẫn hợp lệ về toán, nhưng không còn ý nghĩa “toàn batch kém tin cậy thì giảm supervision”. V2 dùng mẫu số là số contrast được lấy mẫu; không gọi $g$ là xác suất đúng đã hiệu chuẩn.

### 4.4. Các giới hạn còn nguyên sau sửa

- Khác từ không luôn khác nghĩa; ngay cả khác số cũng cần cùng slot, thời điểm và đơn vị.
- Local teacher có thể sai; đóng băng teacher ngăn tự cập nhật target nhưng không sửa bias ban đầu.
- Text token contextual vẫn mang thông tin ngoài span. Cần control encode span riêng; không mặc định span-only tốt hơn.
- Caption có thể thiếu chi tiết thị giác hoặc là bản dịch không hoàn toàn trung thành. Model không thể khôi phục supervision không có trong annotation.
- Bounded residual không sửa được mọi lỗi nền; bằng chứng đã mất ở feature extractor cũng không thể được một projector khôi phục chắc chắn.

## 5. DIVE-SLR v2: phương pháp đề xuất hoàn chỉnh

### 5.1. Câu hỏi nghiên cứu và đơn vị đóng góp

**Câu hỏi:** trên cùng backbone, dữ liệu, số bước và hard-pair pool, supervision trực tiếp tại vùng sign có khả năng biểu đạt **phần khác nghĩa** có giúp retrieval chuẩn hơn supervision chỉ qua toàn câu không?

Đóng góp dự kiến nằm ở việc kết hợp: real contradictory pairs có kiểm soát, support phân biệt đối thủ từ local teacher, và objective nối local evidence với score retrieval. Từng thành phần ranking, teacher, pose, masking hoặc residual đều có tiền lệ. Chỉ claim đóng góp nếu tổ hợp này giải quyết một failure mode được chứng minh bằng control.

### 5.2. Dữ liệu, notation và hợp đồng preprocessing

Cho $\mathcal D_{\rm tr}=\{(v_i,t_i)\}_{i=1}^{N}$. Ma trận score quy ước **hàng là video, cột là text**. V2T xếp hạng theo hàng, T2V theo cột.

| Ký hiệu | Định nghĩa |
|---|---|
| $B_0,S_0$ | Baseline đã tái lập bằng dev-only selection, frozen; score trước contrastive logit scale |
| $r_{i,n}$ | RGB feature của cửa sổ có timestamp xác định; frozen extractor |
| $p^\psi_{i,n}$ | Pose feature cục bộ; BN statistics cố định |
| $u^\psi_{i,n}$ | Unit-norm local evidence vector |
| $e^0_{i,m}$ | Unit-norm textual unit từ encoder nền frozen |
| $E_\psi,E_{\rm ref}$ | Evidence score student và reference local teacher |
| $(m_i,m_j)$ | Một cặp atomic units tạo khác biệt cùng semantic slot |
| $d_i,d_j$ | Chính $e^0_{i,m_i},e^0_{j,m_j}$, không tạo target bằng encoder khác |
| $q_i,q_j,g_{ij}$ | Support và reliability offline, stop-gradient |
| $\gamma$ | Hệ số hiệu chỉnh có giới hạn, dùng cùng quy tắc lựa chọn cho mọi control |

Mỗi sample phải lưu `sample_id`, `source_video_id` nếu có, timestamp, signer metadata nếu có, `text_original`, `text_model`, language, tokenizer/version, unit-to-subword offsets và valid masks. Không đưa ID, transcript của video đối thủ hoặc metadata nguồn vào evidence encoder.

**Text preprocessing:** unitization phải độc lập với đối thủ và giống nhau lúc train/inference. Bắt đầu với word units; gộp subword rồi normalize. Các biểu thức số hoặc lexical compounds chỉ được gộp thành một atomic unit bằng quy tắc xác định trước. Nếu khác biệt cần nhiều units không tạo được một atomic unit ổn định, bỏ local loss của cặp trong phiên bản đầu.

**Ngôn ngữ:** với CiCo, caption Đức/Trung được dịch sang tiếng Anh trong paper. Không mine span ở câu Đức rồi dùng offset ấy lên CLIP tokens tiếng Anh. Phải lưu translation artifact bất biến và xác nhận khác biệt vẫn tồn tại sau dịch; nếu mapping hoặc nghĩa không rõ, abstain. Với SEDS/UPRet, kiểm nội dung caption artifact thực tế, không suy ra ngôn ngữ chỉ vì dùng CLIP. Nếu dùng native-language backbone như mBART, đó là một cấu hình khác cần so sánh công bằng. [CiCo, implementation details](https://arxiv.org/pdf/2303.12793), [C²RL, preprocessing](https://arxiv.org/html/2408.09949v1).

**Duplicates:** caption trùng hệt sau preprocessing, các đoạn chồng lấn hoặc known-equivalent pairs không được tự động làm hard negative. Mặc định mask các quan hệ đã biết không đáng tin khỏi denominator; chỉ dùng multi-positive numerator nếu có relevance xác nhận. Giữ metric và relevance gốc ở test để so paper, kèm phân tích duplicate riêng.

### 5.3. Backbone và nhánh evidence có locality thật

Nền thực dụng là SEDS; CiCo là backbone transfer control. Giữ B0 và toàn bộ text encoder frozen sau khi chọn bằng dev.

$$
p^\psi_{i,n}=F^\psi_P(k_i;\mathcal I_{i,n}),\qquad
u^\psi_{i,n}=\operatorname{norm}_{\epsilon}\!\left[
W_2\operatorname{GELU}\!\left(W_1\operatorname{LN}([r_{i,n};p^\psi_{i,n}])\right)
\right].
$$

`LN` chạy theo channel tại từng clip. Main configuration dùng $d=512$ với SEDS. $\mathcal I_{i,n}$ chứa toàn bộ raw frames có thể ảnh hưởng output, không chỉ nominal clip center. Với đường SEDS đã phân tích, khai báo RF pose tối đa 24 frame quanh cửa sổ 16 frame; RGB 16 frame nếu extractor thật sự chạy riêng từng clip.

Trong ký hiệu trên, normalization dùng epsilon để tránh chia cho zero. Vector hợp lệ phải có norm trước normalize lớn hơn epsilon; nếu không, dừng và kiểm tra lỗi numerical/collapse, không cho student tự loại clip khó bằng một mask phụ thuộc vào output của nó. Empty clips do dữ liệu được mask bằng quy tắc cố định, giống nhau ở student/reference.

Clone pose branch cho student/reference; không cho optimizer thay đổi B0. Giữ BN ở eval nhưng cho phép học affine parameters nếu đã khai báo; đặt dropout của evidence path bằng 0 trong cấu hình chính để reference/student có forward xác định. Không có Transformer toàn video hoặc cross-attention với text trong evidence encoder. Các chi tiết này áp dụng cả control cùng capacity.

**Kiểm tra bắt buộc khi triển khai:** thay raw frames nằm ngoài $\mathcal I_{i,n}$ không đổi $u_{i,n}$ trong eval; kiểm tiếp graph gradient khi train để phát hiện BN/preprocessing leak. Thay frame nằm trong RF phải có khả năng ảnh hưởng output. RF hữu hạn không tự chứng minh semantic grounding.

### 5.4. Warm-up local teacher và mốc residual

1. Huấn luyện/tái lập B0 bằng train, chọn trên dev, khóa B0.
2. Khởi tạo evidence branch từ các trọng số phù hợp của B0; huấn luyện ngắn bằng retrieval loss trên $E_\psi$ với ordinary train pairs. Chọn reference bằng dev theo budget cố định.
3. Freeze bản này thành $E_{\rm ref}$; cache local reference features và timestamps. Copy trọng số sang student $E_\psi$.
4. Reference dùng để tạo support và làm mốc hiệu chỉnh; không cập nhật bằng local loss.

Đây là chi phí training bổ sung, phải tính vào mọi control. Reference vẫn chỉ nhận video–sentence supervision, nên local correctness cần audit độc lập. Nếu local teacher không tạo support sử dụng được, dừng ở gate đó; không thay bằng contextual support rồi giữ nguyên claim locality.

### 5.5. Score inference: giữ cùng họ late interaction và khởi tạo bằng baseline

Đặt $M_{nm}(v,t)=\langle u^\psi_n(v),e^0_m(t)\rangle$. Với tập clip/text unit hợp lệ $\mathcal V,\mathcal T$:

$$
\begin{aligned}
E_\psi(v,t)=\frac12\bigg[&
\frac1{|\mathcal V|}\sum_{n\in\mathcal V}
\sum_{m\in\mathcal T}
\frac{\exp(M_{nm}/\tau_a)}{\sum_{k\in\mathcal T}\exp(M_{nk}/\tau_a)}M_{nm}\\
&+\frac1{|\mathcal T|}\sum_{m\in\mathcal T}
\sum_{n\in\mathcal V}
\frac{\exp(M_{nm}/\tau_a)}{\sum_{k\in\mathcal V}\exp(M_{km}/\tau_a)}M_{nm}
\bigg].
\end{aligned}
$$

Mọi vector dùng trong cosine đều normalized; mọi mask áp trước softmax. Không dùng padding/BOS/EOS như linguistic units. Nếu không còn unit hợp lệ, trả residual bằng 0 cho sample đó và log, tránh softmax toàn $-\infty$.

$$
C_\psi(v,t)=\frac{E_\psi(v,t)-E_{\rm ref}(v,t)}{2},
\qquad
\boxed{S(v,t)=S_0(v,t)+\gamma C_\psi(v,t)}.
$$

Khi student bằng reference và cùng preprocessing, $S=S_0$ ngay từ đầu. Với $\gamma_{\rm train}>0$, retrieval loss vẫn có gradient đi vào student; không cần học scalar từ biên 0. Dùng gamma train cố định trong mỗi cấu hình; sau training, calibrate gamma trên một grid dev nhỏ, có cả 0. Chọn 0 nghĩa là không chứng minh được giá trị bổ sung.

Do $E_\psi,E_{\rm ref}\in[-1,1]$, ta có $C_\psi\in[-1,1]$. Chênh lệch score giữa hai candidates đổi tối đa $2\gamma$. Factor $1/2$ là cần thiết để giữ bound này; bỏ nó thì bound thành $4\gamma$.

Hai chiều dùng cùng một ma trận pairwise score. Candidate captions không có sẵn ở gallery video tại T2V vẫn không phải vấn đề: mỗi $S(v,t)$ chỉ cần video ứng viên và text query. $q,d_i-d_j,g$ và caption train của đối thủ không xuất hiện trong inference.

**Chi phí:** đây là multi-vector retrieval với một reference score bổ sung, không phải single-vector ANN. Centered residual là lựa chọn ổn định hóa cần ablation với ordinary ensemble; không claim miễn phí hoặc luôn tốt hơn cộng $\gamma E_\psi$.

### 5.6. Mine real contrasts: chỉ train, có kiểm soát chi phí và relevance

1. Encode train. Dùng normalized pooled features để lấy khoảng 128 candidates/anchor ở cả hai chiều; có thể union hai danh sách. Đây chỉ là shortlist xấp xỉ.
2. Chấm lại shortlist bằng $S_0$ late interaction, giữ tối đa 16 đối thủ. Trên một audit subset train, đo tỷ lệ hard neighbors của full $S_0$ được shortlist giữ lại.
3. Deduplicate sample, source overlap và known-equivalent pairs; giảm domination của một video hoặc một transcript template bằng quota mỗi epoch.
4. So transcript thật của hai video. Ghép các units chung theo canonical form, giữ đúng **một atomic substitution cùng semantic slot**. Không xóa phủ định, lượng từ, vai nghĩa hoặc quan hệ chỉ để làm hai câu trông giống nhau.
5. Xác nhận $(v_i,t_j)$ và $(v_j,t_i)$ đều là negative đủ tin cậy cho local supervision. Khác token, khác source hay khác signer không đủ xác nhận điều này.
6. Tạo support ở §5.7 và cache provenance; trường hợp mơ hồ nhận $g=0$.

Ưu tiên các loại đã có quy tắc rõ như giá trị số khác nhau ở cùng đơn vị/thời điểm. Lexical substitutions chỉ được dùng sau audit schema bởi người hiểu ngôn ngữ tương ứng. Không giả định mọi khác biệt noun/verb đều loại trừ nhau. Trong phiên bản đầu, bỏ additions/deletions, đổi scope, đổi vai nghĩa nhưng giữ tập từ, hoặc caption có nhiều occurrences không phân giải được.

**Schema tối thiểu có thể triển khai:** sau unitization, hai dãy phải cùng độ dài và chỉ khác một unit biểu diễn hai giá trị số xác định; mọi unit còn lại trùng nhau, kể cả đơn vị đo và điều kiện thời gian. Loại range, approximation, lựa chọn “hoặc”, conditional và negation chưa phân giải được. Những quy tắc rộng hơn phải là các schema có version riêng đã qua audit. Ngay schema hẹp này vẫn dựa vào giả định caption mô tả trung thành nội dung sign; nó không biến annotation thành nhãn cross-relevance đã được người kiểm từng cặp. Nếu coverage quá thấp, ghi nhận gate thất bại trước khi nới rule.

Độ khó phải đo ở retriever và ở local features riêng. High $S_0$ không tự chứng minh sign-level visual hardness. Báo độ phủ theo lexical category và signer/source; phân tích việc filter giữ lại chủ yếu các trường hợp teacher đã làm tốt.

Mỗi contrast lưu ít nhất: hai IDs, caption gốc/model, unit indices, semantic rule, bốn baseline scores, timestamps, support trước/sau lọc, reliability và lý do loại. Không dùng dev/test làm pair bank, normalization statistics hoặc teacher targets.

### 5.7. Support phân biệt đối thủ, có abstention

Đặt $d_i=e^0_{i,m_i}$, $d_j=e^0_{j,m_j}$ và $h_{ij}=\lVert d_i-d_j\rVert_2$. Nếu $h_{ij}$ gần zero, không có tín hiệu phân biệt hữu ích trong text space đã khóa: bỏ local loss và ghi đây là hạn chế representation, không kết luận hai câu tương đương.

Trong video $v_i$:

$$
s_i^+(n)=\langle u_n^{\rm ref}(v_i),d_i\rangle,
$$

$$
a_i^+(n)=\operatorname{softmax}_{n\in\mathcal V_i}
\left(\frac{s_i^+(n)}{\tau_{\rm loc}}\right),
\qquad
b_i(n)=\frac{\langle u_n^{\rm ref}(v_i),d_i-d_j\rangle}{h_{ij}},
$$

$$
w_i(n)=a_i^+(n)[s_i^+(n)-\eta_p]_+[b_i(n)-\eta_c]_+.
$$

Trong $v_j$, đảo vai $i,j$. Nếu tổng $w$ gần zero ở một phía, bỏ local supervision của cả quartet. Softmax positive chỉ thể hiện ưu tiên tương đối và vẫn chuẩn hóa thành 1 khi mọi clip đều kém liên quan; vì vậy thêm gate tuyệt đối $s_i^+>\eta_p$. Khởi đầu $\eta_p=0$ trên cosine, rồi kiểm trên train audit; không coi 0 là ngưỡng relevance phổ quát. Mục đích của tích trên: vùng cần có positive compatibility đủ mức đã khai báo và nghiêng về unit đúng so với rival. Cả hai điều kiện vẫn có thể bị teacher đánh giá sai.

**Stability:** tạo hai view bằng dịch nhỏ cửa sổ sampling, có raw/dense features và timestamps thật. Mỗi view tính $w$, normalize thành phân phối. Canonical grid chính là grid của student ở loss step: mỗi clip center có một bin $B_n$, ranh giới là midpoint giữa hai centers kề nhau, hai biên ngoài là đầu/cuối video. Các bin không chồng lấn phân chia timeline; $q(n)$ sau rebin vì thế ánh xạ đúng một feature $u_n$. Nếu $I_k^{(r)}$ là raw-RF interval của clip view $r$, dùng

$$
p_n^{(r)}=\sum_k a_k^{(r)}
\frac{|B_n\cap I_k^{(r)}|}{|I_k^{(r)}|},
\qquad \sum_n p_n^{(r)}=1.
$$

Ở đây $a^{(r)}$ là $w^{(r)}$ đã normalize; các interval phải được cắt theo thời gian hợp lệ. Tính $c_i=1-\operatorname{JSD}(p_i^{(1)},p_i^{(2)})/\log 2$, rồi lấy trung bình hai phân phối. Chuẩn hóa/rebin không được so trực tiếp hai token indices khác timestamps.

Tạo support $q_i$: chọn các canonical clip bins có mass cao nhất tới 60% mass, với cap 40% số clip; nếu cap không giữ được 50% mass thì bỏ. Đồng thời kiểm tra **hợp raw RF** của các clip được chọn không vượt 50% độ dài video. Normalize phần mass được giữ để tổng $q_i=1$. Các tỷ lệ là cấu hình khởi đầu, không phải quy luật ngôn ngữ học.

Với video quá ngắn hoặc cache không hỗ trợ hai view có ý nghĩa, không giả tạo stability từ hai lần đọc cùng feature. Có thể chạy biến thể single-view được ghi nhãn riêng; nó không được coi tương đương main configuration.

$$
g_{ij}=g^{\rm sem}_{ij}\min(c_i,c_j),\qquad 0\leq g_{ij}\leq1.
$$

Khởi đầu $g^{\rm sem}=1$ cho schema nghiêm ngặt đã qua audit, 0.5 cho schema lexical được audit nhưng kém chắc hơn, và 0 cho trường hợp chưa xác định. Các số này là hyperparameters. Yêu cầu stability ban đầu $\min(c_i,c_j)\geq0.7$. Báo sensitivity và tỷ lệ bị loại; confidence cao vẫn không thay cho người kiểm tra sign.

Reference, text features, unit indices, $q$ và $g$ đều stop-gradient. Không cập nhật bank bằng chính local objective trong cấu hình chính.

### 5.8. Local loss trên các cosine trực tiếp

Với $a,b\in\{i,j\}$:

$$
Z_{ab}=\sum_{n\in\mathcal V_a}q_a(n)
\langle u^\psi_n(v_a),d_b\rangle.
$$

**Không normalize vector sau pooling.** $Z_{ab}$ chính là trung bình có trọng số của các phần tử trong interaction matrix của student, nên nằm trong $[-1,1]$. Dùng cùng $q_a$ để chấm cả target đúng và rival trên video $a$; không cho mỗi caption tự chọn một vùng dễ khác nhau trong local loss.

Đặt $\phi(x;m,\tau)=\tau\log(1+\exp((m-x)/\tau))$. Hệ số $\tau$ phía ngoài giữ độ lớn derivative theo margin không tăng như $1/\tau$ chỉ vì đổi smoothing.

$$
\ell_4(Z;m,\tau)=\frac14\sum_{a\in\{i,j\}}
\left[\phi(Z_{aa}-Z_{ab};m,\tau)+
\phi(Z_{aa}-Z_{ba};m,\tau)\right],\quad b\ne a.
$$

Có hai row comparisons và hai column comparisons. Chọn $m_{ij}=\alpha_m h_{ij}$, khởi đầu $\alpha_m=0.25$. Với hai text targets unit norm, margin này tôn trọng thang hình học của khác biệt hơn margin chung không đổi. Không claim mọi quartet đều đạt được margin khi evidence bị giới hạn hoặc dữ liệu nhiễu.

$$
\boxed{\mathcal L_{\rm loc}=
\frac1{\max(1,H)}\sum_{(i,j)\in\mathcal H}g_{ij}
\ell_4(Z^{ij};m_{ij},\tau_\Delta)}.
$$

$H$ là số contrasts được lấy mẫu trước khi zero-weight các trường hợp thất bại ở support gate. Nếu không có contrast hợp lệ, loss bằng 0. Không chia cho tổng $g$. Log mean reliability và effective contrast count để phân biệt “loss nhỏ vì học tốt” với “loss nhỏ vì không có supervision”.

Với $g=0$, bỏ tính local target thay vì tạo NaN rồi nhân với 0. Dùng hàm `softplus`/`logsumexp` ổn định để thực hiện $\phi$, không tính trực tiếp exponential lớn. Ngưỡng near-zero của norm, $h$ và tổng $w$ được lưu trong config cùng precision.

Quartet statistic $Z_{ii}+Z_{jj}-Z_{ij}-Z_{ji}$ vẫn hữu ích để phân tích nhưng không có loss phụ mặc định. Ablation thêm lại quartet chỉ đáng làm nếu có lý do từ gradient hoặc error analysis.

### 5.9. Loss nối với full-caption inference score

Tạo $Q_{ab}=S(v_a,t_b)$ với **caption đầy đủ** và công thức §5.5; q và rival units không thay score này. Áp cùng bốn directional comparisons bằng $\ell_4$, đặt target margin khởi đầu $m_R=0$.

Vì residual bị chặn, một comparison có baseline margin $\delta^0<-2\gamma_{\rm train}$ không thể chuyển thành margin dương chỉ nhờ nhánh này. Main bridge bỏ các comparison đó bằng mask $\omega=0$; giữ local loss nếu semantic/support hợp lệ. Với các comparison khác, $\omega=1$. Đây chỉ là điều kiện cần về khả năng sửa, không bảo đảm đạt được.

$$
\mathcal L_{\rm pair}=
\frac1{\max(1,H)}\sum_{(i,j)\in\mathcal H}
\frac{g_{ij}}4\sum_{r=1}^{4}\omega_{ij,r}
\phi(\delta_{ij,r}(Q);0,\tau_R).
$$

**Control quyết định** dùng cùng $\mathcal H,g,\omega$, architecture và training steps, nhưng chỉ có $\mathcal L_{\rm pair}$ và global retrieval loss, không có $\mathcal L_{\rm loc}$. Nếu full method không hơn control này, chưa chứng minh cần local supervision.

Thêm $\mathcal L_{\rm pair}$ không làm local score và full score đồng nhất. Nó cung cấp giám sát trực tiếp lên output cần cải thiện; vẫn cần kiểm tra gradient conflict giữa local và retrieval objectives trên shared parameters.

### 5.10. Global retrieval objective và preservation tùy chọn

Gọi $\mathcal P_i$ là positives đã biết của video $i$, $\mathcal C_i$ là candidates hợp lệ sau loại những quan hệ không đáng dùng làm negatives. Mọi positive phải thuộc $\mathcal C_i$. Chiều V2T:

$$
\mathcal L_{V2T}=-\frac1B\sum_i
\log\frac{\sum_{j\in\mathcal P_i}\exp(S_{ij}/\tau)}
{\sum_{j\in\mathcal C_i}\exp(S_{ij}/\tau)}.
$$

T2V dùng biểu thức tương tự theo cột; $\mathcal L_{\rm ret}=\tfrac12(\mathcal L_{V2T}+\mathcal L_{T2V})$. Không coi paraphrase do model đoán là ground-truth positive.

Nếu cần bảo vệ cặp dễ, dùng train-only set $\mathcal A$ có teacher margin tốt và:

$$
\mathcal L_{\rm pres}=\frac1{\max(1,|\mathcal A|)}
\sum_{r\in\mathcal A}[\delta_r^0-\epsilon_A-\delta_r(S)]_+^2.
$$

Loss này chỉ phạt suy giảm vượt tolerance; không phạt margin được cải thiện. Cấu hình chính ban đầu đặt $\lambda_A=0$, mở khi pilot cho thấy suy giảm cặp dễ đáng kể và chứng minh ích lợi qua ablation.

$$
\boxed{\mathcal L=
\mathcal L_{\rm ret}+\lambda_P\mathcal L_{\rm pair}
+\lambda_L\mathcal L_{\rm loc}+\lambda_A\mathcal L_{\rm pres}}.
$$

Tránh mở đồng thời thêm generation, Gaussian, OT hoặc một text backbone mới trong thí nghiệm kiểm chứng cơ chế này: chúng làm thay đổi câu hỏi và cần protocol riêng.

### 5.11. Pseudocode triển khai

```python
# Algorithm specification; this is not a claim of an executed training run.
base = train_and_select_baseline(train, dev, selection_rule)
freeze_all(base)

local = init_local_evidence(base, freeze_bn_stats=True, dropout=0)
local = warmup_with_retrieval(local, train, dev, fixed_warmup_budget)
reference = freeze_all(copy_model(local))
student = copy_model(reference, trainable=True, keep_bn_stats_fixed=True)

bank = mine_from_train(base, train, shortlist=128, rerank_topk=16)
bank = validate_two_cross_negatives_and_atomic_slots(bank)
bank = attach_discriminative_support(reference, bank, timestamp_views=2)

for step, ordinary in enumerate(train_loader):
    sampled = sample_contrasts(bank, fixed_number=16)
    batch = union_by_id(ordinary, sampled.partner_samples)
    # The effective batch cap includes partners; do not silently enlarge it.
    e_text, rgb = frozen_inputs(base, batch)
    with no_grad():
        s0 = base_pair_scores(batch)
        e_ref = reference_pair_scores(batch, e_text, rgb)

    u = student.local_features(batch.pose, rgb)
    e_student = masked_late_interaction(u, e_text)
    scores = s0 + gamma_train * (e_student - e_ref) / 2

    l_ret = symmetric_retrieval(scores, known_positives, excluded_negatives)
    l_pair = quartet_full_score_loss(scores, s0, sampled, gamma_train)
    l_loc = weighted_local_cosine_loss(u, e_text, sampled)
    l_pres = one_sided_preservation(scores, s0, easy_train_pairs)
    update(student, l_ret + lambda_p*l_pair + lambda_l*l_loc + lambda_a*l_pres)

    if validation_step(step):
        save_by_fixed_dev_rule(student, base, reference)

student, gamma = select_checkpoint_and_calibrate_on_dev(fixed_gamma_grid)
lock_config_and_manifests()
evaluate_full_standard_gallery(base, reference, student, gamma, test)
```

Teacher/reference, text encoder và q không nhận gradient. Partners phải nằm trong batch thực sự encode, không lấy stale features rồi coi như jointly updated examples. Cache reference phải khớp crop/sampling hiện tại; nếu đổi view thì re-encode reference hoặc dùng đúng cache của view đó.

### 5.12. Cấu hình pilot đề xuất, chưa được tối ưu

| Thành phần | Khởi đầu | Điều kiện hoặc kiểm tra |
|---|---|---|
| Dataset/backbone | How2Sign / SEDS | Dev route và ID manifest đã sửa trước |
| Evidence dimensions | 512 | Không thay text backbone trong main experiment |
| Local reference warm-up | 5 epochs | Cùng budget cho mọi evidence control |
| Student training | Pilot 5–10 epochs; full budget 30 | Chốt trước test; nếu kéo dài, kéo dài cả controls |
| Effective batch | 128 gồm partners | Giữ cùng số negatives thực; accumulation không tương đương batch lớn |
| Contrast count | Tối đa 16/step | Log số hợp lệ và tỷ lệ anchor được phủ |
| $\tau_a,\tau_{\rm loc}$ | 0.07 | Khởi đầu theo họ aggregation nền |
| $\eta_p,\eta_c$, stability | 0; 0.05; 0.7 | Positive cosine gate, normalized differential gate và stability; chưa calibration |
| $\alpha_m,\tau_\Delta$ | 0.25; 0.1 | Adaptive local margin; theo dõi text-distance distribution |
| $\tau,\tau_R$ | Thang retrieval phù hợp B0; khởi đầu 0.07 | Phân biệt score trước scale và logits |
| $\lambda_L,\lambda_P,\lambda_A$ | 0.1, 0.1, 0 | Kiểm gradient norms trước khi tăng số loss |
| $\gamma_{\rm train}$ | 0.1 | Calibrate dev trên {0, 0.05, 0.1, 0.2}; grid chung cho controls |
| Optimizer | AdamW; projector 1e-4, pose 1e-5 | Weight decay 0.01, clipping 1.0 là điểm khởi đầu |
| Numeric handling | Softmax/logsumexp FP32 | Guard zero norm, empty masks, tiny $h_{ij}$ |
| Seeds | 3 cho kết quả chính | Báo mean/std; paired runs chia sẻ B0/reference theo seed |

Không coi bảng này là evidence về khả năng hội tụ. Khóa một grid nhỏ trong pilot; chỉ mở rộng khi đã xác định failure mode cụ thể.

## 6. Tính khả thi và giới hạn cải thiện

### 6.1. Đo upper bound trước khi chạy dài

Với query dev $q$ có positive duy nhất $p$ và baseline top candidate $c_0$, đặt:

$$
U_\gamma=\frac1{N_{\rm dev}}\sum_{q:\operatorname{rank}_0(p)>1}
\mathbf1\{S_0(q,c_0)-S_0(q,p)\leq2\gamma\}.
$$

Trong notation này score được đọc theo chiều query thích hợp. $U_\gamma$ là **upper bound thô** cho mức tăng R@1 tuyệt đối của bounded residual: chỉ những lỗi này có thể được sửa; một số query đúng còn có thể bị làm sai. Điều kiện này không đủ để sửa được vì phải thắng toàn gallery và correction bị ràng buộc bởi encoder.

Nếu $U_\gamma$ quá nhỏ so với gain có ý nghĩa mà nhóm đặt trước, không nên huấn luyện dài với cùng bound. Có thể tăng gamma trong pilot hoặc đổi hướng fine-tuning; phải khai báo thay đổi và đánh giá lại drift/compute, không nới sau khi nhìn test.

### 6.2. Chi phí được tính đúng

Một stream evidence có 64 vectors × 512 dimensions × FP16 tốn 64 KiB/video. Centered residual cần cả student và reference: tối đa khoảng **128 KiB/video**, tức khoảng **122 GiB cho một triệu video**, chưa tính B0, text cache và overhead. Đây là phép tính dung lượng, không phải memory benchmark đã chạy.

Với batch 128, 64 clips và 32 units, một tensor interaction FP32 có $128^2\times64\times32$ phần tử, khoảng 128 MiB. Activations/backward làm tổng VRAM lớn hơn. B0/reference có thể chấm theo chunks không gradient; student cần chunking phù hợp autograd. Exact all-pairs mining có chi phí $O(N^2N_vN_td)$; shortlist làm giảm chi phí nhưng phải kiểm coverage.

Báo riêng baseline training, warm-up reference, mining/support extraction, student training, storage và latency full-gallery. Không ước tính GPU-hours khi chưa biết hardware, I/O và checkpoint. Nếu dùng top-K reranking để giảm inference cost, phải báo candidate recall và xem đó là một protocol triển khai riêng.

### 6.3. Rủi ro dữ liệu và supervision

How2Sign filtered counts trong SEDS là 31,019/1,738/2,348; CiCo/C²RL mô tả 31,085/1,739/2,348; UPRet mô tả split gốc 31,164/1,740/2,356. Kiểm danh sách ID và caption artifact, không chỉ so tổng số mẫu. [SEDS](https://arxiv.org/html/2407.16394v1), [CiCo](https://arxiv.org/pdf/2303.12793), [C²RL](https://arxiv.org/html/2408.09949v1), [UPRet](https://arxiv.org/html/2405.19689v1).

Ghi riêng BSL/sign-recognition pretraining, SignBERT, CLIP/mBART, pose detector, machine translation và mọi human audit. “Gloss-free downstream” không đồng nghĩa không có sign/gloss supervision trong lịch sử pretraining. Nếu người hiểu sign xem các contrast để hiệu chỉnh rule/threshold, phải công bố công sức này; không mô tả pipeline là hoàn toàn không có hỗ trợ thủ công.

## 7. Thiết kế thực nghiệm để chấp nhận hoặc bác bỏ giả thuyết

### 7.1. Endpoint và protocol

- Endpoint chính: How2Sign mean(T2V R@1, V2T R@1); báo riêng từng chiều cùng R@5/10, MeanR/MedianR, memory và latency.
- Chọn checkpoint bằng một dev rule cố định cho mọi method/control. Lock toàn bộ cấu hình trước đợt test cuối; không dùng test để chọn seed, epoch, gamma, thresholds hay ablation thắng.
- Dùng standard gallery đầy đủ và relevance gốc. Query IDs, gallery IDs, orientation score, tie policy và counts phải được kiểm bằng evaluator độc lập.
- Tái lập cùng preprocessing và nguồn tài nguyên. Các số paper khác tài nguyên chỉ là contextual comparison, không phải controlled method gain.
- Ba seeds chính: mỗi seed có B0/reference dùng chung giữa các control. Nếu chỉ cố định một B0, ghi seed variation là conditional trên baseline đó.
- Chuyển giao sang ít nhất một dataset hoặc backbone thứ hai trước claim tổng quát. Giữ riêng ASL, DGS và CSL; không suy thành một ngôn ngữ ký hiệu phổ quát.

### 7.2. Ablation ưu tiên theo giai đoạn

| ID | Cấu hình | Điều cần phân biệt |
|---|---|---|
| A0 | B0 tái lập với protocol đúng | Nền thực tế |
| A1 | B0 + ordinary ensemble của local warm-up reference, gamma chọn dev | Gain chỉ từ thêm model hoặc warm-up? |
| A2 | Centered-residual architecture, chỉ $\mathcal L_{ret}$ | Capacity và adaptation thông thường |
| A3 | A2 + cùng hard-pair bank và $\mathcal L_{pair}$ | Hard exposure/full-score discrimination |
| A4 | A3 + local loss, support random được match kích thước và union-RF | Supervision cần đúng vùng hay chỉ thêm loss? |
| A5 | A3 + local loss, positive-only teacher support; giữ positive gate, bỏ differential factor | Phần trừ rival có thêm ích lợi? |
| A6 | DIVE-SLR v2 đầy đủ | Có hơn A1–A5 trên endpoint chuẩn? |
| B1 | A6 bỏ $\mathcal L_{pair}$ | Bridge cần thiết hay chỉ thêm complexity? |
| B2 | A6, dùng contextual teacher thay local teacher | Local teacher và locality có đóng góp? |
| B3 | A6, span-only text targets | Gain phụ thuộc contextual text leakage đến mức nào? |
| B4 | A6, confidence self-normalized hoặc không filter | Reliability giúp chất lượng hay chỉ đổi số mẫu? |
| B5 | A6 thêm one-sided preservation | Có giảm query đúng → sai mà không mất gain? |
| B6 | Adapted EqSim trên cùng pair bank và architecture | Local evidence hơn pairwise similarity regularization đã biết? |
| B7 | Bản DIVE cũ được triển khai với cùng fixes nền | Những sửa thiết kế nào thực sự hữu ích? |

A3/A4/A5/A6 phải dùng cùng sample IDs, partners, weights, steps và seeds. Khi ablate filter, báo thêm control **match số contrast hoặc tổng trọng số**; nếu không, khó phân biệt chất lượng với lượng supervision. Khi teacher/support khác nhau, làm thêm subset chung để tránh gain chỉ do đổi dữ liệu được giữ.

Chạy pilot A0–A6 trước; chỉ mở B khi có tín hiệu. EqSim adaptation cần theo công thức và resource budget đã khai báo; không coi đó là nguyên kết quả paper EqSim. SAN và TPM-CL là các đối chiếu bổ sung nếu triển khai được đầy đủ, nhưng không thay thế same-pool control A3.

### 7.3. Kiểm tra cơ chế, tránh đánh giá bằng chính pseudo-label

1. **Validity audit:** lấy mẫu train theo category và confidence; người biết đúng ngôn ngữ sign đánh giá hai positives, hai cross-negatives và vùng support. Report uncertain cases, disagreement và mức đồng thuận. Khoảng 100–200 contrast cho pilot là kế hoạch lấy mẫu ban đầu, không bảo đảm statistical power.
2. **Coverage:** tỷ lệ anchor có cặp, số video/lexical units độc lập, tần suất partners, độ phủ signer/source và phần video bị support chiếm. Không chỉ báo hàng nghìn cặp cùng một template.
3. **Retrieval behavior:** đếm query sai → đúng, đúng → sai, gain theo margin B0, độ dài caption/video và loại khác biệt.
4. **Localization controls:** random và shuffled support phải match số clip, độ dài RF, pose quality và khi khả thi vị trí thời gian. Giữ contrast pool cố định.
5. **Raw-input intervention:** xóa/che support và vùng random cùng budget, re-encode cả student và reference, và B0 khi đánh giá hệ thống đầy đủ. So chênh lệch hai caption, đồng thời so phần gain của S so với S0 dưới cùng phép can thiệp.
6. **Context leakage:** thay phần caption chung hoặc encode span riêng để xem discrimination còn giữ hay không. Chỉ xem attention map đẹp hoặc pseudo-quartet accuracy cao là chưa đủ.
7. **Optimization:** log gradient norms/cosine giữa local và retrieval losses, mean reliability, số active margins, gamma đã chọn và drift trên easy pairs.

Raw deletion có thể gây distribution shift; video sau xóa không có bản dịch âm đã biết. Đây là sensitivity analysis, **không phải causal identification**. Local teacher, student và metric dựa trên cùng pseudo labels không phải ba nguồn xác nhận độc lập.

### 7.4. Ties, duplicate queries và thống kê

Nếu một nhóm có $g$ text queries hoàn toàn giống nhau, nhưng benchmark gán $g$ paired video IDs khác nhau và retriever chỉ dùng nội dung query, cùng một deterministic ranking không thể đưa cả $g$ positives khác nhau lên top-1. Tỷ lệ T2V top-1 tối đa của nhóm đó là $1/g$. Phân tích này không cho phép đổi relevance test để tăng số báo cáo; nó yêu cầu báo duplicate groups và tie policy minh bạch.

Báo paired bootstrap confidence interval cho chênh lệch R@1 trên cùng queries. Khi có nhiều clips chung video nguồn, bootstrap theo source clusters và nói rõ giả định; conditional query bootstrap giữ gallery cố định không đo sự bất định do thay gallery. Seed standard deviation đo loại biến thiên khác. Với endpoint gộp hai chiều, preserve liên kết query/sample khi resample, không coi hai chiều là độc lập hoàn toàn.

Chốt mức gain có ý nghĩa thực tế và noninferiority tolerance cho chiều còn lại trước test. Không coi CI chứa zero là bằng chứng thắng chắc chắn; cũng không coi ba seeds là bảo đảm power cho mọi effect size.

### 7.5. Các gate quyết định

| Gate | Cần quan sát | Quyết định khi không đạt |
|---|---|---|
| G0: protocol | Dev routing, manifests, padding, score orientation/ties đúng | Sửa pipeline trước |
| G1: opportunity | $U_\gamma$ và hard-pair coverage đủ cho mục tiêu gain đặt trước | Đổi bound/setting ở pilot hoặc bỏ residual approach |
| G2: supervision | Cross-negative validity và support có độ chính xác dùng được qua audit | Tăng abstention, sửa schema; không tăng loss để bù nhãn sai |
| G3: pilot | A6 hơn A1, A2 và A3 ổn định trên dev | Nếu chỉ hơn A0, chưa có bằng chứng cho đóng góp local |
| G4: mechanism | A6 hơn support random/positive-only và có sensitivity nhất quán | Nếu random ngang bằng, bác hoặc thu hẹp localization claim |
| G5: generalization | Có ích ở setting thứ hai với control tài nguyên | Thu hẹp claim theo ngôn ngữ/backbone |
| G6: final | Standard test cải thiện, uncertainty và compute chấp nhận được | Không tuyên bố vượt SOTA hoặc đạt mục tiêu ban đầu |

Các ngưỡng coverage 5% hay validity 80% trong bản cũ chỉ là heuristics, không phải tiêu chuẩn khoa học phổ quát. Nên dựa vào số contrasts độc lập, độ phủ lỗi có thể sửa, human audit và gain pilot để quyết định. Nếu chỉ stress test tăng còn standard retrieval giảm, mục tiêu hiện tại chưa đạt.

## 8. Phản biện dự kiến của reviewer

| Câu hỏi khó | Bằng chứng phải trả lời |
|---|---|
| Có gì hơn EqSim hoặc hard-negative ranking? | A3 và B6 cùng pool/compute; local-support ablations |
| Gain đến từ teacher warm-up hoặc ensemble? | A1 và A2; tổng training cost gồm warm-up |
| Teacher đã hiểu đúng thì student chỉ học lại? | Coverage theo teacher error/local specificity; independent support audit; so teacher-only score |
| Cặp thật khác ngữ cảnh/signers quá nhiều? | Slot validation, source/signer stratification; thừa nhận không phải controlled counterfactual |
| Tại sao gọi nhánh này local? | RF interval, frozen BN, preprocessing contract và raw-input perturbation |
| Reference subtraction có triệt luôn tín hiệu tốt? | So ordinary ensemble/uncentered scoring; error analysis phần giữ và mất |
| Local loss học được nhưng không chuyển lên full sentence? | A3/B1, full-score margin changes và endpoint chuẩn |
| Claim semantics quá mạnh với frozen contextual text? | Span-only và common-context controls; giới hạn lexical/local claim |
| Loss nhiều, hyperparameter search không công bằng? | Grid/budget chung, main chỉ mở modules đã vượt gate |
| Chưa kiểm công trình mới nhất? | Nêu rõ toàn văn CVIU/HNMA còn thiếu; không claim độc nhất trước khi hoàn tất |

**Đánh giá rủi ro sau sửa:** rủi ro novelty vẫn đáng kể vì họ ý tưởng gần EqSim/TPM-CL/SAN; rủi ro hiệu quả cao nhất nằm ở pseudo-support validity, lượng real contrasts và thang correction cần thiết. Bản v2 giúp đo các rủi ro này rõ hơn, không loại bỏ chúng bằng lập luận.

## 9. Những gì đã kiểm tra trong lượt review này

Đã chạy các kiểm tra số học tự xây dựng bằng NumPy, không dùng benchmark data:

| Kiểm tra | Quan sát |
|---|---|
| Counterexample quartet với unit vectors | Ma trận [[1, −1], [0.1, −0.1]], delta = 1.8 nhưng một positive vẫn thua |
| Normalization sau pooling | Raw mean match = 0.0001, normalized pooled match = 1 trong ví dụ tổng hợp |
| Positive-only và differential support | Một vùng chung có positive score bằng vùng phân biệt; differential support chọn được vùng phân biệt trong ví dụ 2D |
| Absolute positive gate | Relative support vẫn có mass khi mọi positive cosine âm; gate dương đưa mass về 0 trong ví dụ tổng hợp |
| Timestamp rebinning | Tổng mass được bảo toàn khi chuyển các interval chồng lấn về bins phân hoạch timeline |
| Masked scoring | Thêm padding ngẫu nhiên đã mask không đổi score trong check số học |
| Centered residual | Score ban đầu bằng S0; finite-difference derivative theo một phần tử interaction khác 0 với gamma dương |
| Bound | Kiểm tra E và centered correction nằm trong giới hạn trên các ma trận tổng hợp; chứng minh tổng quát nằm ở §5.5 |
| Confidence normalization | Self-normalization không đổi loss khi mọi g cùng giảm; mẫu số theo số contrast giữ tỷ lệ giảm |
| Tie handling | Ví dụ 2 queries sinh 3 matched-rank positions trong công thức cũ; stable-ID evaluation cho kết quả khác |

Các checks này không thực thi trained encoder, không kiểm định ngữ nghĩa sign và không xác minh latency/GPU memory. Các kiểm tra gradient là finite differences trên scoring formula, không phải end-to-end autograd của một DIVE implementation.

Đã kiểm tra trực quan trang Table 6 của SPOT-ALIGN từ PDF truy cập được. File Markdown đã qua kiểm tra cấu trúc 13 bảng, cú pháp công thức bằng KaTeX và cấu trúc URL; không có lỗi trong các kiểm tra này. Chưa kiểm tra trực quan bản Markdown bằng trình hiển thị của ứng dụng vì môi trường không có browser executable. Kiểm tra URL không đồng nghĩa xác nhận mọi website luôn truy cập được.

## 10. Lộ trình thực hiện và điều kiện viết paper

**Ưu tiên 1:** tái lập SEDS với dev thật, khóa preprocessing/text artifacts và đo $U_\gamma$. **Ưu tiên 2:** chạy local warm-up, audit khả năng tạo real contrasts và support. **Ưu tiên 3:** pilot các control A1–A6. Chỉ chạy full experiment và transfer khi local supervision đã thắng same-pool full-score control.

Tiêu đề làm việc có thể giữ: *Learning Discriminative Visual Evidence from Real Paired Contrasts for Sign Language Retrieval*. Abstract chỉ được viết kết quả sau khi có số thực nghiệm. Các đóng góp có thể bảo vệ được phải là: một failure mode đo được; supervision xử lý nó tốt hơn các control phù hợp; gain retrieval chuẩn với tài nguyên minh bạch.

Nếu không đạt các gate, kết luận nghiên cứu vẫn có giá trị: có thể cho thấy giới hạn của pseudo-local supervision hoặc đánh đổi giữa fine discrimination và corpus retrieval. Không nên tiếp tục thêm module chỉ để tìm một cấu hình thắng nhỏ.

**Đề xuất hành động:** triển khai v2 như một giả thuyết có control, đồng thời giữ bản v1 làm comparator sau khi pipeline nền đã đúng. Quyết định chọn phương pháp cuối cùng phải dựa trên kết quả A3/A4/A6, opportunity bound và chi phí; hiện không có bằng chứng cho một bảo đảm “tốt nhất” hoặc được nhận ở A*.

## 11. Danh mục nguồn và giới hạn truy cập

Các hyperlink trong nội dung trỏ tới claim tương ứng. Ngày truy cập: 07/09/2026. Bảng này ghi phiên bản để có thể kiểm lại, không thay thế một systematic-review protocol tuyệt đối bao phủ mọi publication.

| Nguồn | Tác giả, venue/năm | Phiên bản/mức truy cập |
|---|---|---|
| [Sign Language Video Retrieval with Free-Form Textual Queries — SPOT-ALIGN](https://arxiv.org/abs/2201.02495) | Duarte và cộng sự, CVPR 2022 | PDF arXiv, method/experiments/appendix liên quan; Table 6 đã nhìn trực quan |
| [CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning](https://arxiv.org/abs/2303.12793) | Cheng và cộng sự, CVPR 2023 | Paper/supplement liên quan và pinned core code |
| [Uncertainty-aware Sign Language Video Retrieval with Probability Distribution Modeling — UPRet](https://arxiv.org/abs/2405.19689) | Wu và cộng sự, ECCV 2024 | HTML v1 và pinned released implementation |
| [SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval](https://arxiv.org/abs/2407.16394) | Jiang và cộng sự, ACM MM 2024 | HTML v1 và pinned core code/loaders |
| [C²RL: Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieval](https://arxiv.org/abs/2408.09949) | Chen và cộng sự, arXiv 2024/TCSVT 2025 | Nội dung/số liệu từ v1; chưa audit official implementation |
| [Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative Mining for Sign Language Retrieval — SAN](https://aclanthology.org/2026.acl-long.1302/) | Lee và cộng sự, ACL 2026 | Paper/appendix liên quan, arXiv v1 và pinned released code |
| [EqSim](https://arxiv.org/abs/2303.14465) | Wang và cộng sự, ICCV 2023 | HTML v2, cơ chế và equations; chưa audit code |
| [DMAE/TPM-CL](https://arxiv.org/abs/2309.11082) | Jiang và cộng sự, ACM MM 2023 | HTML v3, token selection và ranking; chưa audit code |
| [A³PRVR](https://ojs.aaai.org/index.php/AAAI/article/view/37271) | Chen và cộng sự, AAAI 2026 | Official PDF và method liên quan; chưa audit code |
| [TripletCLIP](https://arxiv.org/abs/2411.02545) | Patel và cộng sự, NeurIPS 2024 | Abstract/phạm vi đối chiếu; không review toàn bộ implementation |
| [SignCLIP](https://aclanthology.org/2024.emnlp-main.518/) | Jiang và cộng sự, EMNLP 2024 | Publication, abstract và phần scope liên quan |
| [GTRN](https://ro.ecu.edu.au/ecuworks2022-2026/6028/) | Hu và cộng sự, Neurocomputing 2025 | Mô tả từ institutional repository; toàn văn chưa truy cập được |
| [VTaMo](https://arxiv.org/abs/2607.09126) | Hu và cộng sự, preprint 07/2026 | HTML v1, null OT và formulation liên quan |
| [DualAnchor](https://arxiv.org/abs/2607.27614) | Zhang và cộng sự, preprint 07/2026 | HTML v1, partial OT liên quan |
| [SignSeek](https://arxiv.org/abs/2609.03695) | Asasi và cộng sự, preprint 09/2026 | HTML v1, dictionary retrieval/saliency |
| [CBA](https://www.preprints.org/manuscript/202605.1948) | Meng và cộng sự, preprint 05/2026 | Bản v1; prior art về counterfactual alignment |
| [Causality-inspired SLRet](https://doi.org/10.1016/j.cviu.2025.104631) | Yang, Wei, Li, Hu; CVIU 2026, 264:104631 | Metadata/abstract xác minh; thiếu toàn văn và kết quả |
| [HNMA](https://doi.org/10.1016/j.cviu.2026.104918) | Yang và cộng sự, CVIU, online 08/2026 | Metadata/abstract; overlap toàn văn còn phải kiểm tra |

Tài liệu không có fabricated experiment table, không gán prediction thành measured gain, và không kế thừa nguyên trạng các tuyên bố “đã kiểm chứng” trong file đầu vào. Những sửa phương pháp ở §5 là đề xuất mới của lượt review này, cần được kiểm nghiệm như mọi proposal nghiên cứu khác.
