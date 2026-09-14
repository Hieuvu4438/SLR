# Sign Language Retrieval: nghiên cứu, phản biện và đặc tả phương pháp

Hướng đáng đầu tư thử nghiệm là **Paired Local Evidence Learning (PLEL)**: bổ sung giám sát phân biệt từ các cặp video–caption thật cho một nhánh chỉ đọc đặc trưng clip cục bộ, đồng thời giữ một retriever nền mạnh làm điểm tham chiếu. Đóng góp dự kiến nằm ở cách tạo và sử dụng giám sát bằng chứng thị giác; bản thân residual scoring, contrastive loss, attention hay khai thác hard negative đều đã có tiền lệ.

Đây là **proposal có điều kiện kiểm chứng**, chưa phải phương pháp đã đạt SOTA. Chưa có thí nghiệm huấn luyện retrieval của PLEL; các số liệu mới trong báo cáo là kiểm tra metadata và kiểm tra đại số trên dữ liệu mô phỏng. Bằng chứng hiện tại đủ để thiết kế một thử nghiệm có khả năng bác bỏ giả thuyết, chưa đủ để cam kết vượt tất cả phương pháp hiện có hoặc được nhận tại hội nghị hàng đầu.

## 1. Phạm vi và quyết định nghiên cứu

Đối tượng chính là sentence-level sign–text retrieval: truy hồi video ký hiệu liên tục từ truy vấn văn bản tự do (T2V), và truy hồi văn bản từ video (V2T). Dùng các bộ dữ liệu hiện hữu, ưu tiên How2Sign, sau đó PHOENIX-2014T và CSL-Daily. Ngân sách thiết kế là 1–2 GPU 24 GB, tận dụng đặc trưng offline và mã đánh giá sẵn có.

Các quyết định triển khai:

1. Dùng CiCo làm nền đối chứng đầu tiên vì có mã, metadata và đường tải tài nguyên được công bố ngoài Baidu.
2. Giữ SEDS trong literature review và bảng mục tiêu, nhưng **không dùng checkpoint, RGB feature, pose feature hay pretrained SignBERT do SEDS phân phối**. Pipeline PLEL không có phụ thuộc SEDS.
3. UPRet là đối chứng và nền mở rộng tiếp theo khi tái lập được; không mặc định rằng repository của nó đã cung cấp checkpoint chạy ngay.
4. Không xây benchmark hoặc dataset mới. Các cặp lấy từ training split và các phân tích lỗi trên split hiện hữu là thành phần huấn luyện/ablation.
5. Không coi sửa metric, thay ngôn ngữ query, thêm dữ liệu pretraining hay đổi backbone là đóng góp của PLEL.
6. Không dùng gloss thủ công để huấn luyện PLEL. Đặc trưng CiCo vẫn thừa hưởng sign pretraining bên ngoài; phải khai báo điều kiện này, không gọi toàn bộ hệ thống là không có mọi hình thức giám sát ký hiệu.

Phạm vi cập nhật tài liệu đến ngày 09/09/2026. Bảng kết quả bên dưới ghi rõ phiên bản paper được kiểm chứng. Với CMCM, chỉ có phần mô tả công khai và module nguồn được kiểm tra; chưa kiểm chứng được bảng kết quả trong toàn văn. Vì vậy, bảng này chưa đủ để tuyên bố một ngưỡng SOTA toàn lĩnh vực tuyệt đối.

## 2. Các phương pháp hiện tại giải quyết điều gì?

### 2.1. SPOT-ALIGN: chất lượng sign encoder là yếu tố nền

SPOT-ALIGN xây dựng sign representation qua các vòng sign spotting và huấn luyện lại, rồi học không gian embedding video–text. Hệ thống kết hợp tín hiệu retrieval qua nhận dạng dấu với matching xuyên modality. Điều có thể kế thừa là pipeline chuẩn bị How2Sign và bài học về chất lượng visual feature. Không nên mô tả công trình này như một baseline chỉ có mean pooling: phần học sign encoder và nguồn lexical supervision là thành phần quan trọng. [Duarte et al., CVPR 2022, §3](https://arxiv.org/abs/2201.02495).

**Phản biện liên quan đến proposal:** nếu local feature đã bỏ mất chi tiết phân biệt, thay loss ở phía retrieval không thể khôi phục thông tin đó. PLEL phải có phép thử trần biểu diễn trước khi mở rộng tuning.

### 2.2. CiCo: alignment cục bộ đã tồn tại

CiCo kết hợp I3D domain-agnostic và domain-aware, dùng Transformer cho chuỗi clip và văn bản, rồi tính ma trận clip–token. CLCL tổng hợp ma trận này theo hai hướng bằng softmax trước khi contrast video–text. Vì thế, phát biểu “retrieval trước đây chỉ có global alignment” là sai. Trên PHOENIX và CSL-Daily, nhánh văn bản sử dụng bản dịch tiếng Anh để khai thác CLIP. [Cheng et al., CVPR 2023, §3–4](https://arxiv.org/abs/2303.12793).

**Khoảng trống hẹp cần thử:** một token video sau Transformer có thể chứa ngữ cảnh toàn câu. Similarity cao giữa token đó với một từ chưa chứng minh cửa sổ hình ảnh tương ứng chứa đủ bằng chứng cho từ ấy. Đây là hệ quả của kiến trúc, không phải kết luận rằng mọi alignment của CiCo đều sai.

### 2.3. UPRet: uncertainty và optimal transport đã có

UPRet mô hình hóa biểu diễn bằng Gaussian, lấy mẫu và dùng optimal transport để tính matching. Trong bản paper kiểm tra, các mẫu được pooling theo chiều chuỗi trước khi OT giữa các tập mẫu; không nên đồng nhất mọi ma trận OT của UPRet với nhãn timestamp–word. [Wu et al., ECCV 2024, §3.3–3.5](https://arxiv.org/abs/2405.19689).

**Phản biện:** biểu diễn xác suất không tự chứng minh uncertainty đã được calibration; nhiều caption/video đúng trong metadata cũng không tự được xử lý chỉ bằng Gaussian. Thêm OT, Gaussian hoặc “many-to-many alignment” vào CiCo không đủ làm claim mới. Một đối chứng quan trọng là UPRet với cùng feature, cùng tập ứng viên và cùng ngân sách chọn checkpoint.

### 2.4. SEDS: thêm pose và fusion chưa phải khoảng trống mới

SEDS kết hợp RGB offline với pose online, dùng Cross Gloss Attention Fusion và mục tiêu matching RGB–pose ở mức clip, bên cạnh các loss video–text. Kết quả nhấn mạnh giá trị của thông tin chuyển động cục bộ. [Jiang et al., ACM MM 2024, §3](https://arxiv.org/abs/2407.16394).

**Phản biện:** PLEL không thể nhận novelty chỉ từ việc “tập trung vào local motion”. Điểm phải phân biệt là giám sát phần bằng chứng làm hai cặp thật khác nhau, với đường truyền gradient không đọc toàn bộ ngữ cảnh video. SEDS là đối chứng kiến trúc quan trọng, dù tài nguyên phân phối của nó bị loại khỏi kế hoạch.

### 2.5. C²RL: representation learning mạnh hơn có thể quan trọng hơn head mới

C²RL kết hợp ICL dựa trên CLCL với ECL dựa trên language modeling. Giai đoạn retrieval sử dụng feature offline và hai mBART encoder riêng cho video/text. Bản arXiv v1 mô tả pretraining 200 epoch trên 8 RTX 3090 và retrieval 80 epoch trên 8 RTX 3090. [Chen et al., §III–IV](https://arxiv.org/abs/2408.09949); [bản xuất bản TCSVT 2025](https://ieeexplore.ieee.org/document/10933970/).

**Phản biện:** khoảng cách lớn với CiCo trên CSL-Daily khiến giả định “một loss nhỏ chắc chắn vượt mọi SOTA” thiếu căn cứ. C²RL là mục tiêu cần theo dõi, nhưng phép so sánh nhân quả phải kiểm soát representation, encoder văn bản, ngôn ngữ và pretraining.

### 2.6. SAN: vấn đề negative không còn mới; trade-off vẫn cần giải quyết

SAN khai thác sign–word match tin cậy, tìm các sign feature dễ nhầm và tạo negative caption bằng thay từ. Paper được nhận tại ACL 2026. Với CiCo, SAN nâng fine-grained V2T R@1 từ 17,9 lên 39,4; trên gallery gốc, T2V/V2T R@1 đổi từ 69,2/70,1 thành 68,1/67,8. Paper còn phân tích tác động của số negative và trọng số loss. [Lee et al., Table 1–2](https://arxiv.org/html/2607.09263v1).

**Phản biện:** đây là bằng chứng về trade-off trong thiết lập đã báo cáo, không chứng minh nguyên nhân duy nhất là gradient conflict. Stress test caption thay từ cũng chưa bảo đảm cải thiện retrieval trên gallery thật. PLEL cần tách giá trị của **cách chọn negative** khỏi giá trị của **giám sát local evidence**.

### 2.7. CMCM: phải xem đúng nhánh code

CMCM đề xuất retrieval nhiều mức với diễn giải causality. Repository có `main` chứa README và `master` chứa các module. Nhánh `master` tại commit `5d458719d1da2f082e188cc44705003d919e7e97` có CSA, CCG, TMCP và encoder; chưa thấy training entrypoint, evaluation entrypoint hoặc cấu hình tái lập hoàn chỉnh trong cây nguồn được kiểm tra. [Repository, nhánh master](https://github.com/vddong-zjut/CMCM/tree/5d458719d1da2f082e188cc44705003d919e7e97); [publisher](https://www.sciencedirect.com/science/article/abs/pii/S1077314225003546).

CSA tính điều chỉnh giữa feature gốc và augmented feature; CCG có Gaussian alignment; TMCP có temporal attention và covariance pooling. Đây là bằng chứng chống lại claim chung chung “lần đầu giảm confounder bằng residual” hoặc “lần đầu dùng nhiều mức thời gian”. Các module rời chưa đủ để xác nhận toàn bộ thuật toán trong paper hay kết quả định lượng.

### 2.8. Những công trình bổ sung có ảnh hưởng đến quyết định

| Công trình | Điều đã kiểm chứng | Hệ quả đối với proposal |
|---|---|---|
| SLP / Scaling up Multimodal Pre-training | Pose pretraining kết hợp sign–text contrastive và masked pose modeling; có retrieval PHOENIX và CSL-Daily | Cần có trong bảng bối cảnh; dữ liệu pretraining lớn là điều kiện riêng |
| SignCLIP, EMNLP 2024 | Contrastive pretraining từ từ điển đa ngôn ngữ và đánh giá representation | Không nhận claim đầu tiên về sign–text contrastive hoặc multilingual transfer |
| SignSeek, 2026 | Tập trung sign dictionary retrieval bằng video, phân biệt với sentence-level SLRet | Nguồn tham khảo representation; không trộn R@1 khác nhiệm vụ vào bảng chính |
| GTRN, Neurocomputing 2025 | Dùng visual signing query để lấy video trong corpus | Nhiệm vụ truy vấn thị giác; không thay thế đối chứng free-form text retrieval |

Nguồn: [SLP, Table X](https://arxiv.org/html/2408.08544v1#S4.T10), [SignCLIP](https://aclanthology.org/2024.emnlp-main.518/), [SignSeek](https://arxiv.org/html/2609.03695v1), [GTRN, hồ sơ tác giả](https://ro.ecu.edu.au/ecuworks2022-2026/6028/).

## 3. Bảng kết quả đã kiểm chứng và ngưỡng mục tiêu

Các ô là **T2V R@1 / V2T R@1, đơn vị %**. Đây là số do paper báo cáo, không phải kết quả tái lập trong nghiên cứu này. `—` nghĩa là không đưa số chưa kiểm chứng vào bảng.

| Phương pháp | How2Sign | PHOENIX-2014T | CSL-Daily | Nguồn số liệu |
|---|---:|---:|---:|---|
| SPOT-ALIGN, COMB | 34,2 / 23,6 | 55,8 / 53,1 | — | [Paper 2022](https://arxiv.org/abs/2201.02495) |
| CiCo | 56,6 / 51,6 | 69,5 / 70,2 | 75,3 / 74,7 | [Paper 2023](https://arxiv.org/abs/2303.12793) |
| UPRet | 59,1 / 53,4 | 72,0 / 72,0 | 78,4 / 77,0 | [Tables 1–3](https://arxiv.org/abs/2405.19689) |
| SEDS | 62,5 / 57,9 | 76,8 / 78,7 | 85,8 / 85,4 | [Tables 1–3](https://arxiv.org/abs/2407.16394) |
| C²RL | 62,4 / 57,5 | 78,7 / 77,6 | 90,3 / 88,4 | [arXiv v1, Table VI](https://arxiv.org/abs/2408.09949v1) |
| SLP, pose pretraining | — | 74,5 / 75,1 | 87,5 / 87,2 | [arXiv v1, Table X](https://arxiv.org/html/2408.08544v1) |
| SAN trên CiCo | — | 68,1 / 67,8 | — | [Gallery gốc, Table 1](https://arxiv.org/html/2607.09263v1) |
| CMCM | — | — | — | Chưa kiểm chứng bảng toàn văn |
| PLEL | Chưa chạy | Chưa chạy | Chưa chạy | Proposal |

Không diễn giải chênh lệch 0,1 điểm giữa SEDS và C²RL trên How2Sign như bằng chứng thống kê về ưu thế. Các điều kiện representation, modality và text encoder khác nhau. SEDS còn báo 31.019/1.738/2.348 cặp How2Sign, trong khi CiCo/C²RL dùng 31.085/1.739/2.348; phải ghi nhận khác biệt train/dev dù test count trùng nhau. [SEDS, §4.1](https://arxiv.org/abs/2407.16394).

Mốc tham khảo cao nhất trong các số đã xác minh ở bảng là 62,5/57,9 trên How2Sign, 78,7/78,7 trên PHOENIX và 90,3/88,4 trên CSL-Daily. **Không gọi đây là SOTA toàn lĩnh vực đã chốt**, vì còn khoảng trống toàn văn CMCM và điều kiện so sánh chưa đồng nhất.

Mục tiêu nghiên cứu phải phân biệt ba mức:

| Mức | Bằng chứng cần có | Claim được phép |
|---|---|---|
| Cải thiện phương pháp | Thắng nền cùng feature, encoder, split, metric, ngân sách chọn model | PLEL cải thiện retrieval trong thiết lập kiểm soát |
| Vượt số paper | Vượt số công bố, ghi rõ mọi khác biệt tài nguyên/protocol | Vượt số báo cáo ở điều kiện đã khai báo |
| Claim SOTA mạnh | Hoàn tất đối chiếu literature, protocol và đối chứng mạnh; có độ bất định | SOTA trong phạm vi nhiệm vụ và thiết lập được định nghĩa |

## 4. Audit dữ liệu: những điều đã đo được

### 4.1. Đơn vị dữ liệu không luôn là một video–một caption

Các số sau được tính trực tiếp từ metadata CiCo tại snapshot nguồn đã kiểm tra. `Nhóm` là key của manifest; không đồng nghĩa mọi nhóm có nội dung text khác nhau. Chuẩn hóa trong phép đếm lexical dùng casefold và tokenizer regex `\w+|[^\w\s]`, giữ dấu câu như token.

| Manifest | Nhóm | Video | Chuỗi English khác nhau sau chuẩn hóa |
|---|---:|---:|---:|
| How2Sign train | 30.852 | 31.085 | 30.034 |
| How2Sign test | 1.969 | 2.348 | 1.930 |
| CSL-Daily train | 6.598 | 18.401 | 6.572 |
| CSL-Daily test | 798 | 1.176 | 798 |
| PHOENIX train | 7.096 | 7.096 | 6.825 |
| PHOENIX test | 642 | 642 | 630 |

Nguồn dữ liệu: [How2Sign manifests](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/data_h2), [CSL manifests](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/data_csl), [PHOENIX manifests](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/data_ph). Đây là kết quả audit độc lập trên các file đó, không phải bảng từ paper.

Hệ quả triển khai: tạo manifest phẳng theo video nhưng phải giữ `group_id`, `video_id`, `text_original`, `text_model` và `text_equivalence_id` riêng biệt. Không gộp các nhóm evaluation chỉ vì chuỗi text trùng nhau. Trong huấn luyện, tránh coi video cùng nhóm là negative; áp dụng cùng chính sách đó cho baseline và phương pháp đề xuất.

### 4.2. PHOENIX dev.pkl cần được kiểm tra trước khi dùng

File `data_ph/dev.pkl` chứa 7.615 ID: giao với train là toàn bộ 7.096 ID, còn 519 ID ngoài train. Giao train–test và dev–test theo video ID bằng 0 trong các manifest được audit.

**Không dùng nguyên file này làm validation.** Lấy official dev IDs, đối chiếu phần 519 ID còn lại, kiểm tra caption/feature rồi xuất một manifest dev sạch riêng. Phép trừ ID chỉ là bước xác định ứng viên; chưa thay thế xác nhận với split chính thức. Quan sát này không tự chứng minh kết quả của paper đã bị leakage.

### 4.3. Độ phủ cặp khác một từ rất thấp

Đếm các chuỗi huấn luyện có ít nhất một chuỗi cùng độ dài, khác đúng một lexical token tại cùng vị trí:

| Bộ dữ liệu | Chuỗi train khác nhau | Anchor có cặp | Tỷ lệ |
|---|---:|---:|---:|
| PHOENIX | 6.825 | 379 | 5,55% |
| CSL-Daily | 6.572 | 60 | 0,91% |
| How2Sign | 30.034 | 285 | 0,95% |

Đây là kiểm tra lexical trên bản English, không phải độ phủ các cặp dấu khác một nét. Vì vậy, thiết kế chỉ học từ cặp câu tự nhiên khác đúng một từ không phù hợp làm cơ chế huấn luyện duy nhất.

### 4.4. Nới sang cụm từ giúp tăng ứng viên nhưng tăng rủi ro nhiễu

Phép audit mở rộng dùng TF-IDF unigram/bigram trên **train**, `min_df=2`, `max_df=0.3`, `sublinear_tf=True`, dtype FP32; lấy top 21 phần tử có score trong sparse row rồi bỏ chính anchor. Thông thường còn 20 neighbor; ties có thể làm anchor không nằm trong top 21, nên mô tả chính xác phép audit là “top 21 rồi loại self”. Dùng `SequenceMatcher(autojunk=False)` để tìm token chung theo thứ tự. Yêu cầu mỗi câu dài 5–64 lexical token, mỗi phía có 1–6 token không khớp, và số token chung chia độ dài câu ngắn hơn ít nhất 0,65.

Lọc tiếp bằng danh sách English stopword của scikit-learn, giữ lại `no/not/never/without/none`; mỗi phía phải có từ nội dung và hai dãy từ nội dung khác nhau. Bộ lọc này chỉ loại một phần nhiễu cú pháp, không chứng minh hai cụm đối lập về nghĩa.

| Điều kiện: tỷ lệ anchor tự tìm được ít nhất một neighbor | PHOENIX | CSL-Daily | How2Sign |
|---|---:|---:|---:|
| Cặp rộng, overlap ≥ 0,65 | 26,05% | 19,95% | 16,18% |
| Có khác biệt ở từ nội dung | 24,76% | 18,75% | 13,79% |
| Chỉ một vùng replace liên tục, có khác biệt nội dung | 9,51% | 6,98% | 5,73% |
| Điều kiện trên, khác source video theo prefix ID | Không đo | Không đo | 5,33% |

Số cặp vô hướng khác nhau ở điều kiện một vùng replace lần lượt là 3.704, 499 và 2.662. Tỷ lệ anchor là phép đếm có hướng theo danh sách top neighbor riêng; không được dùng nó như tỷ lệ hợp của tất cả endpoint trong các cặp vô hướng.

Kiểm tra mẫu văn bản cho thấy có cặp đổi tên người, có cặp chỉ khác cách diễn đạt, có khác biệt liên quan ngày tháng; một số cặp có chung khuôn câu nhưng không tạo đối chứng ngữ nghĩa rõ. Vì vậy, tất cả tỷ lệ trên là **độ phủ ứng viên trước kiểm tra video**, không phải tỷ lệ pseudo-label đúng. “Khác source prefix” cũng không đồng nghĩa “khác signer”.

### 4.5. Những đại lượng còn thiếu phải đo bằng feature/video

Chưa có số đo về tỷ lệ candidate vượt kiểm tra local support, độ chính xác sign–word alignment, số false negative ngữ nghĩa, độ phủ signer hay mức cải thiện retrieval. Không suy ra chúng từ lexical overlap.

Cần báo cáo một funnel: số anchor ban đầu → cặp qua lexical filter → cặp qua trùng nghĩa/caption filter → cặp có local support ổn định → phân bố word/signing context của cặp được dùng. Nếu chỉ còn vài cụm ngày tháng hoặc tên riêng, giả thuyết không có đủ cơ sở để triển khai rộng.

## 5. Audit mã nguồn và protocol đánh giá

### 5.1. Các điểm code phải giữ chính xác

| Thành phần CiCo | File/hàm nguồn | Yêu cầu tích hợp |
|---|---|---|
| Feature input và chuẩn bị video | `I3D_feature_extractor/get_features.py`, dataloader từng dataset | Giữ chiều channel, thứ tự domain-aware/domain-agnostic và phép trộn alpha |
| Embedding | `CLCL/modules/modeling.py`: `get_sequence_output`, `get_visual_output` | Dùng wrapper lấy đúng hidden state; không đoán tên tensor |
| Scoring CLCL | `flip_similarity_softmax` | Hai phép tổng hợp theo token/clip khác nhau; giữ đủ hai score |
| Text sampling | `_get_text` trong dataloader | Câu dài có thể lấy BPE token cách đều; không mặc định truncation ở đuôi |
| Multi-video group | `eval_epoch` và `cut_off_points` | Giữ thứ tự caption group và video member |
| Metric | `compute_metrics`, `tensor_text_to_video_metrics`, `tensor_video_to_text_sim` | Tên hàm không quyết định hướng nhiệm vụ; kiểm tra shape và cách gọi |

Nguồn: [modeling.py](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/modeling.py), [dataloader How2Sign](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/dataloader_H2_retrieval.py), [evaluation entrypoint](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/main_task_retrieval.py), [metrics.py](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/metrics.py).

`compute_metrics` tìm mọi phần tử bằng điểm diagonal trong ma trận đã sort; ties cần được kiểm tra vì text trùng có thể tạo đồng điểm. Không âm thầm sửa tie policy rồi so số mới với số paper. Nếu có bản sửa metric, chạy cả baseline và PLEL bằng cả evaluator gốc và evaluator đã sửa, ghi riêng kết quả.

### 5.2. Không lựa chọn checkpoint bằng test

Trong `main_task_retrieval.py` của CiCo được kiểm tra, vòng train gọi `eval_epoch(..., test_dataloader, ...)` rồi cập nhật `best_score`; phần gọi validation đang comment. Entry point UPRet được đọc có cấu trúc tương tự. Đây là hành vi của mã công khai, chưa đủ để khẳng định protocol thực tế tác giả dùng cho mọi bảng kết quả. [CiCo training loop](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/main_task_retrieval.py), [UPRet source](https://github.com/xua222/UPRet/blob/046366227417e1d8ec14145965403462df345984/main_task_retrieval_dis.py).

Trong các thí nghiệm mới: chọn checkpoint, alpha, trọng số residual và trọng số loss trên dev sạch. Không nạp test manifest vào trainer. Test chỉ dùng khi cấu hình được khóa; mọi baseline mới cũng chịu cùng quy tắc.

### 5.3. Phân biệt official grouped evaluation và metric bổ sung

Biểu diễn score chuẩn trong code mới là `S[video_id, text_group_id]`. Với T2V theo cách group của CiCo, lấy max score qua video member của từng nhóm ứng viên rồi xếp hạng nhóm. Với V2T, mỗi video xếp hạng các text group.

Truy hồi top-K video instance với nhiều positive là một phép chấm khác. Khi không có ties, R@1 có thể trùng với cách group; R@5/R@10 không nhất thiết trùng vì các video trong cùng một nhóm negative chiếm nhiều vị trí instance nhưng chỉ một vị trí group. Không dùng cải thiện do đổi phép chấm để chứng minh PLEL.

### 5.4. Rủi ro tái lập của repository bổ sung

| Repo | Quan sát cụ thể | Cách sử dụng hợp lý |
|---|---|---|
| UPRet | Có các entrypoint thường và `_dis`; script test chứa đường dẫn checkpoint máy tác giả | Trace script → model import → score path; không mặc định file có tên `_dis` là cấu hình công bố |
| SAN | README “Coming Soon”; có code loss/model, nhưng chưa xác nhận pipeline mining và các file đầu vào đầy đủ | Dùng paper + code đã có để làm đối chứng SAN tái lập; khai báo phần tự dựng |
| C²RL | `sltbaselines` là mã tái triển khai SLT thống nhất, không phải gói retrieval C²RL gốc | Tham khảo module/pretraining; không gọi đó là checkpoint/evaluator C²RL chính thức |
| CMCM master | Có module, thiếu pipeline hoàn chỉnh; `DEVICE` không được định nghĩa trong file CSA, masking trong CCG cần kiểm tra shape | Không lấy trực tiếp làm nền chạy; không suy diễn kết quả tác giả từ lỗi module rời |

Nguồn: [UPRet](https://github.com/xua222/UPRet), [SAN](https://github.com/joonmy/SAN/tree/82aba9cbc1beb403abef6e9a3875ca52479805c8), [sltbaselines](https://github.com/ozgemercanoglu/sltbaselines), [CMCM CSA](https://github.com/vddong-zjut/CMCM/blob/5d458719d1da2f082e188cc44705003d919e7e97/modules/CSA_Module.py), [CMCM CCG](https://github.com/vddong-zjut/CMCM/blob/5d458719d1da2f082e188cc44705003d919e7e97/modules/CCG_Module.py).

## 6. Kiểm tra tính mới với nghiên cứu ngoài SLRet

| Prior art | Ý tưởng đã có | Phần PLEL còn phải chứng minh |
|---|---|---|
| SSAMT, AAAI 2022 | Tạo supervision mức phrase để phạt semantic mismatch trong image–text retrieval | Không claim lần đầu có phrase-level mismatch loss |
| FDCA, ICLR 2025 | Tách thông tin retained/injected/excluded ở sentence và token trong composed video retrieval | Khác biệt nằm ở paired supervision trên SLRet và đường local video, không phải thao tác chia shared/different token |
| GARE, arXiv v5 năm 2025 | Hiệu chỉnh embedding theo cặp bằng increment, có regularization và giữ increment ở inference | Không claim residual hoặc giảm căng thẳng contrastive là mới |
| TeachCLIP, CVPR 2024 | Distillation cho retrieval với tín hiệu ở nhiều mức | Teacher–student hoặc giữ nền bằng distillation chỉ là công cụ |
| SAN | Mining negative theo visual confusability của sign | Phải thắng SAN khi có cùng nền và ngân sách, kể cả đối chứng có cùng cặp negative |
| CiCo/SEDS/UPRet | Fine-grained matching, modality fusion, probabilistic alignment | Phải chứng minh giá trị riêng của supervision và local evidence restriction |

Nguồn: [SSAMT](https://arxiv.org/abs/2109.05523), [FDCA, paper ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/file/2dae7d1ccf1edf76f8ce7c282bdf4730-Paper-Conference.pdf), [GARE](https://arxiv.org/html/2505.12499v5), [TeachCLIP](https://openaccess.thecvf.com/content/CVPR2024/html/Tian_Holistic_Features_are_almost_Sufficient_for_Text-to-Video_Retrieval_CVPR_2024_paper.html).

**Đánh giá novelty hiện tại: có hướng phân biệt, nhưng rủi ro vẫn đáng kể.** Ghép local head, real hard pairs và frozen baseline chưa tự tạo một đóng góp mạnh. Claim chỉ có cơ sở nếu phép kiểm soát receptive field và cách kiểm tra bằng chứng của hai phía tạo cải thiện ổn định mà các đối chứng đơn giản không đạt được.

## 7. Research gap được giữ lại

Giả thuyết trung tâm là: **giám sát ranking ở toàn câu có thể chưa buộc mô hình đặt khả năng phân biệt vào đúng bằng chứng cục bộ; việc học trực tiếp từ hai cặp thật trên đường local, rồi kết hợp có giới hạn với retrieval nền, có thể cải thiện truy hồi trên gallery gốc mà tránh một phần trade-off của negative caption tổng hợp.**

Phân tách mức độ bằng chứng:

| Nhận định | Trạng thái |
|---|---|
| Các kiến trúc hiện tại đã có matching ở mức token | Có bằng chứng paper/code |
| SAN có trade-off fine/coarse trong bảng CiCo được báo cáo | Có bằng chứng định lượng |
| Cặp tự nhiên khác đúng một từ quá ít để làm nguồn giám sát duy nhất | Có audit metadata |
| Local token sau global Transformer có thể chứa thông tin ngoài cửa sổ nguồn | Hệ quả cấu trúc mạng |
| Contextual shortcut là nguyên nhân chủ yếu của lỗi retrieval thực tế | Chưa được chứng minh |
| PLEL lọc được pseudo-label đủ chính xác và đủ đa dạng | Chưa được chứng minh |
| PLEL vượt SOTA | Chưa có thí nghiệm |

Không lựa chọn các hướng sau làm đóng góp trung tâm: chỉ thêm OT; chỉ thêm RGB–pose; chỉ đổi sang hard negative theo sign; chỉ thêm residual/reranking; bắt buộc monotonic alignment giữa spoken words và signing order; hoặc sửa multi-positive metric. Các hướng đó hoặc đã có prior art, hoặc có rủi ro ngôn ngữ/protocol mà lợi ích chưa được chứng minh.

## 8. PLEL: đặc tả phương pháp tối thiểu

### 8.1. Phạm vi của đóng góp

PLEL gồm một nhánh visual nhỏ và một mục tiêu phụ trên cặp dữ liệu thật. Backbone, text encoder và score nền được đóng băng trong bản đầu tiên. Không thêm pose extractor, decoder ngôn ngữ, optimal transport hoặc mô hình sinh caption.

Một cặp huấn luyện phụ gồm hai mẫu thật: `(V_i, Q_i)` và `(V_j, Q_j)`. Hai câu có ngữ cảnh lexical chung và phần khác biệt `U_i`, `U_j`. Ta không tự sinh câu rồi mặc định video gốc là negative của câu đó. Tuy vậy, hai mẫu thật cũng **không bảo đảm** các ghép chéo `(V_i, U_j)` và `(V_j, U_i)` là sai; đây là rủi ro pseudo-label cần đo, không phải vấn đề đã được giải quyết bằng cách đổi nguồn negative.

Claim dự kiến, nếu thực nghiệm ủng hộ, là: **paired supervision có kiểm soát phạm vi thông tin giúp học khả năng phân biệt cục bộ hữu ích cho truy hồi toàn câu**. Không dùng các từ “causal identification”, “ground-truth alignment” hoặc “sign boundary” cho những gì chỉ được suy ra từ similarity.

### 8.2. Score nền và đơn vị tính

Đặt `A(V,Q)` và `C(V,Q)` là hai score CLCL trước nhân `exp(logit_scale)`. Trong đường evaluation CiCo được kiểm tra, `_run_on_single_gpu_new_mix` tạo cùng một phép trộn cho cả hai hướng:

\[
B(V,Q)=\eta A(V,Q)+(1-\eta)C(V,Q).
\]

Giữ `eta=dual_mix` của cấu hình đối chứng, và giữ nguyên phép trộn feature `alpha` — đây là hai tham số khác nhau. Nếu wrapper lấy logits đã scale, phải chia đúng hệ số `exp(logit_scale)` trước khi thêm local score. Không chuẩn hóa score theo từng query hay từng gallery vì sẽ làm thay đổi bài toán và giới hạn bên dưới. [CiCo evaluation mixing](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/main_task_retrieval.py).

Mọi encoder nền chạy `eval()` và không nhận gradient. Với checkpoint, manifest, thứ tự token và sampling cố định, `B` phải lặp lại được. UPRet cần adapter riêng nếu bổ sung sau; không tái sử dụng mù phép bỏ scale của CiCo cho score OT.

### 8.3. Đường visual cục bộ

Lấy `x_t` từ I3D clip feature **trước** Transformer thời gian của CiCo. Dùng đúng stream và phép trộn feature của đối chứng. Thiết kế đầu tiên áp dụng cho feature trộn có chiều 1.024; chế độ concatenate cần adapter riêng và không được âm thầm đổi số channel.

\[
z_t=\operatorname{normalize}\bigl(W_2\,\operatorname{GELU}(\operatorname{LN}(W_1x_t))\bigr),
\quad W_1:1024\to512,\quad W_2:512\to512.
\]

`LayerNorm` chỉ chạy theo channel của từng clip. Nhánh này không có full-sequence attention, pooling toàn video trước projection, hoặc normalization dùng thống kê nhiều timestep. Với cấu hình này, một `z_t` không thể nhận nội dung hình ảnh từ clip xa thông qua local head; tuy nhiên nó vẫn có thể chứa signer, nền hình và ngữ cảnh trong cửa sổ I3D.

CiCo mô tả window 16 frame, stride mặc định 1; dataloader retrieval có thể lấy 64 feature cách đều. Vì thế, hai token liền nhau sau sampling không nhất thiết là hai cửa sổ sát nhau trong video gốc. Cần giữ `original_clip_index` và timestamp nếu có. Nếu chỉ có file feature không có mapping đáng tin, chỉ gọi đầu ra là **tập clip có support cao**, không công bố độ chính xác timestamp hay biên dấu. [CiCo, §3 và ablation stride](https://arxiv.org/abs/2303.12793).

MVP giữ cùng số clip quan sát như baseline. Tăng độ phân giải thời gian là ablation riêng, với baseline cũng được dùng số clip mới. Không nhận cải thiện nhờ nhìn thêm frame là hiệu quả riêng của paired loss.

### 8.4. Text atom: tránh đưa toàn câu vào nhãn “cục bộ”

Đối với bản đầu, query atom là các lexical unit còn lại sau bộ lọc từ chức năng xác định trước; giữ negation và số. Tokenization, dấu câu và stopword list phải được version hóa. Phần khác biệt dùng trong paired loss là toàn cụm `U_i` hoặc `U_j` từ một vùng replace, kể cả từ chức năng bên trong cụm.

Mỗi atom/cụm được encode **riêng** bằng text encoder CiCo đã đóng băng, lấy pooled embedding theo adapter xác minh từ code rồi normalize. Không lấy hidden state của từ trong full caption làm nhãn local mặc định: hidden state đó có thể chứa chính ngữ cảnh chung mà ta muốn kiểm soát. Cũng không giả định một từ English tương ứng đúng một dấu; atom là đơn vị truy vấn yếu, không phải gloss annotation.

Nếu text encoder đã fine-tune toàn câu quá kém với input ngắn, đây là phép thử có thể làm proposal thất bại. So sánh có kiểm soát với frozen CLIP gốc là một ablation, không mặc định thêm text model mới vào pipeline.

Các atom trùng nhau trong một query được gộp để không tăng trọng số chỉ do lặp từ. Bản đầu dùng trọng số đều. Train-IDF có clipping và phrase atom 2–3 từ là hai ablation riêng; không thêm parser/LLM vào cấu hình chính để che lấp vấn đề của supervision.

Với query dài bị dataloader nền subsample BPE, nhánh local chỉ dùng các lexical unit còn đủ token trong phần text được nền quan sát. Offset mapping phải được dựng bằng đúng tokenizer. Một thí nghiệm dùng toàn bộ query phải cấp cùng quyền truy cập văn bản cho baseline; nếu không, đó là thay đổi input budget. Query không có atom hợp lệ nhận `L(V,Q)=0`.

### 8.5. Local support và score dùng khi inference

Cho một tập clip hợp lệ `W`, phrase embedding `e_p` và nhiệt độ dương `tau_l`:

\[
r(W,p)=\tau_l\log\left[\frac{1}{|W|}\sum_{t\in W}
\exp\left(\frac{z_t^\top e_p}{\tau_l}\right)\right].
\]

Đây là log-mean-exp trên cosine similarity. Trong code, dùng masked `logsumexp`, trừ `log(valid_count)` và tính reduction bằng FP32. Không thay padding bằng zero trước softmax. Tập `W` rỗng bị loại khỏi paired loss; không clamp để biến nó thành một mẫu âm giả.

Với toàn bộ clip hợp lệ của video và tập atom `P(Q)`:

\[
L(V,Q)=\sum_{p\in P(Q)}w_p\,r(V,p),\qquad
w_p\ge0,\quad\sum_p w_p=1.
\]

Score cuối là:

\[
S(V,Q)=B(V,Q)+\lambda L(V,Q),\qquad \lambda\ge0.
\]

Tính cùng `S[video,text_group]` cho hai hướng rồi đưa vào evaluator chính thức. Inference **chỉ cần video và query đang xét**; không cần biết caption đúng, cặp donor hay word difference của query với ground truth. MVP chấm toàn bộ gallery theo chunk, không chuyển sang một stress test có gallery tự tạo.

Vì embedding được normalize, `r` và `L` nằm trong `[-1,1]`. Bởi vậy, nếu margin nền giữa hai ứng viên lớn hơn `2*lambda`, residual không thể đảo thứ tự của chúng. Chứng minh chỉ dùng chặn `lambda*(L_a-L_b) >= -2*lambda`; không cần giả định nội dung clip đúng. Nó **không bảo đảm** R@1 không giảm, vì các query khó thường có margin nhỏ.

Log-mean-exp bất biến khi lặp lại toàn bộ tập clip cùng số lần; nó không có thiên lệch tăng đúng `log(T)` như log-sum-exp chưa chuẩn hóa. Tuy nhiên, thêm clip không liên quan hoặc thay mật độ sampling vẫn có thể đổi score. Đây không phải phép chuẩn hóa loại hết thiên lệch độ dài.

### 8.6. Warm-up và teacher dùng để chọn support

Huấn luyện local head vài epoch trên toàn bộ train bằng retrieval loss ở §8.9, chưa dùng paired loss. Đóng băng một bản sao `theta_0` làm teacher. Bản warm-up này đồng thời là **đối chứng bắt buộc về capacity**: nếu thêm local head và ordinary retrieval training đã tạo toàn bộ cải thiện, paired supervision không có đóng góp riêng.

Trong MVP, teacher không cập nhật lại sau mỗi epoch. Lưu version teacher và danh sách cặp/support được chọn để tránh vòng lặp pseudo-label tự củng cố không quan sát được. Refresh teacher chỉ là thí nghiệm tiếp theo nếu có bằng chứng cần thiết.

### 8.7. Mining cặp thật và lọc support

Quy trình chỉ đọc training split:

1. Tạo candidate bằng TF-IDF và một vùng replace như §4.4. Cặp khác group, không có chuỗi tương đương sau chuẩn hóa, mỗi phía có khác biệt nội dung. Bắt đầu từ tập strict đã audit; không coi toàn bộ tập rộng là supervision sạch.
2. Kiểm tra quan hệ caption. Loại known positive, bản diễn đạt tương đương dễ nhận biết và trường hợp phần khác biệt không có lý do rõ để tạo negative thị giác. Bộ lọc tự động chỉ giảm nhiễu; kiểm tra mẫu video/caption vẫn cần thiết.
3. Với teacher đóng băng, chọn tối đa `k=max(1, ceil(0.25*T_valid))` clip có support cao nhất cho `U_i` trong `V_i`, tạo `W_i`; tương tự tạo `W_j`. Đây là bag support, chưa bắt buộc liên tục theo thời gian. Giữ cùng `W_i` khi chấm cả `U_i` lẫn `U_j`, và cùng `W_j` khi chấm cả hai cụm. Không chọn cửa sổ thuận lợi khác nhau cho từng ô score.
4. Kiểm tra support và margin hai phía bằng teacher. Đánh giá độ ổn định khi bỏ ngẫu nhiên một phần nhỏ clip trong bag hoặc dịch một bước lấy mẫu nếu có mapping timestamp. Cấu hình perturbation cần kiểm tra trên train; không mặc định lật trái–phải hay đảo thứ tự là giữ nguyên nghĩa ký hiệu.
5. Giữ các cặp có margin reciprocal dương và ổn định; lấy reliability weight từ độ ổn định. Teacher confidence là độ tin cậy nội bộ, **không phải xác suất pseudo-label đúng đã calibration**. Báo cáo tỷ lệ cặp bị loại và lý do.
6. Trong số cặp qua lọc, ưu tiên những cặp nền còn xếp hạng gần nhau, nhưng giữ cả một phần cặp dễ làm đối chứng. Giới hạn số lần một phrase/video xuất hiện mỗi epoch; nếu có signer ID, kiểm soát phân bố signer. Source prefix không được dùng như signer label.

Điều kiện ổn định khởi đầu có thể là ít nhất 3/4 perturbation giữ đúng dấu của cả bốn margin teacher. Đây là mặc định kỹ thuật cần kiểm tra, không phải threshold đã tối ưu. Nếu teacher chỉ giữ cặp đã quá dễ, hoặc lọc hết các trường hợp có ích, không nới dần tiêu chí đến khi có kết quả đẹp; phải báo kết quả đó như một thất bại của mining assumption.

Tất cả sample train vẫn tham gia ordinary retrieval loss. Paired loss chỉ là auxiliary vì strict candidate của How2Sign mới phủ 5,73% anchor trước kiểm tra visual. Không nhân bản tập nhỏ này thành phần lớn batch mà không so một baseline có cùng tần suất sample.

### 8.8. Paired local loss

Tính bằng local head đang học, trên support bag do teacher cố định:

\[
R_{ab}=r_\theta(W_a,U_b),\qquad a,b\in\{i,j\}.
\]

Bốn margin là:

\[
\Delta_1=R_{ii}-R_{ij},\quad
\Delta_2=R_{jj}-R_{ji},\quad
\Delta_3=R_{ii}-R_{ji},\quad
\Delta_4=R_{jj}-R_{ij}.
\]

Hai margin đầu yêu cầu video phân biệt hai cụm; hai margin sau yêu cầu cụm phân biệt hai video. Với reliability weight `c_ij` không nhận gradient:

\[
\ell_{ij}=\frac{1}{4}\sum_{h=1}^{4}
\operatorname{softplus}\left(\frac{m-\Delta_h}{\tau_p}\right),
\qquad
\mathcal L_{pair}=\frac{\sum_{(i,j)} c_{ij}\ell_{ij}}
{\max(\epsilon,\sum_{(i,j)}c_{ij})}.
\]

Nếu batch không có cặp, đặt loss phụ bằng zero và log số cặp, thay vì chia cho zero. Không truyền gradient qua candidate selection, phrase segmentation, teacher hoặc `c_ij`.

Đại lượng chẩn đoán:

\[
D=R_{ii}+R_{jj}-R_{ij}-R_{ji}
\]

triệt tiêu các bias cộng riêng theo video hoặc phrase trong mô hình `R_ab=interaction_ab+u_a+v_b`. Điều này không loại được bias tương tác signer–word, không xác lập nhân quả và không bảo đảm cả hai phía đều phân biệt tốt. Vì thế, dùng bốn margin cho loss, không chỉ tối đa hóa `D` để một phía bù cho phía còn lại.

Softplus margin loss và symmetric pair discrimination đều là thành phần tiêu chuẩn. Novelty, nếu có, phải đến từ **nguồn và phạm vi supervision**, được xác minh qua ablation; không đặt một tên loss mới rồi coi công thức là đóng góp lý thuyết.

### 8.9. Retrieval loss và xử lý positive

Với mỗi video `v`, giữ tập text positive đã biết `P_v`, tập bỏ qua `I_v`, và tập mẫu hợp lệ còn lại. Dùng masked multi-positive InfoNCE:

\[
\ell_v=-\log\frac{\sum_{q\in P_v}\exp(S(v,q)/\tau_r)}
{\sum_{q\notin I_v}\exp(S(v,q)/\tau_r)}.
\]

Làm tương tự theo text→video, rồi lấy trung bình hai hướng. `P_v` phải nằm trong denominator. Trong trường hợp mỗi group xuất hiện một lần và không có collision, công thức trở về contrastive loss một positive.

Video cùng group là known positive. Các group khác nhưng có text trùng sau chuẩn hóa được bỏ qua trong denominator của nhau ở bản đầu; không tự coi mọi câu gần nghĩa là positive. Chính sách này áp dụng giống nhau cho mọi mô hình được huấn luyện lại. Official test grouping và metric vẫn giữ nguyên.

Mục tiêu tối thiểu:

\[
\mathcal L=\mathcal L_{retrieval}+\mu\mathcal L_{pair}.
\]

Không thêm distillation, uncertainty head hoặc OT ở MVP. Freezing base và residual có giới hạn đã tạo một đối chứng dễ hiểu về tính ổn định. Nếu cần distillation sau này, phải thêm một thí nghiệm chứng minh lợi ích riêng của nó.

### 8.10. Những failure mode phải chấp nhận từ đầu

| Failure mode | Dấu hiệu nhận biết | Quyết định |
|---|---|---|
| Feature I3D không giữ khác biệt cần thiết | Local probe không phân biệt hơn random/context control | Dừng claim loss-only; xem xét representation ở dự án sau |
| Caption difference không tương ứng khác biệt thị giác | Nhiều ghép chéo vẫn đúng hoặc khó xác định trong audit | Dừng/thu hẹp nguồn supervision, không tăng loss weight |
| Teacher chọn shortcut nền hình/signer | Support ổn định nhưng đổi sang clip không chứa phần ký hiệu vẫn giữ margin | Không gọi support là bằng chứng local đủ tin cậy |
| Local atom làm mất quan hệ hoặc negation | Query có cùng từ nhưng khác cấu trúc bị xếp hạng sai | Giảm residual; thử phrase ablation; không bỏ lỗi khỏi báo cáo |
| Pair coverage quá hẹp | Gain chỉ ở số, ngày tháng, cụm lặp | Không claim khả năng compositional tổng quát |
| Ordinary local head đã đủ | PLEL không hơn capacity/mining controls | Không coi PLEL là đóng góp phương pháp mới |
| Gallery gốc giảm, stress test tăng | Lặp lại trade-off SAN | Không đạt mục tiêu của dự án này |
| Dev chọn `lambda=0` | Local score không có giá trị bổ sung đáng tin | Coi proposal chưa được ủng hộ, không bỏ cấu hình zero khỏi search |

## 9. Đường triển khai không phụ thuộc SEDS

### 9.1. Repository và tài nguyên đầu vào

| Mục đích | Nguồn | Trạng thái xác minh trong nghiên cứu này |
|---|---|---|
| Nền retrieval, dataloader, evaluator | [FangyunWei/SLRT — CiCo](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo) | Đã đọc code liên quan và audit metadata |
| I3D feature do CiCo công bố | [Google Drive feature archive](https://drive.google.com/file/d/1Vb-HFZd-rhjN49sB5WwLRpIbyhiC6xTy/view?usp=share_link) | Link ghi trong README; chưa tải archive lớn/kiểm tra checksum |
| CLCL checkpoint do CiCo công bố | [Google Drive checkpoint](https://drive.google.com/file/d/1Hpcn5obCcG5JHa3nLvHqX9pfrp7g6wDu/view?usp=share_link) | Link ghi trong README; chưa chạy inference từ weights |
| Domain-aware I3D | [Google Drive I3D checkpoint](https://drive.google.com/file/d/1TbX3UjaUvXhsXSQX2UAm81nitt8tsgLq/view?usp=share_link) | Link ghi trong README; phải đối chiếu dataset trước khi dùng |
| I3D khởi tạo ngoài miền | [Oxford BSL checkpoint](https://www.robots.ox.ac.uk/~vgg/research/bslattend/data/bsl5k.pth.tar) | Đường nguồn được CiCo tham chiếu; chưa tải weights |
| Video How2Sign | [Trang dataset chính thức](https://how2sign.github.io/#download) | Cần dùng đúng bản frontal/crop/alignment của pipeline |
| Video PHOENIX-2014T | [RWTH archive](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz) | Kiểm tra split chính thức khi chuẩn bị |
| Video CSL-Daily | [Trang nhóm tác giả](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/) | Quy trình truy cập dataset có thể cần thao tác riêng |
| Đối chứng uncertainty | [xua222/UPRet](https://github.com/xua222/UPRet) | Đã đọc score/loss/script; chưa tái lập |
| Đối chứng sign-aware negative | [joonmy/SAN](https://github.com/joonmy/SAN) | Có code công khai một phần; cần dựng phần pipeline còn thiếu |

**Link được công bố không đồng nghĩa tài nguyên đã tải được và chạy được.** Chưa có xác nhận weights/feature ngoài Baidu đã sẵn sàng trong môi trường huấn luyện. Đây là dependency đầu tiên phải giải quyết, không được giấu trong ước lượng compute.

Ưu tiên tải feature/checkpoint CiCo trước. Nếu feature archive không dùng được nhưng raw video và I3D weights hợp lệ, chạy extractor chính thức. Nếu domain-aware checkpoint thiếu, có thể dựng lại pseudo-label/domain adaptation theo CiCo; chi phí này phải báo riêng. Nếu chỉ tái lập được stream domain-agnostic, công bố đó là nền yếu hơn với điều kiện tài nguyên khác. Không lén thay bằng feature SEDS để giữ tên baseline.

Không cần train raw RGB/pose end-to-end để kiểm tra PLEL lần đầu. Việc train lại domain-aware encoder là một công việc tái lập nền độc lập, không tính như module mới của PLEL.

### 9.2. Hợp đồng dữ liệu

Mỗi video record cần các trường sau; tên bên dưới là **schema đề xuất**, không phải các file đã được triển khai:

| Trường | Ý nghĩa và kiểm tra |
|---|---|
| `dataset`, `split`, `video_id`, `group_id` | ID không trùng giữa train/dev/test; group theo official manifest |
| `text_original`, `text_model`, `text_equivalence_id` | Phân biệt bản ngôn ngữ gốc, input English và collision sau chuẩn hóa |
| `feature_path`, `feature_sha256`, `stream_config` | Xác định chính xác feature và cách mix |
| `valid_clip_mask`, `original_clip_index` | Padding không được thành clip; chỉ số gốc sống qua subsampling |
| `timestamps` | Optional, chỉ dùng nếu truy được về extraction/crop đúng |
| `source_video_id`, `signer_id` | Optional; không suy diễn signer từ tên source video |
| `token_ids`, `word_to_bpe`, `retained_atom_ids` | Cùng tokenizer và giới hạn thông tin giữa baseline/PLEL |

Một paired record lưu hai video/group ID, hai token span, các clip index trong `W_i/W_j`, teacher version, reliability, lý do lọc và hash training manifest. Không lưu test neighbor vào paired cache. Với group có nhiều video, chọn member cụ thể rồi kiểm tra support của member đó; không chuyển support index của video này sang video khác trong group.

### 9.3. Các module cần viết và điểm nối

| Module đề xuất | Trách nhiệm | Điểm kiểm tra quyết định |
|---|---|---|
| `prepare_manifest.py` | Flatten metadata, giữ group và xây dev sạch | Split không giao ID; group counts khớp |
| `base_adapter.py` | Nạp CiCo và xuất feature/score đúng đường inference | Với `lambda=0`, score/rank khớp code gốc |
| `atom_encoder.py` | Offset BPE, encode atom/cụm riêng và cache | Không đưa full sentence vào phrase embedding |
| `local_head.py` | Pointwise projection và masked log-mean-exp | Gradient của một clip không phụ thuộc clip khác qua head |
| `mine_pairs.py` | Lexical candidates → teacher support → reliability | Chỉ đọc train; log funnel và phân bố cặp |
| `paired_loss.py` | Tính bốn margin, mask, weighted reduction | Không gradient qua teacher/mining; đúng cross-pair indexing |
| `train_local.py` | Warm-up và auxiliary training | Checkpoint selection chỉ dùng dev |
| `eval_retrieval.py` | Score gallery theo chunk và gọi evaluator gốc | Hướng matrix, group members và tie policy giữ đúng |
| `analyze_errors.py` | So base/local, gain/loss và support controls | Lưu cả lỗi mới, không chỉ ví dụ thành công |

Nên thêm một package nhỏ cạnh CiCo, giữ mã baseline đã pin để đối chiếu. Các patch cần thiết do phiên bản Python/PyTorch mới phải có danh sách riêng. Sửa import hoặc kiểu NumPy cũ là compatibility fix; sửa softmax mask hoặc metric là thay đổi toán học, cần đối chứng riêng.

### 9.4. Pseudocode một bước huấn luyện

Đây là pseudocode mô tả hợp đồng thuật toán, **chưa phải chương trình đã chạy**:

```python
# base và text encoder đã frozen; mining teacher là bản warm-up cố định.
batch = sample_train_with_known_positive_and_ignore_masks()
x, valid = load_the_same_sampled_i3d_features_as_base(batch)

with no_grad():
    b = base_adapter.score_unscaled_eval_mix(batch)  # [Nv, Nq]
    atoms = frozen_text_encoder.encode_isolated_atoms(batch.queries)

z = normalize(local_head(x), dim=-1)
local = masked_local_query_scores(z, valid, atoms)  # [Nv, Nq]
score = b + lambda_train * local
loss_ret = symmetric_masked_multi_positive_nce(
    score, batch.positive_mask, batch.ignore_mask, tau_retrieval
)

# Lấy một số cặp từ train-only cache. Nếu cần nạp video bổ sung,
# tính chúng vào video count / compute; không coi donor là miễn phí.
pairs = sample_fixed_teacher_pairs_with_matched_sampling_budget()
loss_pair = reciprocal_local_margin_loss(
    local_head, pairs, frozen_phrase_embeddings, fixed_support_indices
)

loss = loss_ret + mu * loss_pair
optimizer.zero_grad()
loss.backward()
optimizer.step()  # chỉ tham số local_head
```

Các đối chứng bỏ paired loss vẫn phải nhận cùng ngân sách sample/forward. Một control hữu ích là dùng đúng các video bổ sung để học ordinary retrieval loss; như vậy, lợi ích không chỉ do PLEL nhìn lại các video khó nhiều lần hơn.

### 9.5. Cấu hình khởi đầu và giới hạn tìm kiếm

Đây là điểm bắt đầu có chủ đích giữ nhỏ, **không phải hyperparameter đã được thực nghiệm xác nhận**:

| Biến | Khởi đầu | Sweep đầu tiên |
|---|---|---|
| Visual/text backbone | Frozen CiCo đã tái lập | Không sweep backbone ở MVP |
| Local head | 1024→512→512, tokenwise LN, GELU | Linear projection làm capacity control |
| `tau_l`, `tau_p` | 0,07; 0,07 | Giữ cố định trước khi có bằng chứng cần đổi |
| `tau_r` | Reciprocal logit scale của nền, frozen | Khai báo rõ nếu dùng temperature khác |
| `lambda_train` | 0,05 | Giữ một giá trị khi screening |
| `lambda_eval` | 0,05 | `{0, 0.025, 0.05, 0.10}` trên dev |
| Margin `m` | 0,02 cosine unit | `{0, 0.02}` nếu pair loss có tín hiệu |
| `mu` | 0,1 | `{0.1, 0.3}` trên dev |
| Batch chính | 32 hoặc 64 video, tùy profiling | Tăng sau khi đo peak memory |
| Paired batch phụ | Tối đa 8 cặp mỗi bước khởi đầu | Giữ video/step đồng đều giữa controls |
| Optimizer local | AdamW, LR `1e-4`, weight decay `0.01` | Một LR trước; không sweep dày |
| Warm-up / paired phase | 3–5 / 10–20 epoch | Early stop theo dev, ngân sách bằng controls |
| Support cap | 25% clip hợp lệ | 12,5% và random matched-size sau screening |
| Seed | Một seed để loại lỗi thiết kế | Ba seed cho những cấu hình còn giá trị |

Không chọn `lambda` bằng test. Cho phép `lambda_eval` khác `lambda_train` là calibration sau học; báo cả hai. Capacity baseline cũng được sweep cùng tập `lambda_eval` và cùng số lần chọn model. Không tính lợi thế do phương pháp có nhiều lần thử hơn.

### 9.6. Phân tích bộ nhớ và compute

MLP nêu trên có khoảng **788.480 tham số** nếu dùng bias ở hai linear và affine LayerNorm. Phần học thêm nhỏ, nhưng scoring all-pairs vẫn có chi phí theo số video, query, clip và atom; không thể suy ra “rất nhẹ” chỉ từ parameter count.

Ước lượng tensor cho How2Sign, giả sử lưu 64 clip/video và FP16:

| Tensor hoặc phép tính | Ước lượng lý thuyết |
|---|---:|
| Một stream I3D, 31.085 × 64 × 1.024 | Khoảng 3,79 GiB trên đĩa/RAM |
| Hai stream I3D với cùng kích thước | Khoảng 7,59 GiB |
| Frozen feature cache chiều 512, cùng số clip | Khoảng 1,90 GiB |
| 100.000 phrase embedding chiều 512 | Khoảng 97,7 MiB |
| Similarity `[128,128,64,32]`, FP16 | 64 MiB cho riêng tensor forward |
| Similarity `[512,512,64,32]`, FP16 | 1 GiB cho riêng tensor forward |

Các số này không gồm gradient, softmax/logsumexp FP32, optimizer, temporary buffers và encoder hoạt động. File feature nguyên bản có thể giữ nhiều hơn 64 clip và có dtype khác; dung lượng thực cần đo. Student `z_t` thay đổi sau mỗi update nên không được cache cố định suốt quá trình học; chỉ cache input và phần frozen. Teacher cache là trường hợp khác vì teacher không đổi.

Với 1 GPU 24 GB, khởi đầu bằng batch nhỏ, frozen backbone và chunk scoring. Với 2 GPU, DDP/all-gather phải giữ đúng gradient của local feature nếu sử dụng negative toàn cục. **Gradient accumulation không tự tạo thêm contrastive negative trong denominator.** Không gọi 8 bước batch 64 là tương đương một batch contrastive 512 nếu mỗi bước chấm độc lập.

Chưa đo peak VRAM, throughput, số giờ train hoặc chi phí extraction trên GPU thực. Vì vậy, không cam kết một thời gian hoàn tất giả tạo. Gate đầu tiên sau khi có feature là profiling 100 bước ổn định và một full evaluation; từ đó mới suy ra ngân sách chạy ba seed.

Ở inference, cache gallery visual embeddings và encode từng query/atom một lần. Tính score theo block query–video; trả rank trên toàn gallery. Nếu thêm top-K reranking ở nghiên cứu sau, phải báo recall@K của nền vì đó là trần cho khả năng cứu một positive nằm ngoài shortlist.

## 10. Thực nghiệm để xác nhận hoặc loại proposal

### 10.1. Thứ tự dataset và metric chính

How2Sign là dataset khám phá chính vì câu đa dạng hơn và là mục tiêu retrieval quan trọng trong nhóm paper ban đầu. PHOENIX dùng để kiểm tra trực tiếp trade-off được SAN báo cáo; chỉ chạy sau khi dev manifest đã làm sạch. CSL-Daily kiểm tra khả năng chuyển thiết kế sang dữ liệu có nhiều video trong một caption group và khoảng cách representation lớn hơn.

Metric chọn model ban đầu:

\[
M_{dev}=\frac{R@1_{T2V,dev}+R@1_{V2T,dev}}{2}.
\]

Luôn báo riêng hai hướng cùng R@5/R@10. MRR là metric bổ sung nếu được tính nhất quán. Không dùng tổng hợp để giấu một hướng giảm mạnh. Với metric khác evaluator paper, gắn nhãn bổ sung và chạy lại mọi đối chứng liên quan.

Giữ official gallery làm đánh giá chính. SAN stress test, nếu tái dựng đúng, chỉ là chẩn đoán phụ; nó không phải benchmark mới của dự án. Các phân tích theo nhóm query chỉ phân loại kết quả trên split hiện hữu, không thay đổi tập positive hoặc gallery chính.

### 10.2. Các đối chứng tối thiểu

| ID | Cấu hình | Câu hỏi được trả lời |
|---|---|---|
| E0 | CiCo gốc, checkpoint/feature tương ứng, evaluator gốc | Pipeline đã tái lập nền chưa? |
| E1 | CiCo huấn luyện lại với dev sạch, positive/ignore policy đã khóa | Thay đổi protocol huấn luyện ảnh hưởng bao nhiêu? |
| E2 | Frozen E0 hoặc E1 + local head, chỉ ordinary retrieval loss | Thêm capacity và score cục bộ đã đủ chưa? |
| E3 | Cùng local head và cùng video/cặp được lấy mẫu, dùng full-query hard-pair loss | Gain đến từ mining hoặc oversampling hay local supervision? |
| E4 | Cùng phrase pairs, support ngẫu nhiên có cùng kích thước, cùng pair loss | Teacher support selection có giá trị không? |
| E5 | PLEL đầy đủ | Hiệu quả của toàn bộ cơ chế đề xuất |
| E6 | E5 nhưng local head đọc feature sau global Transformer | Kiểm soát receptive field có cần thiết không? |
| E7 | E5 bỏ reliability/reciprocal filter, giữ ngân sách cặp tương đương | Filtering giúp gì ngoài chọn sample dễ? |
| E8 | SAN trên cùng nền, training split, ngân sách và evaluator | Có hơn sign-aware negative mining hiện có không? |

E3 không nên là một baseline yếu được chọn tùy tiện. Dùng cùng frozen backbone, cùng trainable head, cùng số video và cùng số bước; thay loss local bằng loss trên score toàn query. Có thể thêm một bản conventional hard-negative fine-tuning đủ năng lực nếu nguồn lực cho phép, nhưng ghi riêng compute.

E7 cần match số cặp và phân bố difficulty càng sát càng tốt: so một tập có lọc nhỏ với toàn bộ tập nhiễu lớn không tách được chất lượng filter khỏi thay đổi tần suất data. E4 không chỉ đổi random seed; phải thay đúng support mask trong khi giữ cặp và loss.

E6 cần giữ số clip, output dimension và ngân sách tham số gần E5. Nếu feature sau Transformer có chiều 512 thay vì 1.024, điều chỉnh hidden width để match parameter count và công bố cấu hình, không gọi hai head có capacity chênh lệch là phép thử chỉ thay receptive field. Video quá ngắn để có support bag khác random bag phải được đếm riêng trong phân tích E4.

E8 chỉ được gọi là “SAN tái lập” nếu các phần tự dựng và khác biệt với paper được ghi rõ. Nếu chưa đủ mã để tái lập, vẫn báo số SAN công bố như bối cảnh và đánh dấu thiếu đối chứng; không âm thầm coi một negative sampler sơ sài là SAN chính thức.

### 10.3. Kiểm tra cơ chế trước khi chạy dài

**Phép thử A — local feature có thông tin phân biệt không?** Trên một mẫu train audit độc lập với cặp dùng để kiểm tra định tính, so margin của support được chọn với bag ngẫu nhiên cùng kích thước và bag ngoài vùng support trong cùng video. Giữ cùng visual/text head. Nếu các score không tách được, chưa có lý do để tin paired loss đang giám sát phần ký hiệu liên quan.

Để tránh circularity, teacher dùng chọn support không được đồng thời là bằng chứng duy nhất rằng support đúng. Cần xem video và nhờ người hiểu ngôn ngữ ký hiệu liên quan đánh giá một mẫu có cả cặp được nhận và bị loại. Đánh dấu riêng: khác biệt thể hiện rõ; có thể đúng nhưng chưa xác định; ghép chéo vẫn hợp nghĩa; caption/alignment không hỗ trợ kết luận. Đây là audit cơ chế trên dữ liệu sẵn có, không phải một bộ nhãn train hay benchmark mới.

**Phép thử B — dấu hiệu shortcut từ ngữ cảnh.** Chạy cùng một local clip qua head pre-Transformer và qua mô hình có global context thay đổi, rồi đo độ nhạy embedding/score. Bản pointwise phải bất biến theo thiết kế nếu input clip giữ nguyên. Nhưng bất biến kiến trúc chỉ xác nhận implementation, chưa xác nhận embedding mô tả đúng sign. Cần xem thêm khả năng phân biệt và lỗi trong A.

**Phép thử C — pair loss có ích cho inference score thực sự không?** Theo dõi bốn margin trên train/held-out train audit cùng với `M_dev` trên gallery gốc. Nếu pair loss giảm nhưng local full-query score không tạo lợi ích ngoài baseline, việc tối ưu support bag không chuyển được sang bài toán đích.

**Phép thử D — quan hệ và phủ định.** Phân tích query có negation, số, tên riêng, động từ và câu có cùng nhóm từ nhưng khác quan hệ. Đếm cả gain lẫn loss của PLEL. Không xem attention heatmap đẹp là đủ bằng chứng cho compositional reasoning.

**Phép thử E — kiểm soát sample.** Báo số video/cặp/phrase thực tế mỗi epoch, độ lặp và phân bố source/signer khi có nhãn. Chạy ordinary retrieval control nhận cùng số lượt xem dữ liệu khó. Nếu control đạt kết quả tương đương, đóng góp thực tế là sample selection chứ chưa phải paired local supervision.

### 10.4. Các gate và tiêu chí dừng

Các threshold dưới đây là **quy tắc vận hành đề xuất trước thí nghiệm**, không phải kết quả hay mức cải thiện được literature bảo đảm. Có thể điều chỉnh sau pilot đo noise, nhưng phải khóa trước khi xem test.

| Gate | Điều kiện qua | Nếu không qua |
|---|---|---|
| G0: tái lập | Feature/checkpoint đúng; `lambda=0` khớp score/rank nền trong tolerance số học và tie policy | Sửa adapter/protocol; chưa thử claim phương pháp |
| G1: supervision | Audit video cho thấy nguồn cặp đủ tin cậy; support tốt hơn random control; không chỉ một vài phrase/source | Dừng paired-loss experiment hoặc thu hẹp claim theo bằng chứng |
| G2: lợi ích ban đầu | E5 hơn E2/E3 trên dev gallery gốc; đề xuất ngưỡng screening khoảng 0,5–1 điểm R@1 trung bình | Không mở rộng sweep kiến trúc chỉ để tìm một seed thắng |
| G3: ổn định | Lợi ích cùng dấu ở phần lớn ba seed; kiểm tra cả hai hướng và noise của chênh lệch | Báo kết quả không ổn định; chưa test claim mạnh |
| G4: đóng góp cơ chế | E4/E6/E7 và matched-sampling controls hỗ trợ phần claim giữ lại | Bỏ claim không có bằng chứng, kể cả khi E5 có gain nhỏ |
| G5: khái quát hóa | Thiết kế khóa từ How2Sign có giá trị trên ít nhất một dataset thứ hai | Giới hạn claim vào dataset; không gọi giải pháp chung |
| G6: mục tiêu paper | So được đối chứng mạnh và literature mới; có hiệu quả, cơ chế và chi phí rõ | Chưa đủ bằng chứng cho submission hàng đầu |

Một guardrail khởi đầu là không chấp nhận một hướng dev giảm quá 0,5 điểm chỉ để tăng trung bình. Nếu seed variance lớn hơn mức này, cần xem uncertainty và kết quả từng seed; không dùng threshold cứng để thay suy luận thống kê.

Không yêu cầu một seed phải vượt SEDS ngay mới được học tiếp; cũng không dùng gain 0,5 điểm trên CiCo để kết luận chắc sẽ vượt khoảng cách hơn 10 điểm trên CSL-Daily. Nếu local feature là nút thắt, tiếp tục tuning paired loss không phải chiến lược hợp lý.

### 10.5. Độ bất định và cách trình bày số liệu

Chỉ những cấu hình vượt screening mới chạy ba seed. Báo từng seed, trung bình và độ lệch chuẩn; ba seed không đủ để tuyên bố một ước lượng variance rất chính xác. Giữ ngân sách lựa chọn hyperparameter ngang nhau giữa E2, E3 và E5.

Để so R@1 trên cùng tập query, có thể bootstrap **chênh lệch paired** giữa hai mô hình, giữ nguyên gallery. Với V2T, nhiều video chung caption/source có thể phụ thuộc nhau: resample theo group hoặc source cluster phù hợp thay vì mặc định mọi video độc lập. Với T2V, dùng text query group và giải thích cách xử lý các group text trùng. Khoảng tin cậy như vậy là có điều kiện trên gallery cố định, không mô tả đầy đủ biến động nếu thay cả corpus ứng viên.

Report cuối cần hai bảng riêng: kết quả tái lập có kiểm soát, và số paper công bố kèm resource/protocol. Không trộn chúng thành một ranking duy nhất rồi đánh dấu bold “ours” khi thiết lập khác nhau.

Mẫu bảng kết quả thực nghiệm — các ô cố ý để chưa chạy:

| Dataset / cấu hình | T2V R@1/5/10 | V2T R@1/5/10 | Mean R@1 | Seed | Peak VRAM | Thời gian |
|---|---|---|---|---|---|---|
| How2Sign / base | Chưa chạy | Chưa chạy | Chưa chạy | — | Chưa đo | Chưa đo |
| How2Sign / E2 | Chưa chạy | Chưa chạy | Chưa chạy | — | Chưa đo | Chưa đo |
| How2Sign / E3 | Chưa chạy | Chưa chạy | Chưa chạy | — | Chưa đo | Chưa đo |
| How2Sign / PLEL | Chưa chạy | Chưa chạy | Chưa chạy | — | Chưa đo | Chưa đo |

## 11. Tự phản biện như reviewer

### 11.1. “Đây có phải chỉ là hard negative mining cộng residual?”

Đây là phản biện mạnh và hiện chưa thể bác bỏ bằng số liệu. Để trả lời, phải có E3, E4 và E6: cùng negative/video, cùng capacity, nhưng khác nơi đặt supervision và khác phạm vi thông tin. Nếu bỏ kiểm soát local mà kết quả không đổi, novelty phải thu hẹp. Công thức residual hoặc loss không đủ cứu claim.

### 11.2. “Cặp caption thật vẫn có false negative; teacher đang tự chấm mình”

Đúng. Real pairs loại được một phần rủi ro ngữ pháp của caption sinh, không loại được quan hệ paraphrase/entailment hoặc khác biệt không được thể hiện trong video. Reliability của teacher chỉ là heuristic. Cần audit mẫu video có người hiểu ký hiệu và đối chứng chống circularity; nếu không có, chỉ được mô tả là consistency-gated weak supervision.

### 11.3. “Tại sao chọn local feature cũ thay vì backbone mới?”

Vì mục tiêu đầu tiên là kiểm tra phương pháp dưới một điều kiện tái lập được với ngân sách đã chọn. Đây là quyết định thực nghiệm, không phải khẳng định I3D là biểu diễn tối ưu. Nếu feature không chứa chi tiết cần thiết, PLEL sẽ có trần thấp. Kết quả đó có thể dẫn đến một nghiên cứu representation khác, nhưng không cho phép thêm backbone lớn rồi gán toàn bộ gain cho loss.

### 11.4. “Chỉ 5,73% anchor How2Sign có strict candidate thì có đủ không?”

Chưa biết. Số cặp có thể đủ cho auxiliary regularization hoặc quá hẹp để generalize. Cần funnel sau visual filter, coverage theo phrase/source và ablation theo lượng cặp. Không có căn cứ để nâng độ phủ lexical thành độ phủ semantic; cũng không có căn cứ để đảm bảo vài nghìn cặp đủ vượt SOTA.

### 11.5. “Model chỉ giỏi dự đoán từ riêng lẻ, không hiểu ngữ pháp ký hiệu?”

Đây là hạn chế thiết kế có chủ đích phải kiểm tra. Local branch cung cấp tín hiệu bổ sung cho base toàn câu, không thay thế nó. Không giả định thứ tự English là signing order, không dùng monotonic matching làm mặc định. Nếu gain chỉ đến từ noun overlap còn negation/quan hệ giảm, phải báo và giảm claim về compositionality.

### 11.6. “Có thể hơn SOTA và nộp A* không?”

Hiện chưa có bằng chứng đủ để trả lời có. How2Sign là mục tiêu thực tế hơn để bắt đầu, nhưng vẫn còn khoảng cách giữa CiCo và các số đã xác minh. Một paper mạnh cần cơ chế có thể kiểm chứng, lợi ích qua dataset/seed, đối chứng công bằng và phân tích chi phí. Vượt một số R@1 riêng lẻ không thay thế những yêu cầu đó, và không có kiểm tra lý thuyết nào trong báo cáo bảo đảm quyết định acceptance.

Nếu PLEL chỉ hơn CiCo ít nhưng chưa hơn E2/E3/SAN trong thiết lập công bằng, không nên đóng gói nó như một phương pháp hoàn chỉnh để nộp vội. Nếu cải thiện nhất quán trên nền CiCo và UPRet với cùng feature, cùng gallery, cùng ngân sách, đồng thời ablation hỗ trợ local supervision, đó mới là nền tảng đáng phát triển thành paper.

## 12. Kế hoạch thực thi và checkpoint kế tiếp

| Giai đoạn | Công việc cụ thể | Đầu ra cần có để chuyển bước |
|---|---|---|
| A. Tái lập tài nguyên | Nạp CiCo weights/features; kiểm tra crop, text, split, scale và grouped metric | Manifest/weights hash; bảng baseline; `lambda=0` parity |
| B. Kiểm tra tín hiệu | Train E2; tạo teacher; mining strict; audit support và false negatives | Pair funnel, sample audit, local-vs-random controls |
| C. Screening phương pháp | E3/E4/E5/E6 trên How2Sign; cùng ngân sách | Kết quả dev theo epoch, lỗi mới, quyết định G2/G4 |
| D. Xác nhận | Ba seed cho cấu hình giữ lại; PHOENIX dev sạch; CSL-Daily nếu tài nguyên đủ | Bảng hai hướng, uncertainty, coverage/compute |
| E. Đối chứng mạnh | SAN tái lập; UPRet cùng feature; bổ sung toàn văn CMCM khi truy cập được | Claim novelty và SOTA được giới hạn đúng |
| F. Paper/code | Chốt ablation, release code/config/manifest logic, viết limitations | Bản thảo có số liệu thật và pipeline tái lập |

Không ấn định lịch GPU theo số ngày khi chưa có profiling và tài nguyên. Sau giai đoạn A–B, quyết định đầu tư tiếp dựa trên tín hiệu đo được, không dựa trên tên phương pháp.

**Checkpoint nghiên cứu hiện tại:** literature/code/metadata audit đã đủ để xác định rủi ro và viết đặc tả MVP; chưa có weights inference, visual alignment audit hoặc thí nghiệm train PLEL. Hành động tiếp theo có giá trị nhất là tái lập CiCo và chạy E2 + kiểm tra support. Thêm một vòng tìm tên module mới lúc này ít giá trị hơn kiểm tra giả thuyết đang yếu nhất.

## Phụ lục A. Phạm vi bằng chứng và khả năng tái lập audit

### A.1. Những gì đã làm, những gì chưa làm

| Loại bằng chứng | Đã thực hiện | Không được suy ra |
|---|---|---|
| Paper nhóm chính | Đọc toàn văn phương pháp/kết quả của SPOT-ALIGN, CiCo, UPRet, SEDS, C²RL, SAN; đối chiếu bảng và protocol | Không đồng nghĩa đã tái lập model |
| Mã nguồn | Trace dataloader, score/loss, grouping/evaluation và training loop; đọc module CMCM ở nhánh master | Không đồng nghĩa mọi repository chạy hoàn chỉnh |
| Nghiên cứu bổ sung | Kiểm tra SLP retrieval Table X; đối chiếu prior art SSAMT, FDCA, GARE, TeachCLIP và phạm vi SignCLIP/SignSeek/GTRN | Không tuyên bố đã audit implementation của tất cả các công trình này |
| Metadata | Đếm group/video/caption, split overlap, candidate coverage từ file CiCo cụ thể | Không có nhãn semantic correctness hoặc sign localization mới |
| Kiểm tra công thức | Chạy kiểm tra NumPy trên dữ liệu mô phỏng | Không phải accuracy hoặc hiệu quả huấn luyện |
| CMCM toàn văn | Chưa lấy được toàn văn có bảng kết quả qua đường công khai đã kiểm tra | Không được điền R@1 hoặc claim đã loại hết đối thủ SOTA |
| GPU experiment | Chưa chạy weights inference, feature extraction hoặc train PLEL | Không có số VRAM/thời gian/SOTA thực đo |

Tìm kiếm có chủ đích theo tên paper, citation/prior art và repository. Ưu tiên bản tác giả, proceedings/publisher và mã nguồn. Các kết quả khác nhiệm vụ được tách khỏi bảng retrieval chính. Đây là nghiên cứu định hướng phương pháp, không phải systematic review có tuyên bố recall tìm kiếm đầy đủ. Một kết quả search không tìm thấy không được dùng làm bằng chứng rằng ý tưởng chưa từng tồn tại.

### A.2. Snapshot repository

| Repository | Snapshot dùng cho các nhận xét code |
|---|---|
| [FangyunWei/SLRT — CiCo](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo) | `38a4f7b00da7a858d59b7fabe5093876a84db8e0` |
| [xua222/UPRet](https://github.com/xua222/UPRet/tree/046366227417e1d8ec14145965403462df345984) | `046366227417e1d8ec14145965403462df345984` |
| [longtaojiang/SEDS](https://github.com/longtaojiang/SEDS/tree/434e3f714fcb6a7d1f4001fb9a246bbd93ec0246) | `434e3f714fcb6a7d1f4001fb9a246bbd93ec0246` |
| [joonmy/SAN](https://github.com/joonmy/SAN/tree/82aba9cbc1beb403abef6e9a3875ca52479805c8) | `82aba9cbc1beb403abef6e9a3875ca52479805c8` |
| [vddong-zjut/CMCM, master](https://github.com/vddong-zjut/CMCM/tree/5d458719d1da2f082e188cc44705003d919e7e97) | `5d458719d1da2f082e188cc44705003d919e7e97` |
| [imatge-upc/sl_retrieval](https://github.com/imatge-upc/sl_retrieval/tree/071f1683b6169c954c95024e49235c7355c29bd9) | `071f1683b6169c954c95024e49235c7355c29bd9` |
| [ozgemercanoglu/sltbaselines](https://github.com/ozgemercanoglu/sltbaselines/tree/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13) | `f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13` |

SEDS trong bảng chỉ là nguồn đọc code; không có artifact SEDS nào tham gia đường huấn luyện đề xuất. Snapshot pin giúp kiểm tra nhận xét trong báo cáo; khi tái lập phải ghi thêm dependency versions và hash weights/features thực sự được tải.

### A.3. SHA-256 của metadata đã audit

Đường dẫn tương đối tính từ `CiCo/CLCL/` của snapshot CiCo ở trên:

| File | SHA-256 |
|---|---|
| `data_h2/train.pkl` | `c3d29cd4797ae54ef1ee76f6c0be2ad2cda494f0e182b4a83db90c4eac16a154` |
| `data_h2/test.pkl` | `a3b71b9f666d53767b79b8efc257287f25356d90dd26470b8f85887960d5c7ff` |
| `data_csl/train.pkl` | `15732a3449f624f21e539d7e5b4bba4c6586bfb25bd53ce1f8d145592d5e4651` |
| `data_csl/test.pkl` | `f043a9897e13289dd8c42de5da0f88d26d86c0b84b1c946c7c88dcb8c9e91997` |
| `data_ph/train.pkl` | `b35ae6f90136b3c936182355c439374ae209bb42046f9f1eb8f9746a8b2c29e2` |
| `data_ph/dev.pkl` | `38bcc1248b7bea5f78b4512a2929df3056d563c3df016b440ed625dab32b1326` |
| `data_ph/test.pkl` | `2e5172db53d85eb6762e3da37f4b10f9c233d3f0e32dc901d3968e4370af4b5e` |

Các bước tái tính coverage: đọc dictionary theo thứ tự lưu; lấy `text` của record đầu trong mỗi group; casefold và tokenize bằng regex ở §4; gộp chuỗi token giống nhau, giữ representative đầu tiên. Fit `TfidfVectorizer` theo §4.4 trên chuỗi token nối bằng dấu cách, giữ tokenizer mặc định của vectorizer cho TF-IDF. Tính sparse cosine bằng nhân ma trận đã L2-normalize; sort `np.argsort(scores)[-21:][::-1]`; áp dụng filter và đếm anchor có ít nhất một neighbor đạt. Undirected edge dùng cặp chỉ số đã sort. Các ties TF-IDF phụ thuộc thứ tự input và phép sort, vì vậy một implementation mới có tie-breaking khác có thể đổi nhẹ candidate count; không so số mà bỏ qua cấu hình đó.

### A.4. Kiểm tra đại số đã chạy

| Kiểm tra bằng NumPy | Kết quả | Ý nghĩa đúng |
|---|---:|---|
| `D` triệt tiêu bias cộng theo hàng/cột | Sai số tuyệt đối tối đa `8.88e-16` | Kiểm tra identity số học |
| Bounded residual, 1.000 hàng × 20 ứng viên, `lambda=0.05` | 822 hàng có margin nền > `2*lambda`; 0 vi phạm trên các hàng đó | Phù hợp chặn thứ tự đã chứng minh |
| Lặp lại toàn bộ clip trong log-mean-exp | Sai số tuyệt đối tối đa `4.44e-16` | Kiểm tra replication invariance |

Các số ở đây dùng dữ liệu ngẫu nhiên mô phỏng và không chứa video/text embedding học được. Chúng không xác nhận G1–G6, không ước lượng gain R@1 và không phải ablation thực nghiệm.

## Phụ lục B. Tài liệu chính để tiếp tục nghiên cứu

Các link đã được đặt cạnh nhận xét tương ứng trong thân bài. Danh mục này giúp truy cập lại toàn văn và repository, không thay thế việc kiểm tra phiên bản khi chạy đối chứng.

1. **Sign Language Video Retrieval with Free-form Textual Queries** — CVPR 2022. [Paper](https://arxiv.org/abs/2201.02495); [code SPOT-ALIGN](https://github.com/imatge-upc/sl_retrieval).
2. **CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning** — CVPR 2023. [Paper](https://arxiv.org/abs/2303.12793); [code](https://github.com/FangyunWei/SLRT/tree/main/CiCo).
3. **Uncertainty-aware Sign Language Video Retrieval with Probability Distribution Modeling** — ECCV 2024. [Paper](https://arxiv.org/abs/2405.19689); [UPRet code](https://github.com/xua222/UPRet).
4. **SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval** — ACM MM 2024. [Paper](https://arxiv.org/abs/2407.16394); [code để tham khảo](https://github.com/longtaojiang/SEDS).
5. **C²RL: Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieval** — arXiv 2024; TCSVT 2025. [Bản arXiv v1 dùng đối chiếu bảng](https://arxiv.org/abs/2408.09949v1); [publisher](https://ieeexplore.ieee.org/document/10933970/).
6. **Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative Mining for Sign Language Retrieval** — ACL 2026, theo bản tác giả. [Toàn văn v1](https://arxiv.org/html/2607.09263v1); [SAN code](https://github.com/joonmy/SAN).
7. **Causality-inspired Multi-grained Cross-modal Sign Language Retrieval** — CVIU, 2026; DOI `10.1016/j.cviu.2025.104631`. [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S1077314225003546); [CMCM master](https://github.com/vddong-zjut/CMCM/tree/master).
8. **Scaling up Multimodal Pre-training for Sign Language Understanding** — arXiv 2024; TPAMI 2025, DOI `10.1109/TPAMI.2025.3599313`. [Toàn văn SLP](https://arxiv.org/html/2408.08544v1).
9. **SignCLIP: Connecting Text and Sign Language by Contrastive Learning** — EMNLP 2024. [Proceedings](https://aclanthology.org/2024.emnlp-main.518/).
10. **Constructing Phrase-level Semantic Labels to Form Multi-Grained Supervision for Image-Text Retrieval** — AAAI 2022. [SSAMT paper](https://arxiv.org/abs/2109.05523).
11. **Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval** — ICLR 2025. [FDCA, proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/2dae7d1ccf1edf76f8ce7c282bdf4730-Paper-Conference.pdf).
12. **Rebalancing Contrastive Alignment with Bottlenecked Semantic Increments in Text-Video Retrieval** — arXiv v5, 2025. [GARE paper](https://arxiv.org/html/2505.12499v5); [code](https://github.com/musicman217/GARE-text-video-retrieval).
13. **Holistic Features are almost Sufficient for Text-to-Video Retrieval** — CVPR 2024. [TeachCLIP proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Tian_Holistic_Features_are_almost_Sufficient_for_Text-to-Video_Retrieval_CVPR_2024_paper.html); [code](https://github.com/ruc-aimc-lab/TeachCLIP).
14. **SignSeek** — bản tác giả tháng 09/2026, nhiệm vụ dictionary retrieval. [Paper](https://arxiv.org/html/2609.03695v1).
15. **GTRN** — Neurocomputing 2025, truy vấn signing video. [Hồ sơ công trình ở tổ chức tác giả](https://ro.ecu.edu.au/ecuworks2022-2026/6028/).

Việc gắn venue với các prior art nhằm xác định nguồn và mức đối chứng phù hợp. Không dùng nhãn A* để suy ra một module chắc chắn chuyển tốt sang sign language retrieval.
