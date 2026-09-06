# Báo Cáo Khảo Sát & Đặc Tả Dữ Liệu Sign Language

**Dự án:** Sentence-Level Sign Language Retrieval / Translation (Proposal 1 & ELSC)
**Ngày cập nhật:** 06/09/2026
**Máy chủ:** `ptitiec`
**Đường dẫn tài liệu:** `/home/haipd/SLR/docs/proposal1/datasets.md`

---

## 1. Tổng hợp toàn bộ Datasets hiện có trên Server

Hệ thống hiện lưu trữ dữ liệu tại 2 phân vùng chính:
1. `/home/dongvk/datasets/`: Chứa các dataset làm việc, script tiền xử lý và đặc trưng trích xuất.
2. `/home/shared_data/sign_language/`: Chứa kho video thô dung lượng lớn dùng chung (How2Sign, CSLDaily).

### Bảng tổng quan phân loại và đường dẫn gốc

| Tên Dataset | Thể loại | Ngôn ngữ | Số lượng mẫu / Clip | Có Sign Video? | Có Sentence Text? | Đường dẫn Root trên máy chủ |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Phoenix14T** | Continuous | DGS $\rightarrow$ Tiếng Đức | 8,257 clips | Có (.mp4) | Có (cả Gloss + Text) | `/home/dongvk/datasets/phoenix14T` |
| **How2Sign** | Continuous | ASL $\rightarrow$ Tiếng Anh | ~35,130 clips | Có (.mp4) | Có (Sentence Text) | `/home/shared_data/sign_language/How2Sign`<br>*(labels bổ trợ tại `/home/dongvk/datasets/How2Sign`)* |
| **CSL-Daily** | Continuous | CSL $\rightarrow$ Tiếng Trung | 20,668 clips | Có (.mp4) | Có (cả Gloss + Text) | `/home/dongvk/datasets/CSL_Daily_Sentence_Crop`<br>*(bản sao tại `/home/shared_data/sign_language/CSLDaily`)* |
| **ASL_Citizen** | Isolated | ASL (Từ đơn) | 83,399 clips | Có (.mp4) | Không (chỉ nhãn Word) | `/home/dongvk/datasets/ASL_Citizen` |
| **MS-ASL** | Isolated | ASL (Từ đơn) | 18,855 clips | Có (.mp4) | Không (chỉ nhãn Word) | `/home/dongvk/datasets/MS-ASL` |
| **WLASL2000** | Isolated | ASL (Từ đơn) | 21,095 clips | Có (.mp4) | Không (chỉ nhãn Word) | `/home/dongvk/datasets/WLASL2000` |
| **youtubeASL** | Metadata | ASL | - | Chưa tải video | Không (chỉ có URL ID) | `/home/dongvk/datasets/youtubeASL` |
| **places365** | Phụ trợ | - | - | Không | Không | `/home/dongvk/datasets/places365` *(Scene recognition)* |

---

## 2. Mô Tả Chi Tiết 3 Bộ Dataset Continuous (Sign Video + Sentence Text)

Ba bộ dữ liệu dưới đây đáp ứng trọn vẹn yêu cầu cho bài toán nhận dạng, dịch thuật và truy vấn ngôn ngữ ký hiệu mức độ câu (**Continuous Sign Language Recognition - CSLR**, **Sign Language Translation - SLT**, **Sign Language Retrieval - SLR**).

```
+-----------------------------------------------------------------------------------+
| Continuous Sign Language Pipeline                                                |
|                                                                                   |
|  [Sign Video (.mp4)]  -->  [Visual Encoder / Backbone]  -->  Video Embeddings    |
|                                                                    |              |
|                                                              (Contrastive / ELSC) |
|                                                                    v              |
|  [Sentence Text]      -->  [Text Encoder]             -->  Text Embeddings       |
+-----------------------------------------------------------------------------------+
```

---

### 2.1. RWTH-PHOENIX-Weather 2014 T (Phoenix14T)

#### A. Tổng quan
- **Mô tả:** Dataset chuẩn mực và phổ biến nhất trong các bài báo khoa học về CSLR và SLT. Thu thập từ chương trình dự báo thời tiết của đài truyền hình Phoenix (Đức).
- **Ngôn ngữ ký hiệu:** Ngôn ngữ ký hiệu Đức (DGS - Deutsche Gebärdensprache).
- **Ngôn ngữ đích (Văn bản):** Tiếng Đức chuẩn (German Spoken Language).
- **Người thực hiện:** 9 người ký hiệu (signers) khác nhau trong trang phục và bối cảnh chuẩn studio.
- **Độ phân giải video:** $210 \times 260$ pixels, tốc độ 25 fps.

#### B. Cấu trúc thư mục & Đường dẫn tuyệt đối
```
/home/dongvk/datasets/phoenix14T/
├── videos_phoenix/videos/
│   ├── train/                                    # 7,096 video clip (.mp4)
│   ├── dev/                                      # 519 video clip (.mp4)
│   └── test/                                     # 642 video clip (.mp4)
└── PHOENIX-2014-T-release-v3/PHOENIX-2014-T/
    └── annotations/manual/
        ├── PHOENIX-2014-T.train.corpus.csv       # File nhãn tập train (7,096 dòng)
        ├── PHOENIX-2014-T.dev.corpus.csv         # File nhãn tập dev (519 dòng)
        ├── PHOENIX-2014-T.test.corpus.csv        # File nhãn tập test (642 dòng)
        ├── PHOENIX-2014-T.train.corpus.pkl
        ├── PHOENIX-2014-T.dev.corpus.pkl
        └── PHOENIX-2014-T.test.corpus.pkl
```

#### C. Thống kê tập dữ liệu
| Tập (Split) | Số lượng Video Clip | Định dạng Video | Định dạng Nhãn |
| :--- | :---: | :---: | :--- |
| **Train** | 7,096 | `.mp4` | CSV, PKL |
| **Dev (Validation)** | 519 | `.mp4` | CSV, PKL |
| **Test** | 642 | `.mp4` | CSV, PKL |
| **Tổng cộng** | **8,257** | `.mp4` | - |

#### D. Cấu trúc nhãn & Dữ liệu mẫu
Các file CSV nhãn dùng dấu gạch đứng `|` để phân tách cột:
`name | video | start | end | speaker | orth | translation`

**Mẫu thực tế trích xuất từ file train:**
```csv
name|video|start|end|speaker|orth|translation
11August_2010_Wednesday_tagesschau-1|11August_2010_Wednesday_tagesschau-1/1/*.png|-1|-1|Signer08|JETZT WETTER MORGEN DONNERSTAG ZWOELF FEBRUAR|und nun die wettervorhersage für morgen donnerstag den zwölften august
11August_2010_Wednesday_tagesschau-4|11August_2010_Wednesday_tagesschau-4/1/*.png|-1|-1|Signer08|ORT REGEN DURCH REGEN KOENNEN UEBERSCHWEMMUNG KOENNEN|mancherorts regnet es auch länger und ergiebig auch lokale überschwemmungen sind wieder möglich
```

- **`orth` (Sign Gloss):** Chuỗi nhãn cử chỉ chữ hoa (e.g. `JETZT WETTER MORGEN DONNERSTAG...`).
- **`translation` (Sentence Text):** Câu dịch văn bản tiếng Đức tự nhiên hoàn chỉnh (e.g. `und nun die wettervorhersage für morgen...`).
- **Đánh giá:** Rất thích hợp làm benchmark chính cho nghiên cứu (kích thước vừa phải, chất lượng nhãn chuyên gia cao, đối sánh trực tiếp được với hầu hết các bài báo SOTA).

---

### 2.2. How2Sign

#### A. Tổng quan
- **Mô tả:** Dataset multimodal quy mô lớn về ASL liên tục, gồm các video hướng dẫn (Instructional / "How-To") đa chủ đề (nấu ăn, thủ công, thể thao, gia đình, công nghệ...).
- **Ngôn ngữ ký hiệu:** Ngôn ngữ ký hiệu Mỹ (ASL - American Sign Language).
- **Ngôn ngữ đích (Văn bản):** Tiếng Anh (English).
- **Góc quay:** RGB phía chính diện (`rgb_front`) độ nét cao.

#### B. Cấu trúc thư mục & Đường dẫn tuyệt đối
> [!IMPORTANT]
> Toàn bộ video clip đã được cắt theo câu (.mp4) và đặc trưng Pose (.pkl) được lưu trữ tập trung tại thư mục chia sẻ `/home/shared_data/sign_language/How2Sign/`.
> Thư mục `/home/dongvk/datasets/How2Sign/` lưu trữ thêm các file nhãn định dạng pickle / python tiện lợi cho training.

```
/home/shared_data/sign_language/How2Sign/
├── train/
│   ├── raw_videos/                               # 31,048 video clip đã cắt (.mp4)
│   ├── train_pose/                               # Tệp pose trích xuất tương ứng (.pkl)
│   ├── train_label/labels.train                  # Nhãn JSON Lines chuẩn
│   └── train_rgb_front_clips.zip                 # Bản nén 33 GB
├── eval/
│   ├── raw_videos/                               # 1,739 video clip (.mp4)
│   ├── eval_pose/                                # Tệp pose tương ứng (.pkl)
│   ├── eval_label/labels.dev.json
│   └── how2sign_realigned_val.csv                # File nhãn TSV/CSV tập dev
├── test/
│   ├── raw_videos/                               # 2,343 video clip (.mp4)
│   ├── test_pose/                                # Tệp pose tương ứng (.pkl)
│   └── test_rgb_front_clips.zip                  # Bản nén 2.4 GB
└── subset_2000/                                  # Tập mẫu thử nghiệm nhanh 2,000 clips

/home/dongvk/datasets/How2Sign/
└── from_uni_sign_source/
    ├── labels.train                              # Pickle file (mapping video -> text, 31,086 mẫu)
    └── labels.test                               # Pickle file (mapping video -> text, 2,349 mẫu)
```

#### C. Thống kê tập dữ liệu
| Tập (Split) | Số lượng Video Clip | Dữ liệu Pose | Nhãn Sentence Text |
| :--- | :---: | :---: | :--- |
| **Train** | 31,048 | Có (`train_pose`) | Có (31,165 câu trong `labels.train`) |
| **Eval (Val)** | 1,739 | Có (`eval_pose`) | Có (1,741 câu trong `how2sign_realigned_val.csv`) |
| **Test** | 2,343 | Có (`test_pose`) | Có (2,349 câu trong `labels.test`) |
| **Tổng cộng** | **~35,130** | Đầy đủ | Tiếng Anh tự nhiên |

#### D. Cấu trúc nhãn & Dữ liệu mẫu

1. **Định dạng JSON Lines (`/home/shared_data/sign_language/How2Sign/train/train_label/labels.train`):**
```json
{
  "name": "--7E2sU6zP4_10",
  "video_id": "--7E2sU6zP4",
  "video_name": "--7E2sU6zP4-5-rgb_front",
  "sentence_name": "--7E2sU6zP4_10-5-rgb_front",
  "start_time": 129.06,
  "end_time": 142.48,
  "duration": 13.42,
  "text": "And I call them decorative elements because basically all they're meant to do is to enrich and color the page.",
  "video_path": "--7E2sU6zP4_10-5-rgb_front.mp4",
  "pose_path": "--7E2sU6zP4_10-5-rgb_front.pkl"
}
```

2. **Định dạng Tab-separated (`how2sign_realigned_val.csv`):**
```tsv
VIDEO_ID	VIDEO_NAME	SENTENCE_ID	SENTENCE_NAME	START_REALIGNED	END_REALIGNED	SENTENCE
-d5dN54tH2E	-d5dN54tH2E-1-rgb_front	-d5dN54tH2E_0	-d5dN54tH2E_0-1-rgb_front	12.97	19.52	We're going to work on a arm drill that will help you have graceful hand movements in front of you.
-d5dN54tH2E	-d5dN54tH2E-1-rgb_front	-d5dN54tH2E_1	-d5dN54tH2E_1-1-rgb_front	20.65	24.28	I call it painting the wall.
```

- **Đánh giá:** Bộ dữ liệu ASL liên tục lớn nhất, từ vựng mở (open domain), rất lý tưởng để huấn luyện mô hình Retrieval quy mô lớn hoặc kiểm chứng tính tổng quát của phương pháp.

---

### 2.3. CSL-Daily (Chinese Sign Language Daily)

#### A. Tổng quan
- **Mô tả:** Dataset ngôn ngữ ký hiệu liên tục tiếng Trung chất lượng cao do USTC xây dựng, tập trung vào giao tiếp thường ngày (gia đình, du lịch, ăn uống, thời tiết, y tế...).
- **Ngôn ngữ ký hiệu:** Ngôn ngữ ký hiệu Trung Quốc (CSL - Chinese Sign Language).
- **Ngôn ngữ đích (Văn bản):** Tiếng Trung giản thể (Simplified Chinese).
- **Đặc trưng bổ trợ:** Toàn bộ video clip đã được crop chuẩn tập trung vào người ký hiệu, đi kèm cả dữ liệu trích xuất keypoints cơ thể và khuôn mặt.

#### B. Cấu trúc thư mục & Đường dẫn tuyệt đối
```
/home/dongvk/datasets/CSL_Daily_Sentence_Crop/
├── videos/                                       # 20,668 video clip (.mp4)
├── train_data_with_num_frames.csv                # File nhãn tập train (18,401 mẫu)
├── dev_data_with_num_frames.csv                  # File nhãn tập dev (1,077 mẫu)
├── test_data_with_num_frames.csv                 # File nhãn tập test (1,176 mẫu)
├── labels.train                                  # File nén pickle
├── labels.dev                                    # File nén pickle
├── labels.test                                   # File nén pickle
├── metadata_meaning_frames.json                  # Metadata số frame có ý nghĩa
├── keypoint/                                     # Pose keypoints thông thường
└── keypoint_focus_hand_w_face/                   # Keypoints tập trung bàn tay & mặt

/home/shared_data/sign_language/CSLDaily/
├── videos/videos/                                # 20,668 video clip (.mp4) (Bản backup)
└── labels.train, labels.dev, labels.test
```

#### C. Thống kê tập dữ liệu
| Tập (Split) | Số lượng mẫu (CSV) | Số lượng Frame trung bình | Nhãn Gloss | Nhãn Sentence Text |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 18,401 | ~80 – 250 frames | Có | Có (Tiếng Trung) |
| **Dev (Val)** | 1,077 | ~80 – 250 frames | Có | Có (Tiếng Trung) |
| **Test** | 1,176 | ~80 – 250 frames | Có | Có (Tiếng Trung) |
| **Tổng cộng** | **20,654 mẫu** *(20,668 file .mp4)* | - | Đầy đủ | Đầy đủ |

#### D. Cấu trúc nhãn & Dữ liệu mẫu
File CSV nhãn chứa đầy đủ các trường thông tin:
`index, name, gloss, text, video_path, full_path, frame_count`

**Mẫu thực tế trích xuất từ file train CSV:**
```csv
,name,gloss,text,video_path,full_path,frame_count
0,S000000_P0000_T00,"['你们', '好']",你们好！,S000000_P0000_T00.mp4,../CSL_Daily_Sentence_Crop/videos/S000000_P0000_T00.mp4,168
1,S000000_P0004_T00,"['你们', '好']",你们好！,S000000_P0004_T00.mp4,../CSL_Daily_Sentence_Crop/videos/S000000_P0004_T00.mp4,108
3,S000001_P0000_T00,['对不起'],对不起！,S000001_P0000_T00.mp4,../CSL_Daily_Sentence_Crop/videos/S000001_P0000_T00.mp4,202
```

- **`gloss`:** Danh sách các từ cử chỉ ký hiệu tiếng Trung (e.g. `['你们', '好']`).
- **`text`:** Câu văn bản tiếng Trung hoàn chỉnh (e.g. `你们好！`).
- **`frame_count`:** Đã tính sẵn số lượng khung hình của từng clip, rất tiện cho việc lọc độ dài hoặc batching theo duration.

---

## 3. Danh Mục Các Dataset Isolated (Từ Vựng Đơn)

Các dataset này lưu trữ video về các từ cử chỉ đơn lẻ (Isolated Sign Language Recognition), **không** chứa câu hội thoại:

1. **ASL_Citizen**
   - **Đường dẫn:** `/home/dongvk/datasets/ASL_Citizen/`
   - **Video:** `/home/dongvk/datasets/ASL_Citizen/videos/` (83,399 video clip)
   - **Nhãn:** `asl_citizen_all_in_one.json`, `label_maps.json` (từ vựng tiếng Anh đơn lẻ, ví dụ: *APPLE*, *IMPOSSIBLE*).
   - **Keypoint:** `/home/dongvk/datasets/ASL_Citizen/keypoints/`.

2. **MS-ASL**
   - **Đường dẫn:** `/home/dongvk/datasets/MS-ASL/`
   - **Video:** `/home/dongvk/datasets/MS-ASL/cut_processed_video/` (18,855 clip)
   - **Nhãn:** `MSASL_train.json`, `MSASL_val.json`, `MSASL_test.json` (1,000 nhãn từ đơn ASL).

3. **WLASL2000**
   - **Đường dẫn:** `/home/dongvk/datasets/WLASL2000/`
   - **Video:** `/home/dongvk/datasets/WLASL2000/WLASL2000/` (21,095 clip)
   - **Nhãn:** `metadata_meaning_frames.json` (2,000 nhãn từ đơn ASL).

---

## 4. Khuyến Nghị Sử Dụng Cho Proposal 1 (Sign Retrieval / ELSC)

1. **Dataset thử nghiệm chính (Primary Benchmark):**
   - Sử dụng **Phoenix14T** làm benchmark cơ sở ban đầu vì dữ liệu gọn nhẹ (8,257 clips), đầy đủ cả chuỗi Gloss và Sentence Text tiếng Đức, giúp kiểm chứng nhanh giả thuyết của ELSC (Evidence-Localized Sign Contrast).

2. **Dataset mở rộng quy mô lớn (Large-Scale Extension):**
   - Sử dụng **How2Sign** khi mở rộng mô hình lên quy mô lớn (35k+ clips ASL-English). Toàn bộ video clips và poses đã có sẵn trong `/home/shared_data/sign_language/How2Sign/`.

3. **Dataset đa ngôn ngữ (Cross-lingual / Non-Latin):**
   - Sử dụng **CSL-Daily** để chứng minh phương pháp ELSC hoạt động hiệu quả trên cả ngôn ngữ phi Latin (ký tự tượng hình tiếng Trung).

4. **Tổ chức thư mục trong code (`/home/haipd/SLR`):**
   - Khuyến nghị tạo symbolic link vào thư mục `data/` trong repo làm việc thay vì copy trực tiếp để tiết kiệm dung lượng đĩa:
     ```bash
     mkdir -p /home/haipd/SLR/data
     ln -s /home/dongvk/datasets/phoenix14T /home/haipd/SLR/data/phoenix14T
     ln -s /home/shared_data/sign_language/How2Sign /home/haipd/SLR/data/How2Sign
     ln -s /home/dongvk/datasets/CSL_Daily_Sentence_Crop /home/haipd/SLR/data/CSL_Daily
     ```
