# Title and Method Name

**Proposal 7: Kiểm toán SLRet và quyết định NO-GO cho phương pháp mới**

Ngày chốt nghiên cứu: **14-09-2026**, Asia/Ho_Chi_Minh. Không đặt tên một “SOTA method” chưa vượt qua kiểm định. Đầu ra là kết luận nghiên cứu theo nhánh NO-GO được cho phép trong guide, kèm thí nghiệm có giá trị thông tin cao nhất để quyết định vòng thiết kế tiếp theo.

# 1. Executive Summary

**Chưa có phương pháp mới đủ bằng chứng để khuyến nghị triển khai như một ứng viên SOTA.** Sau kiểm toán proposal1–6, mã nguồn liên quan, dữ liệu hiện có, literature và hai vòng tìm kiếm/chẩn đoán, bốn hướng khác nhau đều không vượt qua đồng thời các cổng evidence, novelty và fairness. Đây không phải kết luận rằng SLRet không thể tiến bộ.

Ba kết quả quyết định:

- [M] PH dev có 92 lỗi T2V và 87 lỗi V2T lặp lại ở cả ba seed; 70/63 lỗi đó đã nằm trong top 10 của seed 42. Có dư địa phân biệt, nhưng chưa biết thông tin nào thực sự thiếu.
- [M] Chỉ bảy query nằm trong các lớp mà native captions khác nhau bị ánh xạ vào cùng model input. Chúng giải thích khoảng 5–6% lỗi, không đủ cho giả thuyết đây là bottleneck chi phối.
- [V] Một số baseline/protocol không thể trộn trực tiếp: gallery How2Sign, tie/group semantics, feature provenance và checkpoint selection khác nhau. Public SAN trainer dùng test loader để chọn checkpoint; CMCM có lỗi chạy tối thiểu. Không suy diễn các lỗi public code thành cáo buộc về thực nghiệm không công khai của tác giả.

Khuyến nghị: **không chạy thêm chiến dịch support/OT/hard-negative/context-loss dưới tên mới**. Thí nghiệm tiếp theo nên tách **thông tin raw visual bổ sung** khỏi **capacity/compute bổ sung**, trên train/dev với gallery đầy đủ. Thí nghiệm này chưa chạy; không gọi nó là phương pháp mới hay dự báo mức tăng recall.

# 2. Scope and Non-Negotiable Constraints

Mục tiêu vẫn là sentence-level text↔sign-video retrieval trên PHOENIX-2014T, How2Sign, CSL-Daily; không thay bằng translation, recognition, dictionary retrieval hoặc một stress gallery dễ hơn. R@1 hai chiều là trọng tâm, kèm R@5/R@10/MedR/MnR.

Tất cả ELSC, DIVE-SLR, PLEL, OCEM, SSSC, sampling-consistent partial alignment, PMGR, RPCA, R1–R5 và conceptual alternatives trong proposal1–6 đều **CLOSED**. Thiếu log không mở lại hướng. Không dùng SEDS pretrained checkpoint, teacher, feature hay asset dẫn xuất. Chỉ đọc paper/source/primitive annotation khi cần. Không sửa proposal1–6, source datasets hoặc upstream history; không commit/push. Repo mới chỉ đặt trong `third_party/`.

Một baseline mạnh hơn vẫn có thể là control; không được biến việc đổi backbone, thêm pose hay thêm pretraining thành novelty mà không có cơ chế và bằng chứng riêng.

# 3. Evidence Methodology and Research Cutoff

Ký hiệu xuyên suốt: **[V]** trực tiếp kiểm tra source/document; **[A]** tác giả báo cáo; **[M]** đo lại tại workspace; **[I]** suy luận; **[H]** giả thuyết/thiết kế chưa chạy; **[U]** chưa xác minh. [V] paper có một con số không biến con số [A] thành independent reproduction.

Trình tự đã thực hiện: đọc tài liệu lịch sử → audit call paths/gates → lập Negative Results Registry → tìm primary literature → residual diagnostics → bốn RQ → bốn candidate → collision search/four-perspective review → vòng search/diagnostic thứ hai → NO-GO. [Nhật ký truy vấn và phạm vi đọc](Literature_Search_Log.md), [audit triển khai](Implementation_Audit.md), [reading ledger](AUDIT_PROGRESS.md), [candidate record](Research_Questions_and_Candidate_Screen.md) lưu chi tiết.

Tìm kiếm thực tế ngày 14-09-2026 gồm 39 queries: tên nhiệm vụ, core-paper forward keywords, backward references và mechanism-specific adjacent literature. Ưu tiên proceedings/publisher, author preprint, official repo. Đây là targeted research audit, **không phải exhaustive systematic review/PRISMA hay meta-analysis**. Citation graph không được thu thập đầy đủ; không có API citation-count/retraction-database audit. Các bản preprint không được tự động coi là identical với camera-ready. Một số publisher không truy cập được; không điền số đoán.

AI disclosure: AI thực hiện tìm kiếm, đọc nguồn, viết/chạy diagnostic, tổng hợp và soạn báo cáo. Chưa có human sign-language expert validation, human-read certification hoặc external independent review. Bốn “reviewer” là bốn góc phân tích inline của cùng assistant. Người dùng cần kiểm tra kết luận trước quyết định khoa học/công bố. Skills ARS và Deep Research được dùng để tách evidence, RQ, collision review và provenance, không làm bằng chứng thay cho thực nghiệm.

PDF proposal1 được kiểm tra integrity nhưng verdict UNAVAILABLE do xref warning; nội dung đối chiếu với Markdown, không dựa vào page anchor chưa tin cậy. Browser screenshots SPOT/CiCo không thành công. Citation dưới đây dùng section/table locators, không tuyên bố đã kiểm tra hình ảnh bằng mắt.

# 4. Task Definition and Evaluation Protocol

Cho tập video $\mathcal V$, text queries $\mathcal T$, positive relation $P\subseteq\mathcal V\times\mathcal T$ cố định từ annotation protocol. Score $S\in\mathbb R^{N_v\times N_t}$: hàng là video, cột là text. T2V xếp theo cột; V2T theo hàng. Không transpose nhầm vì tên upstream `I2T/T2I`: **cả hai score channels đều video×text**.

Nếu protocol dùng sentence groups $G_g$, phải tạo $\bar S_{gh}=\max_{v\in G_g}S_{vh}$ cho T2V, với $S=\eta S^{(1)}+(1-\eta)S^{(2)}$ **trước max**. Max từng channel rồi mix là phép khác. V2T giữ mỗi performance làm query và text groups làm candidates nếu đó là protocol đã khóa. PH paired-ID không tự đổi thành semantic-positive evaluation.

Với rank zero-based $r_q$, $R@K=100|Q|^{-1}\sum_q\mathbf1[r_q<K]$; MedR/MnR dùng rank+1. Tie policy, positive relation, ordered IDs và metric implementation phải được lưu. Legacy PH T2V tie expansion khác một-rank-mỗi-query; báo song song diagnostic stable ranks nếu cần, không tráo cột chính. [M] Fixture grouped-versus-flat cho R@5 100% versus 50% với cùng score, cho thấy metric change có thể lớn hơn method gain.

Full gallery nghĩa là mọi candidate thuộc version đã khai báo, kể cả các performance lặp. Missing asset phải làm evaluation fail với danh sách ID; không dùng intersection im lặng. Không thay gallery để “khớp số paper”. Train/dev/test IDs tách biệt; mọi checkpoint/hyperparameter/miner chỉ chọn từ train/dev. Test chỉ chạy sau design freeze.

# 5. Dataset Forensics

Nguồn [M]: [initial forensics](evidence/dataset_forensics_initial.json), [How2Sign reconciliation](evidence/how2sign_reconciliation.json); hashes và ordered manifests nằm trong evidence/artifacts. “File tồn tại” không chứng minh feature/pose provenance.

| Local annotation population | Train / dev / test | Unique native text train / dev / test | File status |
|---|---|---|---|
| PH | 7,096 / 519 / 642 | 6,853 / 509 / 630 | Các video được tham chiếu đều tồn tại |
| CSL | 18,401 / 1,077 / 1,176 | 6,578 / 797 / 798 | Các video được tham chiếu đều tồn tại |
| H2 raw TSV hiện có | 31,165 / 1,741 / [U] raw test TSV không còn local | 30,109 train / 1,516 dev / [U] | Thiếu 118 train, 2 dev |
| H2 CiCo primitive annotations | 31,085 / [U] chưa chứng nhận exact dev manifest / 2,348 | Không suy ra từ số video | Thiếu 44 train, 6 test video |

## PHOENIX-2014T

[M] Split IDs không giao nhau. Native-text exact overlaps có 25 train–dev, 32 train–test, 19 dev–test rows. Mọi dev/test filename source prefix đều xuất hiện trong train; 315 dev sources, 331 test sources. Prefix overlap không chứng minh visual duplication hoặc leakage. Dev–test có 152 common prefixes, liên quan 278 test rows; không được dùng test outcomes để chọn hypothesis.

[M] Dev duration trung bình 4.299 s, p95 7.364 s, max 10.88 s theo metadata extraction. 379/519 video có hơn 64 dense windows; 10/519 captions vượt 30 BPE content tokens. Model captions là English translation, không phải native German. Signer/source/length strata đã xuất trong residual evidence; chênh lệch giữa strata là association.

## How2Sign

[M] Raw train duration trung bình 6.854 s, median 5.27 s, p95 17.63 s, max 143.03 s; 1,509 rows nằm trong repeated-text classes. Dev TSV có 1,529 sentence groups/1,741 recordings, trong đó 212 groups có hai performances. Local JSON chỉ có 1,527 recordings và 1,514 exact texts: không phải full raw dev gallery.

[M] JSON khớp chính xác “giữ last TSV row” ở 1,527/1,527 retained groups; first-row chỉ khớp 1,315. Hai group bị thiếu ứng với video thiếu. Đây là measured equivalence, chưa xác định script lịch sử gây ra.

[M] Không ánh xạ `clipN` thành sentence N. Làm vậy tạo 8,301 train/680 test caption conflicts giả. Dùng dictionary sentence key + performance suffix thì toàn bộ 31,085 train và 2,348 test entries khớp local IDs/text, không conflict. Local raw train thêm 80 rows; Uni-Sign test thêm một row. Restricted unpickler từ chối mọi pickle GLOBAL; không load pose/model pickle trong reconciliation.

## CSL-Daily

[M] Lặp caption là cấu trúc quan trọng: 18,356/560/756 rows trong repeated-text classes ở train/dev/test. Dev có 797 text groups cho 1,077 videos; test 798 groups cho 1,176 videos. Dev–test trùng 795 exact texts, phủ 1,173/1,176 test rows. Đây là scripted repeated-performance structure, **không tự động là contamination**. Không biến dev captions thành training supervision hoặc sửa positive sets theo test.

## Dataset-version and gallery caveats

Nominal split count bằng nhau không chứng minh ordered IDs bằng nhau. H2 CiCo/C²RL nêu 31,085/1,739/2,348; SEDS 31,019/1,738/2,348; SPOT v2 train 31,075. Không gộp các version. Full raw H2 test TSV, provenance của một số pose/features và near-duplicate visual audit vẫn [U]. Các giới hạn này được giữ trong claim boundary thay vì “sửa” dữ liệu gốc.

# 6. Reproduction and Baseline Audit

Workspace HEAD: `a5fe287536db55b44ce519050a23d389f27d9c2e`. Pins đã kiểm tra:

| Official source | Commit | Local evidence / limitation |
|---|---|---|
| FangyunWei/SLRT | `38a4f7b00da7a858d59b7fabe5093876a84db8e0` | CiCo call-path/bridge/evaluator; local feature regime không identical paper |
| xua222/UPRet | `046366227417e1d8ec14145965403462df345984` | Wrapper/PDE/OT/RNG repair audit; không có independent completed SOTA reproduction |
| longtaojiang/SEDS | `434e3f714fcb6a7d1f4001fb9a246bbd93ec0246` | Source-only boundary; pretrained assets cấm và không dùng |
| joonmy/SAN | `82aba9cbc1beb403abef6e9a3875ca52479805c8` | Missing mining-bank construction; public default test-based selection |
| vddong-zjut/CMCM | `5d458719d1da2f082e188cc44705003d919e7e97` | Component repo; thiếu full driver/config/result; hai CPU probes fail |
| C²RL | **PUBLIC CODE NOT VERIFIED** | Không thay bằng repo SLT không chính thức |
| SPOT | [U] official training code chưa tìm thấy | Project và paper được đọc; không khẳng định code không tồn tại |

[V] `shared/slr_common/upstream/cico_bridge.py` nhận `[B,F,1024]`, chuyển `[B,1024,F,1]`; CLS visual bị loại khỏi matching mask. `factory.py` kiểm tra architecture/allowlisted missing keys; không phải strict-all load. `data/cico_dataset.py` dùng trusted local pickle và heuristic orientation; `[1024,1024]` cần metadata để giải mơ hồ. Generic upstream `modules.*` imports cần process isolation giữa CiCo/UPRet.

[V] `shared/slr_common/evaluation/cico_eval.py` nonsingleton branch là flat best-positive, không phải grouped-max T2V. PMGR evaluator có grouped-max/stable candidate order nhưng không được dùng PMGR objective như phương pháp mới. Mask-corrected CSL checkpoint giảm 2.384/2.878 pp so với legacy: đây là scoring-policy effect, không phải training failure của một method.

[V] PH baseline/ELSC scores dùng H2-transfer-aware features; OCEM/SSSC dùng local PH-adapted features. ELSC dùng AdamW/cosine thay vì pinned BertAdam; alpha .9 là 90% agnostic. SSSC run đầu RNG order sai bị loại; corrected parity không bằng efficacy. DDP gather/negative-pool behavior chỉ được kiểm tra ở targeted paths, chưa chứng nhận mọi multi-GPU variant.

[V/M] SAN `train_vlp_v2.py:191,273` đánh giá configured test loader từng epoch rồi lưu tên “dev-best”; supplied dev path không được dùng. CMCM `CausalBackdoorAdjuster` lỗi undefined `DEVICE`; attention 64 tokens lỗi mask dimensions, 1,024-token control chạy. [Evidence và script](evidence/new_upstream_audit.json). Actual author runs vẫn [U]. Không tải pretrained để vượt qua các vấn đề này.

# 7. Current SLRet Landscape and Protocol-Aware SOTA Table

Các bảng là **[A] author-reported test results**, không phải chạy lại trong audit. Mỗi ô `a/b/c` là R@1/R@5/R@10 (%). “—” là chưa có số được xác minh cho ô đó, không phải zero. Tách metrics khỏi resource ledger để tránh một leaderboard thiếu ngữ cảnh.

## Track A — standard sentence-gallery reports

| Work/version | Dataset | T2V | V2T | MedR T/V; MnR T/V |
|---|---|---|---|---|
| SPOT-ALIGN v2 | PH | 55.8/79.6/87.2 | 53.1/79.4/86.1 | 1/1; — |
| SPOT-ALIGN v2 | H2 | 32.8/47.7/52.9 | 23.3/48.5/53.7 | 7/7; — |
| CiCo | PH | 69.5/86.6/92.1 | 70.2/88.0/92.8 | 1/1; — |
| CiCo | H2 | 56.6/69.9/74.7 | 51.6/64.8/70.1 | 1/1; — |
| CiCo | CSL | 75.3/88.2/91.9 | 74.7/89.4/92.2 | 1/1; — |
| UPRet v1 | PH | 72.0/89.1/94.1 | 72.0/89.4/93.3 | 1/1; 4.4/4.6 |
| UPRet v1 | H2 | 59.1/71.5/75.7 | 53.4/65.4/70.0 | 1/1; 54.4/76.4 |
| UPRet v1 | CSL | 78.4/89.1/92.0 | 77.0/89.2/92.7 | 1/1; 6.7/5.5 |
| CMCM | PH/H2/CSL | [U] inaccessible tables | [U] | [U] |

Nguồn: [Duarte et al. (2022), Experiments](https://arxiv.org/pdf/2201.02495) <!--ref:spot2022--><!--anchor:section:Experiments-->; [Cheng et al. (2023), Experiments](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/06/CiCo.pdf) <!--ref:cico2023--><!--anchor:section:Experiments-->; [Wu et al. (2024), Tables 1–3](https://arxiv.org/html/2405.19689v1) <!--ref:upret2024--><!--anchor:section:Tables 1–3-->; [Yang et al. (2026), publisher preview](https://www.sciencedirect.com/science/article/pii/S1077314225003546) <!--ref:cmcm2026--><!--anchor:section:Abstract-->.

SPOT H2 v2 khác các later tables chép 34.2/23.6; giữ nguyên version, chưa giải quyết discrepancy. CMCM xuất bản tháng 2-2026 dù DOI chứa 2025. Không suy ra method causal identification từ tên “causality”.

## Track B — stress task

| PH setup | Stress R@1 | Standard T2V/V2T R@1 |
|---|---|---|
| CiCo → +SAN | 17.9 → 39.4 | 69.2/70.1 → 68.1/67.8 |
| GFSLT → +SAN | 16.8 → 49.1 | 67.9/69.4 → 70.2/67.4 |

Stress gallery có caption gốc + 40 generated negatives, không phải full sentence gallery. Confusability là learned proxy, không sign-level human ground truth. [Lee et al. (2026), Experiments](https://arxiv.org/html/2607.09263v1) <!--ref:san2026--><!--anchor:section:Experiments-->.

## Track C — standard task, different resources

| Work | Dataset | T2V R1/R5/R10 | V2T R1/R5/R10 |
|---|---|---|---|
| SEDS | PH | 76.8/91.7/95.3 | 78.7/92.5/95.2 |
| SEDS | H2 | 62.5/75.1/80.1 | 57.9/70.4/74.9 |
| SEDS | CSL | 85.8/94.4/95.6 | 85.4/93.8/95.8 |
| C²RL v1 | PH | 78.7/92.2/94.9 | 77.6/91.3/94.2 |
| C²RL v1 | H2 | 62.4/75.9/80.1 | 57.5/68.4/73.0 |
| C²RL v1 | CSL | 90.3/96.4/97.7 | 88.4/95.7/97.1 |
| SL-1.5M pretraining v1 | PH | 74.5/93.3/95.6 | 75.1/92.1/95.3 |
| SL-1.5M pretraining v1 | CSL | 87.5/95.2/97.6 | 87.2/95.0/97.2 |

Nguồn: [Jiang et al. (2024), Experiments](https://arxiv.org/html/2407.16394v1) <!--ref:seds2024--><!--anchor:section:Experiments-->; [Chen et al. (2024), Table VI](https://arxiv.org/html/2408.09949v1) <!--ref:c2rl2024--><!--anchor:section:Table VI-->; [Zhou et al. (2024), Table X](https://arxiv.org/html/2408.08544v1) <!--ref:scale2024--><!--anchor:section:Table X-->. C²RL/SEDS MedR/MnR không chép khi chưa xác minh; SL-1.5M Table X MedR đều 1. Không chuyển H2 translation result của SL-1.5M thành retrieval.

## Resource/selection ledger liên kết với các bảng

| Work | Input/backbone/text; external supervision | Training/selection; comparability với local PH baseline |
|---|---|---|
| SPOT | RGB frozen I3D; mean/NetVLAD/GEU, GrOVLE; BSL spotting resources, PH language/gloss-related resources | H2 40 epochs, dev geometric-mean recalls; khác encoder/resource, không direct |
| CiCo | RGB dual I3D agnostic/aware; CLIP visual/text sequence encoders; BSL labeled pretraining, target pseudolabel adaptation; PH/CSL translated English | 64 visual/32 text cap; contrastive soft local correspondence; public code available, local partial reproduction only |
| UPRet | CiCo-like features + Gaussian/PDE/OT; CLIP; external sign prior inherited | Paper 200 epochs, four A100s; OT training-only; exact selection/gallery identity [U]; partial comparator after repairs |
| SEDS | Frozen RGB + online pose GCN; 49 joints, RTMPose, SignBERT hand prior, CLIP | 200 epochs; different H2 retained population; assets prohibited; reported-only comparator |
| C²RL | RGB ResNet18+temporal conv; content/context pretraining; separate mBART encoders | 200 pretrain +80 retrieval epochs, eight 3090s; test ablations reported, selection provenance [U]; PUBLIC CODE NOT VERIFIED |
| SAN | Sign–word mining/teacher; supplied trainer German BERT + mBART-form visual encoder | Test-loader selection in public default; mining bank unavailable; not a fair local reproduced baseline |
| CMCM | Augmentation adjustment, Gaussian attention, temporal covariance; full resource detail [U] | Public components incomplete, tables unavailable; cannot rank numerically |
| SL-1.5M | Pose-only 79 joints, multilingual 1.5M-scale corpus, masked pose + contrastive pretraining | 100 epochs/eight A100s; 60 downstream epochs; prose/table optimizer/LR discrepancies; Track C |

Các resource statements dựa trên method/settings của các primary sources tương ứng, phạm vi được ghi trong search log. “Gloss-free target” không có nghĩa không dùng external sign labels. Full-gallery identity/selection hashes không được công bố đủ để chứng nhận equal protocol chỉ bằng table count.

**Strongest reported trong các số đã xác minh:** PH T2V C²RL 78.7, V2T SEDS 78.7; H2 SEDS 62.5/57.9; CSL C²RL 90.3/88.4. Đây không phải chứng nhận overall SOTA đến cutoff vì CMCM numeric gap/citation coverage còn [U]. **Strongest fairly reproduced dưới local regime chưa được chứng nhận trên cả ba datasets.** PH saved baseline tốt nhất theo bidirectional dev mean là seed 42 (74.181/76.301), không được đối chiếu trực tiếp với test table hoặc chọn seed để quảng cáo.

# 8. Postmortem of Proposal 1–6

## Negative Results Registry

| Family | Hypothesis/intervention đã đóng | Evidence quyết định | Bài học |
|---|---|---|---|
| ELSC | Teacher support, local lexical margins, residual adapter | [M/V] Three-seed mean +0.096 pp; beats matched controls 1/3; hai seed chọn epoch −1 | Auxiliary improvement không đủ; initialization không phải learned gain |
| DIVE | Rival-specific local RGB/pose quartet, reference-subtracted evidence | [V] Fixture/integration có; native efficacy chưa chứng minh, dependencies thiếu | Không biến fixture thành scientific success; không tải SEDS để cứu hướng |
| PLEL | Real near-pair phrase evidence reranking | [U] Negative run mechanism chưa khôi phục; user closure rõ | Đổi synthetic thành real pair không tạo cơ chế mới |
| OCEM | RF overlap/capacity/null assignment | [M] 21/259 errors=8.108% <10%; adjusted effect 0.000695, CI [−0.04315,0.04480] | Target proxy không có đủ support; old GO bị gate mới thay thế |
| SSSC | Same-support contrast for changed words | [V] RNG-invalid run 870/2600 steps; corrected parity, chưa completed attributable efficacy | Parity fix không phải method gain |
| Sampling-consistent alignment | Coverage masses, partial OT, EMA stability | [U] Variant-specific empirical mechanism thiếu; CLOSED | Không thay dustbin/solver để khởi động lại |
| PMGR | Actual grouped population risk, current complete performances | [V] C4 65.71725 versus C3 65.60809, +0.10916 pp <0.5; R5/R10 giảm | Population mismatch không đủ chứng minh useful headroom |
| RPCA | Context generation + retrieval-protected optimizer step | [U] Causal efficacy chưa xác minh; CLOSED | Không gradient-surgery/decoder remix |

Registry đầy đủ về supervision/modules/confounders và conceptual alternatives: [Negative Results Registry](Negative_Results_Registry.md). “CLOSED” là constraint thiết kế của user, không phải theorem rằng mọi thành viên trong scientific family đều thất bại.

## What has already been ruled out

Không dùng lại reliability RGB/pose gates, signer/nuisance invariance, hard monotonic alignment, LLM paraphrase scaling, RCES, boundary span ensembles, relation/GW matching, soft/equivalence positives, query-bank calibration, generic temporal SSL, global composition/order hoặc scale-only backbones. Không phối hai ý tưởng bị loại để tạo tên mới.

# 9. Residual Error Analysis

[M] Dùng đúng 519 PH dev ordered IDs và saved score/checkpoint hashes; không load test scores. Recompute ranks trùng artifacts. Nguồn: [PH residuals](evidence/ph_dev_residuals.json), [second-cycle diagnostics](evidence/information_channel_diagnostics.json).

| Diagnostic | Kết quả | Không được suy ra |
|---|---|---|
| Three baseline seeds T2V/V2T R1 | 42: 74.181/76.301; 1337: 74.181/75.337; 2026: 73.988/75.915 | Best seed = expected method effect |
| Seed42 continuation vs release init | +0.482 pp mean; 95% source-cluster CI [−0.493,1.509] | Baseline continuation có significant gain |
| Short ≤64 dense windows | n140, R1 53.571/56.429 | Long-video truncation là main bottleneck |
| >64 windows | n379, R1 81.794/83.641 | More windows causally improve recall |
| Caption overflow | n10; 3 errors mỗi chiều | Text truncation giải thích đa số lỗi |
| Exact input collisions mới | 2 classes/7 rows; score columns identical | Khác native string = khác sign meaning |
| Persistent errors | 92 T2V/87 V2T all-three wrong | Cần thêm stream hoặc raw detail chắc chắn |
| Persistent inside top10 seed42 | 70/63 | Oracle reranking là achievable result |

Bootstrap 10,000 draws resample 315 filename-source clusters, **conditional on fixed gallery**. Không tạo lại gallery theo bootstrap vì sẽ thay nhiệm vụ. Không giả vờ 519 queries là 519 independent training runs. Seed oracle 82.274/83.237 chỉ dùng để mô tả disagreement; không đưa vào bảng SOTA.

# 10. Research Gaps

G1: [M→U] Có lỗi ranking ổn định nhưng chưa phân biệt thiếu thông tin đầu vào, representation bottleneck hay optimization. Các correlation không xác định nguyên nhân. G2: [V] Common-protocol reproduction yếu hơn độ chính xác của các headline numbers; trước khi claim gain cần đóng gallery/selection/resource contract. G3: [U] Độ đúng ngữ nghĩa của những near-miss chưa được expert kiểm định; automatic caption difference không thay human sign evidence.

Tổng hợp liên nguồn: representation-rich systems có reported gains nhưng resource khác; SAN stress gains không đảm bảo standard gains; local negative gates không ủng hộ việc tiếp tục chỉnh matching theo trực giác. Vì vậy bước hợp lý là một crossed information experiment, không thêm loss. Không coi ít paper về một bottleneck là bằng chứng bottleneck đó tồn tại.

# 11. Research Questions and Falsifiable Hypotheses

| RQ đã ghi trước candidate | Hypothesis; diagnostic | Invalidation/status |
|---|---|---|
| Q1: Bao nhiêu PH lỗi do native distinctions bị collapse trong model input? | ≥10% directional errors; exact partition + optimistic slice correction | [M] 5–6%, reject dominant-collapse hypothesis; broader mistranslation [U] |
| Q2: Lỗi có ổn định qua seeds? | ≥50% seed42 errors recur all-three; compare IDs/ranks | [M] 68.66%/70.73%, giữ stable-error question; không causal conclusion |
| Q3: Raw visual information thêm có lợi ngoài capacity/compute? | ≥0.5 pp mean dev R1 beyond matched controls, ≥2/3 seeds | [H] Chưa chạy; reject nếu shuffled/uninformative control bằng hoặc tốt hơn |
| Q4: Gain có tồn tại sau common protocol repair? | IDs/positives/selection giống nhau, gain ≥0.5 pp | [H] Reject method attribution nếu gain mất khi repair chung |

FINER averages Q1/Q2 4.4, Q3 4.0, Q4 4.4; không criterion <2/5. Các score là feasibility judgment, không measurement. Subquestions, evidence motivation, prior coverage và blueprint nằm trong [RQ record](Research_Questions_and_Candidate_Screen.md). Chúng không mở rộng scope sang task khác.

# 12. Candidate Methods Considered

## Candidate A

**Native-caption information preservation:** giữ native+English thay vì chỉ translation; gradient từ full-caption retrieval vào text mapping, video scorer giữ nguyên. Cần bilingual prior/extra text pass. Gap liên quan collisions là có thật nhưng nhỏ; multilingual encoding đã có. Không support mining/OT/generation, song chưa có substantive novelty và không giải thích H2 English. **Reject.**

## Candidate B

**Inverse-graphics articulation:** raw hand crops → differentiable geometry → sentence representation; retrieval và reconstruction gradients vào visual estimator. Có thể bổ sung measurement, nhưng chưa có residual evidence về hand ambiguity; thêm pose/backbone là conceptual closure. Occlusion/monocular ambiguity và thiếu non-manual signals là rủi ro. **Reject.**

## Candidate C

**Budgeted active visual acquisition:** dùng policy mua thêm raw crop/frame khi cần, có cost term, rồi score/rerank. Khác reweight existing tokens về ý định, nhưng adaptive sampling đã có; local/support/gating collision nếu không thực sự lấy measurement mới. Full-gallery query-dependent cost chưa có justification. **Reject.**

## Candidate D

**Conditional visual likelihood ratio:** score log p(video features|text)−log p(video features), thay vì local matching. Không sentence decoder/protected gradients, nhưng density-ratio/generative retrieval không mới về nguyên lý; chưa evidence density modeling cải thiện ranks. Cheap Gaussian case có thể algebraically collapse; general case đắt và nuisance-sensitive. **Reject.**

## Collision analysis

Adjacent primary anchors: [Lei et al. (2021), mTVR abstract](https://arxiv.org/abs/2108.00061) <!--ref:mtvr2021--><!--anchor:section:Abstract-->; [Hu et al. (2021), HMA Our Approach](https://cdn.aaai.org/ojs/16247/16247-13-19741-1-2-20210518.pdf) <!--ref:hma2021--><!--anchor:section:Our Approach-->; [Hu et al. (2022), mmSampler abstract](https://proceedings.mlsys.org/paper_files/paper/2022/hash/d59a1dc497cf2773637256f50f492723-Abstract.html) <!--ref:mmsampler2022--><!--anchor:section:Abstract-->; [Jin et al. (2023), DiffusionRet abstract](https://arxiv.org/abs/2303.09867) <!--ref:diffusionret2023--><!--anchor:section:Abstract-->.

Đây là tiền lệ cơ chế, không khẳng định equations/tasks identical. Closest SLRet lần lượt CiCo/C²RL/SAN, SEDS, CiCo/DIVE-local variants, UPRet. Claim “chưa thấy đúng tên này” không chứng minh novelty.

## Reviewer panel

| Candidate | Novelty | Actual retrieval | Sign-language validity | Reproducibility |
|---|---|---|---|---|
| A | Generic multilingual branch | Ceiling nhỏ, broader cause chưa rõ | Native spoken text không phải sign syntax | Extra prior, weak H2 transfer |
| B | HMA/extra-stream collision | Geometry quality ≠ recall | Hands không bao phủ toàn ngữ nghĩa | New extraction/prior/license [U] |
| C | Adaptive sampling prior | Shortlist oracle chỉ là bound | Query crop có thể bỏ simultaneous articulators | Decode/encode cost trên mọi candidate |
| D | Generative/density-ratio precedent | Nuisance density, V2T cancellation | Caption underspecifies visual realization | Chưa có permitted working pipeline |

## Candidate scoring

Weights đúng guide: evidence20/mechanism20/novelty20/falsifiability10/fairness10/implementation10/resources5/distance5. A=53, B=31, C=37, D=30 /100; breakdown trong candidate record. Fatal criticism loại candidate bất kể tổng điểm.

**Vòng 2:** A2 native-only trở thành baseline control, không method; C2 fixed raw acquisition bỏ policy để kiểm tra information channel, là diagnostic; D2 yêu cầu nontrivial density score nhưng không có evidence mới, Gaussian reduction và V2T cancellation làm yếu justification. Scores A2/C2/D2 51/40/28, không vượt gate. Hai vòng đã có search, phép đo mới, analytic/synthetic tests và redesign; không chỉ đổi tên/cập nhật status.

# 13. Selected Method

**Không chọn phương pháp nào. NO-GO theo §12 của guide.** Không gọi candidate “conditional GO” chỉ vì có thể viết code. Các mục toán/architecture/training sau đây mô tả **baseline contract và unresolved diagnostic**, không giả làm complete specification của một surviving method.

## Core idea

Thí nghiệm có giá trị cao nhất: tách raw-information channel khỏi additional compute/capacity. Chỉ khi intervention vượt equal-input/equal-compute controls mới thiết kế một cơ chế mới ngoài registry.

## Why it should work

[H] Đây là lý do diagnostic có thể hữu ích: persistent near-miss pool cho phép kiểm tra information-sensitive gain. Không có bằng chứng hiện tại đủ để nói intervention sẽ cải thiện retrieval.

## Why existing methods do not already solve it

Không tuyên bố họ không giải quyết: nhiều priors đã khai thác pose, contextual pretraining hoặc sampling. Điều chưa biết là **nguyên nhân lỗi trong đúng local resource regime**, không phải sự tồn tại của một tên module.

# 14. Mathematical Formulation

Baseline notation: raw $x_v\in\mathbb R^{T\times3\times H\times W}$; fused frozen features $h_v\in\mathbb R^{F\times1024}$; text IDs $u_t\in\mathbb N^L$. Encoders $E_v,E_t$ trả normalized tokens $V\in\mathbb R^{F\times d},T\in\mathbb R^{L\times d}$, valid masks $m_v,m_t$. $A_{fl}=V_f^\top T_l$.

Một **mask-corrected reference** late-interaction channel là

$$s_{v\to t}=\frac{\sum_f m_{vf}\sum_l \operatorname{softmax}_{l:m_{tl}=1}(A_{fl}/\tau_a)A_{fl}}{\sum_f m_{vf}},\qquad
s_{t\to v}=\frac{\sum_l m_{tl}\sum_f \operatorname{softmax}_{f:m_{vf}=1}(A_{fl}/\tau_a)A_{fl}}{\sum_l m_{tl}}.$$

Mixed score $S=\eta s_{v\to t}+(1-\eta)s_{t\to v}$; logits $\ell=S/\tau_c$. Legacy inner padding khác công thức masked này: phải chọn policy chung và báo parity delta, không sửa riêng M. Empty valid sequence làm fail. Không chèn CLS vào counted temporal support.

Với batch paired IDs, reference loss $\mathcal L_0=\frac12[\operatorname{CE}(\ell,y)+\operatorname{CE}(\ell^\top,y)]$. $y$ diagonal chỉ hợp lệ khi protocol/sampler bảo đảm positive relation đó. Grouped datasets phải giữ existing baseline objective được khai báo; không thêm soft positives để “sửa” false negatives và gọi đó là novelty. Duplicate handling và negative pool phải identical giữa arms.

Diagnostic representation $z_v=H_\theta(h_v,r_v)$ với raw measurement $r_v=F_\phi(x_v)$; control thay $r_v$ bằng compute-matched measurement không thêm thông tin hoặc train-only permuted measurement. Dùng $\mathcal L_0$ duy nhất. Không new support loss, no teacher, no mining bank; permutation chỉ diagnostic và không dùng dev labels để fit. $\nabla\mathcal L_0$ vào $H_\theta,E_v,E_t$ theo freeze policy chung; raw extractor chỉ train nếu cùng initialization/prior exposure cho mọi arm. Đây là schema của phép thử, không approved new module.

Analytic rejection D2: nếu $p(z|t)=\mathcal N(\mu_t,\sigma^2I)$ và $p(z)=\mathcal N(0,\sigma^2I)$,

$$\log p(z|t)-\log p(z)=\frac{z^\top\mu_t}{\sigma^2}-\frac{\|\mu_t\|^2}{2\sigma^2}.$$

Nếu $\|\mu_t\|=1$, rank tương đương dot product hai chiều. Với general density, trừ $\log p(z)$ không đổi V2T vì là hằng theo candidate text; không phải proof mọi conditional density vô ích. [M] Bốn synthetic tests kiểm tra partition/reductions pass. Density-ratio scoring đã xuất hiện trong [van den Oord et al. (2018; v2 2019), §2.2–2.3](https://arxiv.org/html/1807.03748v2) <!--ref:cpc2018--><!--anchor:section:2.2–2.3-->; background-likelihood correction có tiền lệ [Ren et al. (2019), abstract](https://papers.neurips.cc/paper_files/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html) <!--ref:ren2019--><!--anchor:section:Abstract-->.

# 15. Architecture and Data Flow

```text
locked train/dev manifests + resource hashes
                 ↓
same frozen baseline features ── optional permitted raw measurement
                 ↓                         ↓
       equal-capacity diagnostic head / control head
                 ↓
      same text encoder + same full-caption scorer
                 ↓
   full score rectangle → locked grouping/ties → dev metrics
```

Interface dự kiến, chưa implement: `encode_measurement(rgb, frame_mask) -> [B,F,D_r]`; `combine(h, measurement, valid) -> [B,F,1024]`; bridge trả token encodings; evaluator nhận score rectangle cùng ordered IDs/positive map. Không đưa query label vào acquisition; không fit feature selection từ dev errors. Dev slices chỉ dùng phân tích sau score.

Không có new architecture được chấp thuận, vì raw encoder/provenance chưa khóa. Chọn một backbone cụ thể lúc này chỉ để điền sơ đồ sẽ đảo ngược evidence→method order.

# 16. Training Objective and Optimization

Diagnostic giữ $\mathcal L_0$, không auxiliary-loss tuning. Khóa optimizer type, LR schedule, weight decay, batch/effective batch, number of updates, precision, dropout/augmentation, RNG call order và gradient accumulation giữa arms. Ghi initial checkpoint hash và dev evaluation tại epoch −1; nếu initialization thắng thì báo “no learned gain”.

Collapse không được “ngăn” bằng một khẩu hiệu: kiểm tra finite features, variance/effective rank và constant-score reference; positive-vs-negative loss/ranks phải tốt hơn control. Contrastive objective có thể có collapsed stationary points, nên không tuyên bố nó bảo đảm không collapse. Same positives/negatives, no test-derived pseudo labels. Nếu thay freeze policy, đó là một factor cần crossed control, không âm thầm thêm capacity.

# 17. Inference Procedure

Primary diagnostic phải encode tất cả video/text của locked gallery rồi score mọi pair theo chunking; không shortlist trong primary. Cache chỉ khi encoder frozen, key gồm checkpoint/manifest/preprocessing hash. Raw extraction thời gian và disk phải tính vào cost. No teacher/miner/decoder ở inference trừ khi explicit resource arm cần raw extractor.

Token matching time $O(N_vN_tFLd)$; naive interaction memory $O(N_vN_tFL)$, chunk về $O(C_vC_tFL)$. Encoder caches $O(N_vFd+N_tLd)$. Group max thêm $O(N_vN_t)$; sorting khoảng $O(N_tN_v\log N_v+N_vN_t\log N_t)$. Raw branch cộng toàn bộ decode/extract cost. Không quảng cáo sublinear retrieval khi vẫn dùng all-pairs.

# 18. Why This Is Not Proposal 1–6 Again

# Why This Is Not Proposal 1–6 Again

Không có selected method để nhận “novelty certificate”. Bảng sau giải thích tại sao diagnostic chưa phải một bản tái triển khai và quy định ranh giới không được vượt.

| Closed family | Hypothesis/supervision; representation/negatives | Local mechanism/objective/inference; expected failure | Difference of diagnostic, not method novelty |
|---|---|---|---|
| ELSC | Lexical support teacher; adapter/word negatives | Support margin, CLCL deploy; local signal không thành rank gain | Không word bank/support margin; chỉ đo raw channel bằng caption loss |
| DIVE | Contradictory real pairs; local RGB/pose/reference | Quartet/local residual rerank; dependency/evidence risk | Không rival-conditioned support/reference subtraction/SEDS |
| PLEL | Near-pair phrases; local head/real alternatives | Bag evidence residual; coverage/validity risk | Không phrase bank hoặc inference evidence residual |
| OCEM | Shared-time overuse; RF atoms/assignment | Capacity/null OT; concentration proxy failed | Không transport, overlap masses, dustbin, physical support accounting |
| SSSC | Changed spans on shared support; reference miner | Training hinge, standard deploy; attribution/RNG risk | Không changed-span margin/miner; raw/frozen is an experimental factor |
| PMGR | Population mismatch; full current groups | Exact grouped-risk/replay; under-gate gain | Evaluator correctness thôi, không new population/rank loss |
| RPCA | Context needed without forgetting; decoder/update projection | AR context+gradient protection; unproved benefit | Không sentence generation, distillation, optimizer surgery |

Nếu diagnostic được đổi thành CiCo+alignment, UPRet+uncertainty, SEDS+fusion, C²RL+context hoặc SAN+miner, nó vi phạm registry. Positive diagnostic chỉ cho phép đặt **RQ mới**, không tự mở lại closed family.

# 19. Novelty Analysis Against Published Work

Không claim “first”. A–D va vào existing multilingual, hand-model, acquisition và generative retrieval mechanisms; second cycle còn loại density-ratio slogan. Có thể tồn tại một substantive new mechanism trong tương lai, nhưng tài liệu này chưa xác định được.

SignSeek là dictionary retrieval; DualAnchor/Selective Contrastive là SLT; AVIOT là video-LLM compression; SignGPT/VLTK là corpus/recognition toolkit. CCC báo MAX/MIN R1 trên training trajectory trong generation study, không locked dev-selected sentence benchmark. GTRN dùng visual signing query tìm video documents, không text↔video. Chúng được task-screened bằng primary records trong search log, không nhập số vào Track A. Phạm vi đọc các adjacent papers được giới hạn ở phần cần xác định task/collision, không claim full-paper review.

# 20. Repository-Level Implementation Plan

**Đã implement diagnostic, chưa implement new method.** Các script của audit nằm trong `docs/proposal7/tools/`; không sửa rejected-method code. Nếu thực hiện information pilot sau này, dùng namespace `methods/information_probe/` cho diagnostic, không giả là tên method mới.

| Proposed file, chưa tạo | Interface/reuse được kiểm tra | Gate |
|---|---|---|
| `methods/information_probe/src/information_probe/data.py` | `CiCoFeatureDataset` + explicit raw manifest; `[B,F,1024]`, `[B,F]` | All IDs/shapes/hashes; fail missing assets |
| `.../model.py` | `CiCoBridge`, `combine(...)`; no method-specific imports | Zero-change parity, finite gradients, param count |
| `.../train.py` | Shared factory/tokenizer; independent namespace/process | Seed/RNG/exposure equality; dev-only selector |
| `.../evaluate.py` | Shared score contract; isolated grouped metric implementation validated against fixtures | Full rectangle, no intersection; named tie policy |
| `methods/information_probe/configs/ph_information_probe.json` | Explicit resource/mask/group/freeze/optimizer hashes | Refuse unresolved raw extractor provenance |
| `methods/information_probe/tests/test_contracts.py` | Reuse fixture ideas, not rejected objectives | ID permutation, grouped-vs-flat, ties, leakage, gradient controls |

Exact reusable locations: `shared/slr_common/upstream/{factory,cico_bridge}.py`, `shared/slr_common/data/{cico_dataset,manifest,tokenize}.py`, `shared/slr_common/evaluation/cico_eval.py`. PMGR metric source may serve as read-only reference, not runtime method dependency. Checkpoint schema phải chứa config, source pin, ordered manifest hashes, resource hashes, optimizer/RNG states, update count, selected epoch/metric; evaluation refuses mismatch. Đây là file-level conditional plan, không claim files đã tồn tại.

# 21. Experiment Matrix

| Arm | Purpose | Status |
|---|---|---|
| B0 | Corrected/reproduced baseline với locked common protocol | Existing artifacts audited; full new fair reproduction chưa chạy |
| B1 | B0 + equal parameter head chỉ dùng existing features | Planned control |
| B2 | Equal wall-time/FLOPs/update exposure, additional uniform processing không new signal | Planned control; cần đo budget |
| B3 | Closest permitted published mechanism trong cùng input/prior regime | Chưa chọn vì không có surviving M; không thay bằng prohibited SEDS assets |
| M | Một surviving novel method | **Không tồn tại/NO-GO**, không điền kết quả |
| D-raw / D-frozen / D-shuffled | Crossed information diagnostic, same head/text/loss | Planned, không phải M |

PH exploration trước; H2/CSL replication chỉ sau khi gallery/assets/provenance đầy đủ. Không transfer PH hyperparameter theo test. Matched controls không được dùng ít epochs hoặc weaker text prior hơn D-raw.

# 22. Ablation Studies

Nếu pilot được triển khai: raw channel absent; raw frozen/trainable crossed; same features with same head; equal-resolution/full-frame versus fixed crops; train-only shuffled measurement; frozen baseline/new-head-only. Đánh giá toàn gallery và persistent-error slice đã khóa. Permutation labels không dùng để đánh giá real correspondence; đó là negative control.

Không ablate support teacher, null OT hay generation head vì chúng không thuộc phép thử. Không lập exhaustive hyperparameter grid để cứu hypothesis thất bại. Một diagnostic mới phải có budget và criterion trước khi nhìn outcomes.

# 23. Strong Controls and Alternative Explanations

Alternatives bắt buộc: extra parameters, extra encoder prior, more updates, differing augmentation/RNG, changed mask/group/ties, fewer candidates, test-informed selection, duplicated captions, source-template effects. Nếu một factor khác, ghi [U] attribution chứ không gọi causal gain.

Raw-versus-frozen không tự cô lập “information” nếu raw encoder có stronger pretrained semantics; cần same-pretraining full-frame control hoặc kết luận chỉ là resource effect. Shuffled raw can be distribution-shifting, nên nó không đủ một mình: cần B1/B2 và fixed acquisition controls. Expert sign assessment, nếu có sau này, cần planned sampling/blinding/consent; audit này không mô phỏng expert judgment.

# 24. Low-Cost Pilot and Kill Criteria

**Pilot ưu tiên: crossed raw-information test, chưa chạy.** Không đầu tư full training trước khi asset/protocol gates qua.

1. Gate P0, không training: đủ full dev IDs, permitted extractor provenance, baseline score parity, deterministic input ordering, no test selector. Fail bất kỳ mục nào → dừng pilot, sửa protocol/resource; không drop video.
2. Gate P1: chọn train subset tối đa 2,048 samples bằng train-only hash/source grouping; full dev 519 queries/gallery không giảm. Dùng seeds 42/1337/2026, cùng tối đa 260 updates/arm; cap tổng pilot 24 GPU-hours sau khi timing warm-up xác minh khả thi. Không coi 24 h là measured runtime.
3. Success threshold được đề xuất trước run: mean bidirectional dev R1 vượt strongest matched control **≥0.5 pp ở ≥2/3 seeds**; paired source-cluster bootstrap CI lower bound >0 trên seed-averaged query deltas; mỗi direction không giảm >0.25 pp, R5/R10 không giảm >0.5 pp. Target persistent slice phải cải thiện cùng chiều, không chỉ loss giảm.
4. Kill nếu raw không hơn equal-capacity/compute control, only-shuffle cũng tăng tương tự, initialization được chọn ở ≥2 seeds, raw branch không có attributable effect, hoặc chỉ target slice tăng mà overall gate fail. Không tăng budget/tune threshold hậu nghiệm.

Các ngưỡng là research decision rules, **không phải predicted gains hoặc post-hoc significance claim**. PH dev đã được khảo sát nhiều lần nên ngay cả pass cũng exploratory; cần independent train-derived held-out confirmation hoặc replication dataset, không dùng test để chọn tiếp.

# 25. Full Evaluation Protocol

Chỉ áp dụng khi có surviving method mới và full asset parity. Train/dev/test manifests khóa riêng theo version; positive maps và candidate order hashes publish cùng config. Test mở một lần cho frozen configuration mỗi seed; không chọn seed/checkpoint qua test. Mọi arm có cùng gallery, mask/group/ties, features/backbone/text prior và training exposure hoặc được chuyển Track C.

Báo R1/R5/R10/MedR/MnR hai chiều, mean/std across ≥3 seeds, paired differences, source/group-cluster bootstrap 10,000 draws với fixed gallery. Với CSL group text repeated, bootstrap theo sentence group đồng thời giữ performances; với H2 theo source/recording hierarchy đã xác minh; không mặc định IID. Confidence intervals conditional on gallery không thay seed variance; n=3 seeds giới hạn inference.

Không dừng report ở favorable metric/dataset. Test regressions, exclusions, failed seeds và invalid runs phải công bố cùng lý do. Nếu thiếu H2 assets thì ghi “not evaluated on complete protocol”, không claim three-dataset SOTA.

# 26. Compute and Resource Estimate

[M, historical run] PH B0 200 epochs/2,600 updates: 2,667.64 s, peak allocated 43,171,500,032 bytes (~40.21 GiB), 149,724,163 trainable parameters, bf16. Đây là measured baseline configuration, không phải runtime cho raw branch hoặc paper setup. GPU được thấy là RTX 5880 Ada ~49 GB; không giả định luôn idle/available.

[I] Tuyến tính thô 260 updates ~267 s/arm nếu giữ chính xác workload cũ; raw decoding/encoding có thể làm sai ước lượng nhiều lần. Vì vậy phải đo 20 warm-up +100 timed steps, allocated/reserved VRAM, throughput, preprocessing/disk, latency cold/warm-cache trước phân bổ full pilot. Lưu cả failed OOM. 24 GPU-hours là cap quyết định, không chứng minh đủ tài nguyên.

Saved score matrix PH fp32 ~1.03 MiB; full token interaction lớn hơn nhiều, phải chunk. H2 2,348² scores ~21 MiB nếu square, nhưng actual query/group count phải lấy từ locked manifest. Không đoán memory chỉ từ score matrix; encoder activations chiếm chính trong training. No SEDS resource budget.

# 27. Failure Modes and Risks

Rủi ro lớn nhất: tiếp tục thiết kế trên correlation, dev overfitting sau nhiều proposal, unknown checkpoint history, raw asset provenance, inaccurate sign-semantic labels, và overclaim từ single-model review. H2 thiếu asset có thể cản full evaluation nhưng không chứng minh hypothesis sai. Public upstream bugs cần bounded attribution.

Responsible use: sentence retrieval không nên dùng để nhận diện signer, giám sát người Điếc hay đưa quyết định hệ trọng không có kiểm tra con người. Không publish identifiable video/pose ngoài giấy phép. Không coi sign language là word-for-word spoken language hoặc thiếu hụt cần “sửa”. Human annotation/participant recruitment tương lai phải có institutional ethics/IRB determination, consent, access/retention plan; audit không tự cấp exemption. Dataset license/secondary-use permissions phải được xác minh trước redistribution/new training campaign.

Funding/COI đọc được chỉ dùng để disclose: CMCM declares no competing interests; các paper có public/industry funding khác nhau. Funding không tự chứng minh bias. Không claim đã kiểm tra retractions. AI có xu hướng ghép module và tạo novelty narrative; registry/controls được dùng để chống xu hướng này.

# 28. Conditions Required for a SOTA Claim

Chưa điều kiện nào cho phép gọi Proposal 7 là SOTA. Claim tương lai cần: surviving materially new mechanism; reproduced strongest **comparable** baseline; exact full-gallery protocol; permitted identical resource regime; dev-only freeze; multi-seed gain vượt controls; không serious regression hai chiều/ba datasets; code/config/checkpoint/evaluation hashes tái chạy được.

Overall-SOTA còn cần cập nhật literature và giải quyết material inaccessible results như CMCM, không chỉ vượt CiCo. Nếu dùng stronger pretraining, chỉ claim theo Track C và kiểm soát resource attribution. Pass pilot không tự đạt SOTA; absence of located competitor không chứng minh first/best.

# 29. Reproducibility Checklist

Đã có: complete historical-document reading ledger; targeted code audit; negative registry; 39-query source log; official repo pins; hashed dataset reconciliations; saved-dev rank/checkpoint validation; source-cluster bootstrap; RQs trước candidates; four distinct candidates/four-perspective critiques; second-cycle diagnostics/redesign; no SEDS assets; no source dataset/rejected method edits.

Đã chạy: `audit_workspace.py`, `reconcile_how2sign.py`, `diagnose_saved_ph_dev.py`, `audit_new_upstreams.py`, `diagnose_information_channels.py`. Evidence trong `docs/proposal7/evidence/`. Final combined protocol/reconciliation/information/package suite **21 passed**; Ruff passed. Các suite 17-test và 11-test trước đó overlap, không cộng tổng. New test collection ban đầu lỗi local-module import, đã sửa dual package/script import trong audit helper và rerun pass. [Mechanical verification](evidence/report_verification.json) ghi report hash, đủ 30 sections, 14 citation/anchor pairs, local links và diagnostic input hashes. Passing checks không chứng nhận scientific novelty.

Chưa có và không giả vờ có: new method implementation, raw-information pilot outcomes, full three-dataset independent reproduction, complete official H2 assets, verified C²RL public code, exhaustive citation graph, retraction checks, human linguistic review. Đây là giới hạn NO-GO report, không ô “passed”.

Reproduce diagnostics, không training:

```bash
/home/haipd/miniconda3/bin/python docs/proposal7/tools/diagnose_information_channels.py
/home/haipd/miniconda3/bin/python -m pytest -q docs/proposal7/tools/test_information_channels.py docs/proposal7/tools/test_reconciliation.py docs/proposal7/tools/test_protocol_boundaries.py
/home/haipd/miniconda3/bin/python -m ruff check docs/proposal7/tools
```

# 30. Final Recommendation: GO / CONDITIONAL GO / NO-GO

**NO-GO cho việc chọn/triển khai một new SOTA method từ các ứng viên đã khảo sát.** Hai vòng đã không tìm được candidate có novelty và causal evidence đủ mạnh dưới constraints. Không đổi tên hướng thất bại để thỏa mãn hình thức proposal.

**Highest-value unresolved experiment:** crossed raw-information versus frozen-feature/equal-compute probe ở §24. Nó có thể bác bỏ giả thuyết “thiếu visual information” trước khi đầu tư vào architecture mới. Kết quả hiện tại không khẳng định probe sẽ thành công và không mở lại proposal1–6. Nếu cần tiếp tục, ưu tiên phép thử này sau P0, không thêm một loss/reranker.

# References

Danh mục primary sources được dùng trong các bảng/collision arguments, với version cụ thể; metadata/coverage và các task-screened leads bổ sung ở search log. URL citation ngay tại claim là authoritative locator.

- Duarte, A., Albanie, S., Giró-i-Nieto, X., & Varol, G. (2022). *Sign Language Video Retrieval with Free-Form Textual Queries*. CVPR; arXiv v2 dùng trong audit.
- Cheng, Y., Wei, F., Bao, J., Chen, D., & Zhang, W. (2023). *CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning*. CVPR; author-hosted preprint.
- Wu, X., Li, H., Luo, Y., Cheng, X., Zhuang, X., Cao, M., & Fu, K. (2024). *Uncertainty-aware Sign Language Video Retrieval with Probability Distribution Modeling*. ECCV; arXiv v1.
- Jiang, L., Wang, M., Li, Z., Fang, Y., Zhou, W., & Li, H. (2024). *SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval*. ACM MM. DOI 10.1145/3664647.3681237.
- Chen, Z., Zhou, B., Huang, Y., Wan, J., Hu, Y., Shi, H., Liang, Y., Lei, Z., & Zhang, D. (2024). *C²RL: Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieval*. arXiv:2408.09949v1. TCSVT 2025 final-version parity chưa xác minh.
- Lee, J., Hur, C., Choi, C., Cho, S., Gaim, F., Hwang, E. J., Song, H., & Lim, K. (2026). *Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative Mining for Sign Language Retrieval*. ACL, 28262–28277.
- Yang, X.-H., Wei, D., Li, W., & Hu, H. (2026). *Causality-inspired multi-grained cross-modal sign language retrieval*. CVIU, 264, 104631.
- Zhou, W., Zhao, W., Hu, H., Li, Z., & Li, H. (2024). *Scaling up Multimodal Pre-training for Sign Language Understanding*. arXiv:2408.08544v1.
- Lei, J., Berg, T. L., & Bansal, M. (2021). *mTVR: Multilingual Moment Retrieval in Videos*. ACL.
- Hu, H., Zhou, W., & Li, H. (2021). *Hand-Model-Aware Sign Language Recognition*. AAAI.
- Hu, Z., Ye, N., & Mohomed, I. (2022). *mmSampler: Efficient Frame Sampler for Multimodal Video Retrieval*. MLSys, 4.
- Jin, P., Li, H., Cheng, Z., Li, K., Ji, X., Liu, C., Yuan, L., & Chen, J. (2023). *DiffusionRet: Generative Text-Video Retrieval with Diffusion Model*. ICCV.
- Ren, J., Liu, P. J., Fertig, E., Snoek, J., Poplin, R., DePristo, M. A., Dillon, J. V., & Lakshminarayanan, B. (2019). *Likelihood Ratios for Out-of-Distribution Detection*. NeurIPS, 32.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. arXiv:1807.03748; v2 (2019) dùng cho §2.2–2.3.
