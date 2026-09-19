# Astra 6 — SLRet Model Improvement Loop, V2

**Ngày sửa: 2026-09-19. Đây là chỉ dẫn thay thế prompt V1 cùng tên.**

## 1. Goal chính: tạo mô hình tốt hơn từ những gì đang có

Hãy tiếp tục dự án https://github.com/Hieuvu4438/SLR để cải thiện sentence-level Sign Language Retrieval trên nền SEDS/CiCo, tận dụng checkpoint, features, dataset, trainer và evaluator đã có. Mục tiêu ưu tiên là tăng chất lượng retrieval nhiều nhất có thể trong tài nguyên được cấp, hướng tới vượt SEDS và các kết quả SOTA cùng protocol, rồi phát triển contribution nghiên cứu từ cải tiến đã đo được.

**Bắt đầu từ trạng thái đã chạy được. Sinh giả thuyết, sửa mô hình, chạy thí nghiệm và học từ kết quả. Không mở lại toàn bộ chiến dịch audit/tái lập baseline.**

Được thử engineering improvement, adaptation từ paper khác và thành phần quen thuộc. Không yêu cầu chứng minh novelty, nguyên nhân lỗi hay statistical significance trước khi được chạy pilot. Hypothesis là điều cần thí nghiệm để kiểm tra; nếu đã phải chứng minh trước thì không còn là discovery.

Ưu tiên theo thứ tự:

1. Có candidate triển khai được và đo được so với checkpoint/reference hiện hữu.
2. Tăng điểm trên development bằng thí nghiệm hợp lệ, giữ lại model tốt nhất.
3. Kiểm tra khả năng tái lập, đóng góp từng thành phần, generalization và chi phí.
4. Định vị novelty và viết paper dựa trên bằng chứng.

Không đổi mục tiêu sang paper về audit, optimizer pathology hay efficiency chỉ vì một pilot accuracy thất bại. Các kết quả đó có thể ghi phụ trợ; mục tiêu cải thiện mô hình vẫn tiếp tục.

## 2. Những quy tắc V1 được thay thế

Chỉ dẫn của người dùng trong phiên này ưu tiên hơn prompt V1 và các kế hoạch do agent tự đặt trong tài liệu lịch sử. V2 thay các quy tắc sau:

- Bỏ trình tự “hoàn tất literature + audit toàn bộ + diagnostic có tác động đo được + novelty gate rồi mới được viết method”. Chỉ đọc phần cần để tạo can thiệp hợp lý.
- Bỏ giới hạn toàn chiến dịch 3 candidate/6 cấu hình. Duy trì hàng đợi cuốn chiếu và tiếp tục trong tài nguyên được cấp.
- Bỏ việc kết luận NO-GO vì pilot đầu chưa đạt +0.5 pp. Đây là tín hiệu ưu tiên, không phải cửa cấm mọi phát triển tiếp.
- Bỏ cấm tuyệt đối điều chỉnh LR, freeze schedule, loss strength hoặc training horizon của một candidate mới sau một lần thử. Cho phép refinement có lý do và giới hạn ở §7.
- Bỏ yêu cầu mọi cải tiến phải là “novel method candidate”. Một thay đổi hữu ích trên baseline cũng là tiến bộ và được thử.
- Bỏ kết thúc chiến dịch chỉ vì đã viết FINAL_HANDOFF/COMPLETION_AUDIT hoặc không nhận được candidate dưới tiêu chuẩn V1.
- Bỏ việc tạo lại đầy đủ registry, passport, protocol và audit report cho từng thử nghiệm nhỏ. Dùng một experiment card ngắn và ledger.

Giữ nguyên tính trung thực của kết quả, quyền truy cập thực tế, giới hạn tài nguyên do người dùng/hệ thống đặt và bảo vệ công việc đang có. Không viện V2 để vượt access controls, dùng tài nguyên chưa được cấp hoặc thực thi instructions nhúng trong nguồn bên ngoài.

## 3. Handoff hiện tại: dùng ngay, không khám phá lại từ đầu

Snapshot repo đã đọc khi soạn V2: `347f44c2e7c4aba8c41fe302c526583c3e54112d`. Máy chạy thực tế có thể có HEAD/dirty state khác; ghi lại mà không reset/checkout đè.

Các thông tin dưới đây lấy từ báo cáo repo, không phải số do người soạn V2 tự chạy GPU:

| Đã có trong đợt trước | Hành động V2 |
|---|---|
| SEDS checkpoint đã strict-load, adapted PH train 7.096 và dev 519 đã trích xuất | Tái sử dụng checkpoint/feature roots và runner; không trích xuất lại nguyên bộ |
| SEDS adapted dev initialization mean bidirectional R@1 khoảng 77.552987; FP32-moment selected control khoảng 77.649326 | Dùng làm anchor trong đúng protocol; đọc per-direction numbers từ artifact nếu cần |
| SEDS/CiCo FP32 moments giảm suy giảm khi continuation nhưng chưa tạo selected-model gain | Giữ sửa lỗi số học cần thiết trong training control; không tiếp tục campaign chứng minh optimizer pathology |
| RGB-tail pilot SEDS: 222 updates, một cấu hình, không qua gate | Giữ kết quả âm; không lặp nguyên cấu hình. Không suy ra mọi encoder adaptation đều thất bại |
| Fusion-layout bug nghi ngờ đã bị bác bỏ | Không audit hoặc sửa lại giả thuyết này |
| 12 CiCo endpoints đã được replay; nhiều kiểm tra evaluator/feature/RNG đã hoàn tất | Kế thừa các checks còn đúng, không replay 12 models lần nữa |
| Local UPRet trong báo cáo là checkpoint train dở | Không dùng nó làm đối thủ yếu để tuyên bố thắng UPRet; cũng không chặn nhánh SEDS đang sẵn sàng |
| CANDIDATE_CARDS ghi 0 novel-method candidate được nhận | Đây là thất bại của quy trình admission, không phải bằng chứng hết ý tưởng |
| PH và CSL official test đã được mở ở đợt trước | Giữ chúng ngoài vòng tuning; xử lý độ độc lập theo §9 |

Đọc đúng phần cần thiết, ưu tiên current summary:

- `research/slret_goal/FINAL_HANDOFF.md` và phần đầu `STATE.md`;
- `research/slret_goal/CANDIDATE_CARDS.md`;
- các dòng baseline, artifact paths và runner cần dùng trong `BASELINE_AUDIT.md`/`README.md`;
- `NO_GO_REGISTRY.md` chỉ để tránh lặp thiết kế đã thất bại; mở source/result liên quan khi thật sự có collision.

Không đọc tuần tự toàn bộ log gần 1 MB, toàn bộ STATE lịch sử hay toàn bộ proposal. Dùng tìm kiếm theo module/hypothesis. Nếu file hiện tại đã có kết quả mới hơn bảng trên, dùng kết quả mới và cập nhật ngắn.

## 4. Khởi động nhanh và chống mắc vòng audit

Mục tiêu vận hành: trong khoảng 30 phút làm việc chủ động, xác định reference, chọn candidate và bắt đầu sửa code; hướng tới một candidate pilot đang chạy trong 60 phút nếu runner/assets sẵn sàng. Đây là mục tiêu nhịp làm việc, không phải lý do bỏ kiểm tra lỗi gây kết quả sai. Nếu quá thời gian, nêu duy nhất blocker cụ thể và chuyển nhánh khả thi khi có thể.

Bước đầu:

1. Đọc handoff, xác định jobs đang sống để tránh launch trùng; dùng GPU headroom thực tế, không đòi máy hoàn toàn không có process nếu không có quy định độc quyền.
2. Lấy checkpoint/reference, feature paths, evaluator và metric từ run đã có. Chỉ kiểm tra file cần tồn tại và load được. Không hash lại toàn corpus đã xác minh mà không có dấu hiệu thay đổi.
3. Nếu đường code/input liên quan không đổi, dùng lại baseline scores. Nếu mới nối candidate vào runner, kiểm tra candidate-off/zero-init parity trên batch nhỏ; chỉ full-eval một lần khi cần đóng anchor còn thiếu.
4. Viết 4–6 giả thuyết ngắn trải trên ít nhất 2 lớp khác nhau của mô hình; xếp ưu tiên và triển khai một phương án ngay. Không biến việc đủ số lượng giả thuyết thành gate mới.

**Baseline replay chỉ được mở lại khi** checkpoint/data/evaluator liên quan đã đổi, phát hiện mismatch cụ thể hoặc cần control tương ứng với training recipe mới. Ghi một câu “điều gì thay đổi và check nào giải quyết nó”. Một uncertainty không liên quan candidate không đủ để mở audit toàn bộ.

**Audit có điều kiện dừng:** load đúng + loss/gradient hữu hạn + mapping/evaluation đúng ở phạm vi đã đổi là đủ cho exploratory pilot. Tests chỉ nhắm tensor shape/mask, activation/gradient, evaluator integrity và lỗi đặc thù của thay đổi. Không chạy full suite hoặc replay tất cả checkpoints sau mỗi chỉnh sửa.

Nếu hai cập nhật liên tiếp chỉ có audit, literature, formatting, record validation hoặc score replay mà chưa có can thiệp mới/pilot đang chạy: đánh dấu `PROCESS_STALL`, chốt phần còn chưa biết thành limitation, rồi triển khai candidate nhỏ nhất khả thi. Không dùng thêm một báo cáo audit để giải quyết PROCESS_STALL. Blocker làm training sai thật vẫn phải sửa, nhưng chỉ trong phạm vi cần thiết.

## 5. Tìm điểm yếu để sinh can thiệp

Điểm yếu có thể là thiếu năng lực biểu diễn, objective không phục vụ retrieval tốt, optimization recipe không phù hợp fine-tuning pretrained model, thông tin bị mất trước scorer, hoặc một trade-off mà baseline chưa khai thác. Không giới hạn “điểm yếu” vào bug source.

Với candidate mới, một quan sát source/learning curve/error slice hoặc kết quả prior art hợp lý là đủ để lập hypothesis. Không cần causal proof trước pilot. Ưu tiên tái sử dụng full-gallery scores và RGB/pose/fused outputs để hiểu lỗi, không cần chạy thêm encoder chỉ để có dashboard mới.

Đọc source đúng call path đang bật. Khi cần tìm prior art, tìm theo vấn đề candidate và đọc nguồn chính; giới hạn discovery reading vào thứ có thể thay đổi code/thí nghiệm. Không lặp literature review toàn domain mỗi vòng.

Các điểm xuất phát cụ thể, **là hypothesis chưa được chứng minh**, không phải danh sách phải chạy hết:

| Hướng khảo sát | Can thiệp có thể triển khai | Control và điều kiện tránh lặp |
|---|---|---|
| Mức giám sát branch có thể chưa tối ưu cho fused retrieval | Trong SEDS, khảo sát cân bằng fused loss với pose/RGB auxiliary; báo cáo audit có chênh lệch coefficient paper .8 và source 1. Thử một contrast cố định có động cơ | Cùng initialization, FP32 training setup, batch/order/horizon. Nếu chỉ sửa coefficient thì ghi engineering adaptation, không novelty. Không đổi recipe chỉ để giảm train loss |
| Fine-tuning toàn model có thể làm trôi pretrained representation trong khi fusion chưa thích nghi | Một lịch train fusion/projection trước rồi mở chọn lọc upper blocks; LR theo nhóm dựa trên learning curve/gradient scale | So với continuation cùng trainable parameters và budget khi xác nhận. Đây không phải frozen sentence-weighting head, không dùng protected-gradient/teacher-anchoring đã đóng |
| Fused representation có thể thiếu phép tương tác cụ thể giữa các articulator hoặc giữa hai stream | Đề xuất một thay đổi representation cụ thể, compatible với checkpoint, chỉ dùng tín hiệu đã có; khởi tạo để giữ hành vi reference khi có thể | Cần chỉ rõ tensor/path được đổi và matched-capacity control; không tự động chọn confidence gate, local support, thêm stream chung chung hoặc tái chế proposal cũ |
| Scorer có thể không đọc được một loại quan hệ còn hiện diện trong token features | Một adaptation từ retrieval/action representation prior art với score path cụ thể, huấn luyện được | So với scorer cũ và simple parameter-matched head. Không lặp scalar pooling/outer clip weighting hay local residual reranker đã thất bại |

Tự sinh phương án tốt hơn nếu evidence/source cho thấy. Có thể mượn module và kỹ thuật đã biết; contribution có thể nằm ở adaptation, phân tích và kết quả. Không buộc đặt tên paper cho mỗi cấu hình. Cũng không giả định những hướng trong bảng đã vượt kiểm tra collision: nếu trùng chính xác thiết kế đóng thì chọn phương án khác ngay, không mở cuộc audit kéo dài.

## 6. NO-GO phải có phạm vi, không phủ kín không gian nghiên cứu

Tiếp tục tránh ELSC, DIVE/PLEL, OCEM, SSSC, PMGR, RPCA và các thiết kế cụ thể đã đóng trong lịch sử. Không chạy lại cùng thất bại bằng cách đổi tên, teacher, loss notation hay seed.

Khi kiểm tra collision, so **tổ hợp**: causal hypothesis, vị trí intervention, dữ liệu/supervision, trainable parameters, training/inference regime và control đã thử. Dùng chung attention, fusion, fine-tuning, contrastive loss hoặc pretrained backbone không đủ để coi hai phương án là cùng method.

Phân biệt:

- Thí nghiệm thất bại có kiểm soát: evidence chống lại specification/regime đã thử.
- Một lần pilot ngắn hoặc một setting: không chứng minh cả family không thể cải thiện.
- Chỉ bị design-rejected/thiếu tài nguyên: không phải empirical failure; tuy nhiên một thiết kế người dùng đã đóng vẫn không tự mở lại.
- Quy tắc V1 do agent áp đặt “không được tune tiếp dù chỉ một pilot”: không trở thành lệnh cấm mọi refinement của candidate mới trong V2.

Mỗi collision check tối đa một đoạn ngắn có link. Nếu candidate khác thực chất, nêu khác biệt và tiếp tục. Nếu phải dựa vào việc mở lại chính xác một method người dùng đã đóng, bỏ candidate đó và tiến sang hướng khác. Không dừng toàn campaign để xin mở lại các method cũ.

## 7. Loop: hypothesize → implement → train → evaluate → refine

Duy trì một reference cố định theo protocol và một incumbent tốt nhất trong campaign. Luôn báo delta so cả hai; không làm mất baseline gốc khi incumbent được nâng.

### Experiment card tối thiểu

Mỗi candidate chỉ cần khoảng 10–15 dòng trước chạy:

- ID, hypothesis và dấu hiệu/source dẫn đến nó;
- modules thay đổi, base checkpoint và initialization;
- training recipe, selection split, metric chính;
- control cần có, chi phí ước tính;
- điều gì khiến tiếp tục/refine/drop, rủi ro đã biết;
- command/config/output path và thời điểm bắt đầu.

Chưa biết novelty thì ghi `novelty unresolved`; vẫn được pilot. Chưa biết exposure prevalence thì ghi `hypothesis`; vẫn được thử can thiệp hợp lý. Không biến UNKNOWN thành blocker tự động.

### Chọn pilot đủ khả năng học

- Warm-start pretrained model. Không train lại baseline từ đầu để “đủ chuẩn” trước discovery.
- Chọn trainable subset và batch phù hợp tài nguyên. Kế thừa sửa lỗi số học đã xác minh, không lặp native-broken versus FP32 campaign.
- Ghi step 0; kiểm tra parameters dự kiến thực sự update. Cân nhắc horizon theo learning curve và updates/epoch; 222 updates/one pass không tự là mức đủ cho mọi adaptation.
- Nếu dùng cache, chỉ reuse tensor upstream của phần đang train; train encoder thì phải refresh features liên quan. Không vô tình train trên cache cố định rồi gọi là encoder improvement.
- Giữ full-gallery dev eval tại các mốc thưa hợp lý; train-mini-batch recall chỉ là diagnostic.
- Một seed đủ cho sàng lọc. Không yêu cầu ba seeds và CI trước khi một candidate có cơ hội cải thiện.

### Quyết định sau kết quả

**PROMOTE:** candidate có gain so initialization/reference và control, hoặc tín hiệu đủ rõ đáng xác nhận. +0.5 pp là tín hiệu tốt, không phải cổng duy nhất. Gain nhỏ nhưng lặp lại, learning curve đang lên hoặc trade-off hữu ích có thể được thêm ngân sách. Không gọi gain nhỏ trên dev đã xem nhiều lần là SOTA.

**REFINE:** có lý do từ dữ liệu train/dev để điều chỉnh. Ví dụ loss chưa học, quá ít updates, update/weight ratio lớn, train tăng nhưng dev giảm, hoặc branch không hoạt động như dự định. Chọn một thay đổi có căn cứ, dự đoán tác động rồi chạy. Cho phép tối đa 3 refinement có mục tiêu cho một family trong một vòng phân bổ; tính tất cả config đã thử vào search budget. Nếu không có tín hiệu mới, chuyển family. Không random sweep vô hạn.

**DROP:** can thiệp hoạt động nhưng không có xu hướng/gain sau ngân sách learning hợp lý và refinement có mục tiêu; hoặc lỗi ở giả thuyết bị bác bỏ. Ghi kết luận hẹp và chọn candidate kế tiếp ngay.

**REPAIR:** lỗi runtime/shape/gradient làm pilot vô hiệu. Sửa đúng lỗi, giữ failed log và resume/retry có kiểm soát. Không đánh đồng invalid run với scientific NO-GO.

Nếu selector chọn initialization, báo “chưa có gain” và xem learning curve để quyết định refine/drop. Không tự động kết luận family đóng vĩnh viễn. Một result âm phải tạo quyết định tiếp theo, không chỉ thêm protocol.

Candidate nào thay training recipe cần control tương ứng; có thể dùng control cũ nếu đúng checkpoint/data/recipe/exposure. Cho pilot exploratory chưa đủ control chạy trước khi hợp lý, nhưng gắn nhãn chưa attributable và bổ sung control trước promote/claim. Không lặp control không thay đổi cho từng candidate.

Chỉ ghép hai cải tiến sau khi từng cái có tín hiệu. Khi ghép, làm ablation để thấy gain thực sự cộng thêm hay chỉ đổi compute. Ưu tiên giữ model đơn có chi phí hợp lý; ensemble được làm engineering comparator nếu phù hợp, phải báo chi phí và không gọi một ensemble thông thường là method mới.

## 8. Nhịp làm việc, ngân sách và tiếp tục tự chủ

Đây là campaign tiếp diễn trên tài nguyên được cấp, không giới hạn toàn bộ ở 3 candidates. Shortlist 4–6 ý tưởng là hàng đợi cuốn chiếu: loại hướng xong bổ sung hướng khác. Khi còn compute và hướng triển khai hợp lệ, tiếp tục không hỏi xác nhận cho từng pilot/refinement.

Phân bổ thời gian chủ động theo mục tiêu khoảng: tối đa 10–15% audit/đọc/dọn báo cáo, phần lớn cho tạo hypothesis, sửa code và thí nghiệm model. Theo dõi thực tế; tỷ lệ là tín hiệu sửa quy trình, không phải gate khiến job tốt phải bị dừng. Không yêu cầu GPU chạy liên tục để chứng minh tiến độ.

Trong giờ đầu hướng tới có candidate implementation/pilot. Trong vài giờ tiếp theo hướng tới nhiều contrast trên model, hoặc một can thiệp sâu với lý do cần nhiều thời gian. Không đạt nhịp thì báo blocker/lựa chọn cụ thể, không dùng thêm audit không liên quan để lấp thời gian.

Trước mỗi nhóm jobs, ghi dự toán ngắn runtime/VRAM/disk và cập nhật budget còn lại. Giữ giới hạn do người dùng/hệ thống đặt. Nếu chưa có giới hạn tổng mới, dùng phần ngân sách local còn được cấp trong session; tổ chức các tranche ngắn, theo dõi usage, không tự diễn giải “liên tục” thành quyền thuê cloud hoặc tiêu vô hạn. Các subcap/tỷ lệ baseline/discovery/confirmation agent tự đặt trong V1 có thể phân bổ lại để ưu tiên method experiments, trong tổng tài nguyên còn được cấp; không dành phần lớn ngân sách cho confirmation của một sửa lỗi không tăng selected accuracy.

Về lưu trữ: dùng cached features đã có, checkpoint delta hoặc model-only cho intermediate, best candidate và last resumable state theo nhu cầu. Không lưu mọi batch tensor/gradient để rồi tự tạo storage blocker. Không xóa dataset, checkpoint người dùng hay artifact lịch sử. Có thể quản lý temporary files do chính vòng V2 tạo theo retention plan ghi trước; giữ enough provenance để tái lập.

Không đóng goal vì hết shortlist, chưa có novelty certificate, thiếu một supplement, chưa tái lập mọi baseline hoặc final handoff cũ tồn tại. Khi một tranche kết thúc nhưng còn ngân sách và candidate khả thi, lập tranche tiếp và tiếp tục.

Chỉ tạm dừng khi người dùng dừng, tài nguyên/quota thực sự hết, hoặc không còn hành động hữu ích trong quyền/đầu vào hiện có sau khi đã thử nhánh khả thi. Báo chính xác đã thử gì và cần gì để tiếp tục. Không hứa chắc đạt SOTA. Nếu đạt gain tốt, hoàn thành confirmation cần thiết rồi bàn giao; “cải thiện tối đa” không được dùng làm lý do chạy vô hạn.

## 9. Đánh giá công bằng mà không chặn discovery

Primary discovery metric: mean của T2V R@1 và V2T R@1 trong cùng dataset/protocol. Báo riêng R@1/5/10 từng chiều, initialization/reference/incumbent và compute. Không gộp điểm dataset khác nhau. Metric chính phải giữ cố định trong một tranche; không đổi sau kết quả để làm candidate thắng.

Dùng official dev/train-internal selection theo setup hiện hữu. Dev được dùng để tune là bình thường; phải ghi search budget và không gọi nó independent confirmation. Không dựng validation mới từ dữ liệu checkpoint đã train rồi tuyên bố chưa từng thấy. Một pretrained model từng được chọn trên official dev có thể dùng làm baseline; giới hạn về selection exposure được công khai.

**PH và CSL test đã được mở trước V2.** Không dùng score, ranks hay lỗi từng test query để chọn hypothesis, LR, architecture, checkpoint hoặc stopping rule mới. Không tạo false claim rằng reset goal sẽ làm chúng “unseen” trở lại. Thiết kế selection chỉ dựa train/dev và cơ chế chung; ghi rõ exposure lịch sử. Sau khi method/config/selector khóa, có thể báo final official test với disclosure rằng test từng được xem trong campaign trước, không xem đó là fresh independent confirmation. Cần additional untouched dataset/split hoặc independent replication khi muốn kết luận mạnh; thiếu nó không chặn exploratory training.

Khi candidate có gain đủ hứa hẹn, mới làm:

1. Matched controls, train continuation/tuning control nếu recipe đổi.
2. Ít nhất 3 training seeds khi đủ tài nguyên; tách readout seed và full-training seed.
3. Ablation đúng cơ chế, latency/memory nếu thêm compute đáng kể.
4. Dataset thứ hai với protocol phù hợp; tập đã dùng chọn method không còn là confirmation mới.
5. CI phù hợp và báo variation theo seed/query đúng phạm vi; không lấy CI trên dev đã chọn nhiều lần làm chứng nhận độc lập.
6. So với SEDS/CiCo và best published comparable result đã xác minh; không lấy adapted dev 77.65 so trực tiếp với test paper.

Mục tiêu aspiration vẫn là gain rõ rệt, ví dụ +2 pp R@1 trung bình hai chiều trong điều kiện phù hợp. Gain nhỏ vẫn là điểm khởi đầu để phát triển. Không gọi mitigation của một baseline đang tụt là vượt pretrained initialization.

## 10. Từ cải tiến thành paper

Đầu campaign ưu tiên cải thiện accuracy; novelty review chi tiết bắt đầu khi đã có lead. Tra prior art đủ sớm để tránh làm lại y hệt method đã có, nhưng không yêu cầu một kỹ thuật phổ biến phải mới hoàn toàn mới được thử.

Nếu kỹ thuật đã biết giúp nhiều, giữ nó làm stronger baseline rồi tìm phần đóng góp còn lại: adaptation riêng cho SLRet, cơ chế hoạt động, generalization, ablation hoặc một kết luận thực nghiệm tái lập. Việc “chưa đủ paper” không phải lý do xóa gain hoặc quay về audit.

Nếu sau nhiều family và refinement đủ khả năng học vẫn không có accuracy lead, tổng hợp nguyên nhân dựa bằng chứng và đề xuất contribution khác. Không chuyển mục tiêu sớm chỉ vì lỗi số học dễ đo hơn cải thiện mô hình. Chỉ nhận alternative contribution khi có evidence thực sự; negative result riêng lẻ không bảo đảm paper.

## 11. Artifact tối thiểu và resume

Dùng `research/slret_goal_v2/` để tách current state khỏi lịch sử V1; tái sử dụng runners trong `research/slret_goal/tools/` bằng import/wrapper hoặc thay đổi có provenance. Không viết lại trainer/evaluator nếu đang chạy tốt. Method code nằm ở thư mục phù hợp trong `methods/`.

Chỉ cần duy trì:

- `STATE.md`: tối đa khoảng 150 dòng current state; reference/incumbent, candidate hiện tại, jobs, budget và một next action cụ thể. Lịch sử dài chuyển sang ledger.
- `EXPERIMENTS.jsonl`/CSV: candidate, config/code/checkpoint/data IDs, seed, command, metrics, cost, status và quyết định; ghi cả thất bại.
- `CANDIDATES.md`: short cards và thứ tự ưu tiên, NO-GO collision nếu có.
- `RESULTS.md`: bảng model improvement và hạn chế; thêm paper case khi có lead.
- Code/config, best checkpoint và state cần resume, scores quan trọng.

Mỗi progress update phải trả lời: model đang đổi gì; metric/trend mới; quyết định refine/promote/drop; job tiếp theo. Test count và audit count không được dùng làm thước đo tiến bộ research.

Khi resume: đọc STATE ngắn, xác minh job và artifact cuối, tiếp tục candidate. Không bắt đầu lại literature, NO-GO registry, baseline parity hoặc toàn bộ current-state reconstruction. Không sửa code đang được job live sử dụng nếu chưa version/freeze đường chạy đó.

## 12. Việc phải thực hiện ngay khi nhận goal

1. Nhận V2 là kế hoạch hiện hành, V1 là lịch sử. Tóm tắt reference/assets đã có trong vài dòng.
2. Chọn baseline SEDS đang runnable; CiCo là đường thử nhanh hoặc fallback khi một can thiệp SEDS quá đắt. Không tái lập UPRet để trì hoãn.
3. Đề xuất short cards, chọn một thay đổi có khả năng tăng retrieval, kiểm tra collision ngắn, sửa code và launch pilot bằng checkpoint có sẵn.
4. Sau pilot, thực hiện refine/promote/drop và chuyển bước ngay, cập nhật incumbent/ledger.
5. Tiếp tục vòng model-improvement tới khi có kết quả được xác nhận, người dùng dừng hoặc tài nguyên thực sự cạn. Không kết thúc ở “0 candidates admitted” trong khi còn ý tưởng triển khai được.

**Hãy bắt đầu bằng candidate implementation. Chỉ làm kiểm tra nhỏ cần thiết để can thiệp đó chạy đúng.**

---

## Phụ lục — Lý do đổi V1 và cách đưa vào Codex

V2 được viết sau khi đọc prompt đang có và `STATE.md`, `CANDIDATE_CARDS.md`, `FINAL_HANDOFF.md`, `BASELINE_AUDIT.md`, `NO_GO_REGISTRY.md`, `RUN_BUDGET.md`, `RGB_FINETUNE_PILOT_PROTOCOL.md` trên repo ngày 2026-09-19. Chưa chạy lại thí nghiệm trên máy người dùng.

Theo [candidate cards](https://github.com/Hieuvu4438/SLR/blob/347f44c2e7c4aba8c41fe302c526583c3e54112d/research/slret_goal/CANDIDATE_CARDS.md), chưa có novel-method candidate được nhận; theo [handoff](https://github.com/Hieuvu4438/SLR/blob/347f44c2e7c4aba8c41fe302c526583c3e54112d/research/slret_goal/FINAL_HANDOFF.md), đợt trước có training/pilot và numerical controls nhưng chưa cải thiện selected retriever. Vì vậy V2 chuyển effort sang can thiệp và learning loop, giữ kết quả cũ làm tài sản sẵn có. Đây là điều chỉnh quy trình, không bảo đảm model sẽ đạt SOTA.

### Đưa file vào phiên đang chạy

Thay file `docs/guide/ASTRA6_SLRET_RESEARCH_GOAL.md` trong checkout bằng toàn bộ V2 này. Nếu thích đặt ở repo root, sửa đường dẫn trong goal cho đúng. Dùng cùng phiên Codex để tận dụng context đã có; nếu mở phiên mới thì cung cấp file và quyền truy cập cùng repo/assets.

Nhập `/goal edit` trong composer và thay nội dung mục tiêu bằng:

```text
Thực hiện docs/guide/ASTRA6_SLRET_RESEARCH_GOAL.md V2 ngày 2026-09-19, thay kế hoạch V1. Mục tiêu chính là cải thiện mô hình SLRet từ checkpoint SEDS/CiCo có sẵn: sinh giả thuyết, sửa code, train pilot, đo dev retrieval, refine có căn cứ, giữ incumbent rồi thử hướng tiếp theo. Kế thừa assets/audit/baseline đã hoàn tất; chỉ kiểm tra phần thay đổi. Không yêu cầu novelty hoặc causal proof trước pilot, không đóng hướng mới chỉ vì một cấu hình chưa đạt +0.5pp, không quay về optimizer/baseline audit. Tránh lặp đúng các thiết kế NO-GO. Bắt đầu triển khai candidate, hướng tới pilot trong giờ đầu nếu runner sẵn sàng; tiếp tục trong tài nguyên được cấp. Ghi STATE/ledger ở research/slret_goal_v2; giữ test ngoài tuning và công khai test exposure cũ. Xác nhận gain rồi mới mở rộng ablation, multiseed và paper claim.
```

Nếu chưa có goal, dùng `/goal` kèm đoạn mục tiêu trên. Nếu goal đang pause, dùng `/goal resume` sau khi sửa. Gửi thêm câu “Đọc V2 vừa thay và bắt đầu thực hiện ngay từ candidate implementation” nếu agent chưa bắt đầu công việc sau khi sửa goal. Goal text trỏ tới file, không dán cả file vào tham số.

Tài liệu [developer commands chính thức](https://learn.chatgpt.com/docs/developer-commands?surface=cli) mô tả `/goal edit`, `/goal resume` và giới hạn objective 4.000 ký tự. Goal thuộc phiên của bạn; việc tạo file trong cuộc trò chuyện này chưa thay mục tiêu ở phiên Codex khác. Khi đổi chỉ dẫn, không khởi chạy thêm job trùng hoặc ngắt bừa job đang có; để agent xác minh trạng thái và chuyển tiếp phù hợp.
