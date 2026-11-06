# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU (Tesla T4, `Device: cuda`)

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (cả ba `PASS` checksum SHA-256)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (không đổi)

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**. Record không có `bbox_xyxy` hay `polygon_xy`, chỉ có một nhãn kèm `image_width`=640 và `image_height`=428 của toàn ảnh. Ảnh `traffic` có nhiều xe buýt, ô tô và người, nhưng model chỉ trả về một danh sách top-5 (cab, minibus, police_van, recreational_vehicle, streetcar) cho cả khung hình. Nó không chỉ ra chiếc "cab" nằm ở đâu.
3. Danh sách lớp **đi kèm taxonomy/checkpoint**. `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K nên chỉ chọn được trong 1000 lớp đó, model không tự nghĩ ra tên lớp. Vì vậy ảnh đường phố COCO nhận top-1 là "cab" (taxi), dù cảnh chủ yếu là xe buýt.
4. Cùng một tên có thể mang ID khác hoặc nghĩa khác giữa các taxonomy. Ví dụ ImageNet có `cab`/`minibus`, còn COCO-80 chỉ có `car`/`bus`, và trong file detection thì `bus` có `class_id`=5. Ghi đủ `class_id` + `class_name` + `taxonomy_name` thì nhãn mới tra ngược được, không bị nhầm khi gộp dữ liệu hoặc đổi model, và máy đọc theo ID còn người đọc theo tên.
5. Guideline cần nói rõ: (a) bài toán là single-label hay multi-label; (b) quy tắc chọn lớp khi có nhiều vật, ví dụ chọn vật chiếm diện tích lớn nhất, vật ở trung tâm, hoặc chủ đề chính của cảnh; (c) thứ tự ưu tiên giữa các lớp gần nghĩa (cab / minibus / bus); (d) khi nào thì đánh "không rõ" và escalation thay vì đoán.
6. `score` = 0.51 chỉ là độ tin của model, trong khi top-2 đến top-5 cộng lại vẫn chiếm khoảng 0.35 xác suất. Model có thể sai và đang dùng taxonomy không khớp với ảnh. Ground truth phải do người gán theo guideline và qua QC. Nếu lấy prediction làm nhãn thì khi huấn luyện lại, model sẽ học chính lỗi của nó.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Ảnh rộng 640 px, cao 427 px, gốc (0, 0) ở góc trên-trái. Góc trên-trái của box nằm ở x = 385 px (hơi lệch phải giữa ảnh), y = 69 px (gần mép trên, ngang đầu người đứng). Góc dưới-phải ở x = 499 px, y = 349 px (ngang chân/mép dưới tạp dề). Box rộng khoảng 114 px, cao khoảng 280 px, tức là một hình chữ nhật đứng ôm người đang quay lưng ở giữa-phải ảnh.
3. Ngưỡng 0.20 → **17** prediction; ngưỡng 0.60 → **6** prediction (`person, bowl, bowl, oven, oven, person`). Ngưỡng 0.35 dùng cho JSON → 11 prediction.
4. Khi hạ ngưỡng, reviewer phải xem thêm nhiều box score thấp (ở 0.35 đã có bowl 0.38, cup 0.38). Reviewer phải loại false positive, xử lý box chồng nhau (vd. bowl 0.38 [0.32, 305.02, 87.69, 371.32] chồng một phần lên bowl 0.72 [32.65, 342.12, 100.16, 384.93], IoU ≈ 0.23: cần xem ảnh để quyết định là hai bát thật hay một bát bị đoán hai lần), sửa class sai và kiểm tra box có ôm sát vật không. Khối lượng QC tăng gần gấp ba (6 → 17 box).
5. **Quy tắc box:** "Vẽ một box cho mỗi vật thể thuộc taxonomy, ôm sát **phần nhìn thấy được** của vật (không chừa nền, không cắt mất phần thấy được). Không đoán phần bị che hay phần nằm ngoài khung. Vật bị cắt mép ảnh thì box chạm mép ảnh và gắn cờ `truncated`. Vật bị che hơn khoảng 50% vẫn vẽ box nếu còn nhận ra được lớp, và gắn cờ `occluded`."
6. Vật bị cắt mép: `person` score 0.610931, box [0.12, 263.18, 61.55, 310.78] (61×48 px). Box bắt đầu từ x = 0.12, tức là chạm mép trái, và chỉ thấy một phần cơ thể/tay. Mình xử lý ca này bằng **guideline**, vì vật bị cắt mép xuất hiện ở rất nhiều ảnh và quy tắc ở câu 5 (box ôm phần thấy được + cờ `truncated`) áp dụng nhất quán được. Mình chỉ escalation nếu phần nhìn thấy quá ít, đến mức không chắc đó có phải người hay không.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]]
2. Box chỉ cho biết khung chữ nhật [385.45, 66.44, 498.02, 348.58], gồm cả nền (tường, nồi) giữa hai tay và quanh chân. Polygon 348 điểm bám theo đường viền thật: đầu, vai, tay, khe giữa hai chân. Nhờ đó biết được pixel nào thuộc người, diện tích thật và hình dạng của vật, và tách được người khỏi vật đứng sát bên.
3. `instance_id` (vd. `kitchen-001`) chỉ để phân biệt từng đối tượng trong output của ảnh này. Ví dụ `kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010` đều là `bowl` (cùng `class_id`) nhưng là 4 cái bát riêng. Nó **không** phải class ID, vì class do `class_id`/`class_name` cho biết. Nó cũng **không** phải tracking ID, vì nó chỉ là số thứ tự trong một lần chạy trên một ảnh. Chạy lại hoặc sang khung hình khác thì cùng một người có thể mang số khác.
4. **Quy tắc vẽ biên mask:** "Đường biên đi theo mép nhìn thấy được của vật, sai số không quá khoảng 2 px. Không gộp nền hay vật khác vào mask. Phần bị vật khác che thì để trống, không vẽ đoán. Hai vật chạm nhau thì mỗi pixel chỉ thuộc **một** instance, vật nằm trên/phía trước được giữ trọn mask. Lỗ hổng bên trong vật (khe tay, quai) không tô vào mask."
5. Hai vật chạm nhau: `kitchen-005` dining table (598 điểm, bbox [0.97, 234.74, 345.86, 423.87]) và các bát đặt trên bàn. Box của `kitchen-002` và `kitchen-003` nằm gọn trong box bàn, còn `kitchen-006` gần như gọn, chỉ lệch khoảng 1 px ở mép trái. Polygon bàn thì chừa chỗ cho bát: tâm của cả ba bát đều nằm ngoài polygon bàn. Riêng `kitchen-006` có 14/84 điểm biên lọt vào trong polygon bàn, nghĩa là hai mask chồng nhau một phần. Đây đúng là chỗ biên bàn và bát chạm nhau, bàn bị bát che. Mình xử lý bằng **guideline** (quy tắc "mỗi pixel thuộc một instance, vật phía trên được giữ trọn" ở câu 4), vì cảnh đồ vật đặt trên bàn rất phổ biến và cần xử lý thống nhất. Riêng các nồi chảo treo trên tường thì COCO-80 không có lớp "pan", model không ra mask nào cho chúng. Ca đó cần **escalation** để Lab Coach quyết định taxonomy, annotator không tự đặt lớp.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một (hoặc vài, nếu multi-label) `class_id` + `class_name` + `taxonomy_name` cho **cả ảnh** | `traffic` top-1 = `cab` 0.51, trong khi cảnh có nhiều xe buýt (detection thấy `bus` 0.91). Taxonomy ImageNet không có lớp kiểu "cảnh giao thông", nên các lớp gần nghĩa cab/minibus/bus dễ gây tranh cãi | Xem cả ảnh, chọn lớp theo quy tắc chủ đề chính trong guideline, escalation nếu không lớp nào khớp | Lớp có đúng taxonomy và đúng quy tắc chọn không, mức đồng thuận giữa các annotator, các ảnh bị gắn nhầm lớp gần nghĩa |
| Phát hiện vật thể | Mỗi vật một dòng: `class_id`, `class_name`, `bbox_xyxy` pixel (+ cờ truncated/occluded) | `kitchen`: 2 box `bowl` chồng nhau một phần (IoU ≈ 0.23; [0.32, 305.02, 87.69, 371.32] và [32.65, 342.12, 100.16, 384.93]), chưa rõ là hai bát hay trùng; `person` 0.61 bị cắt mép trái chỉ thấy một phần; nồi chảo treo tường không được phát hiện (không có lớp trong COCO-80); số box đổi từ 17 → 6 khi tăng ngưỡng 0.20 → 0.60 | Vẽ box ôm sát phần thấy được cho **mọi** vật thuộc taxonomy (không bỏ vì nhỏ hay mờ), gắn cờ truncated/occluded | Vật bị bỏ sót, box thừa hoặc trùng, box lỏng/chặt quá, class sai, cờ truncated/occluded |
| Instance segmentation | Mỗi instance: `instance_id` riêng + `class_id` + polygon/mask pixel; hai vật cùng lớp là hai instance | `kitchen-005` dining table chạm các bowl đặt trên bàn: polygon bàn chừa chỗ cho bát nhưng vẫn lấn 14/84 điểm biên của `kitchen-006`; seg tìm thêm `potted plant`, `spoon`, `dining table` mà detection không có, nhưng lại thiếu `oven` bên phải và `cup` (detection có `oven` 0.63, `cup` 0.45), nên hai model không nhất quán | Vẽ polygon theo biên thấy được, tách instance chạm nhau, không tô phần bị che, mỗi pixel thuộc một instance | Biên mask có bám sát không, mask có lấn sang vật khác không, có gộp nhầm hai instance thành một không, instance bị sót |

## 5. An toàn dữ liệu

1. Mình chỉ dùng ba ảnh COCO công khai mà notebook đã cố định bằng checksum; không tải ảnh cá nhân hay dữ liệu nội bộ lên Colab, và không ghi họ tên, MSSV hay thông tin cá nhân vào báo cáo/output.
2. Mình dừng ngay, không xử lý tiếp và không tự thay ảnh, rồi báo cho **Lab Coach / mentor** phụ trách lớp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
