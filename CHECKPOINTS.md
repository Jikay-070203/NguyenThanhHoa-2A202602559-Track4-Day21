pip i# Checkpoints

Bài lab làm **cá nhân**. Mỗi checkpoint ghi rõ 4 thứ: **cần làm gì**, **sản phẩm phải có khi kết thúc**, **cần hiểu gì** và **cách tự kiểm tra**. Lab coach sẽ đi vòng quanh lớp ở cuối mỗi checkpoint. Nếu bạn chưa qua được một checkpoint, hãy báo lab coach ngay, đừng im lặng làm tiếp.

> **Nguyên tắc chung:** commit sau **mỗi** checkpoint với message dạng `CPx: <mô tả ngắn việc vừa làm>`, ví dụ `CP3: yaw sweep 0-3 deg`. Lịch sử commit là bằng chứng bạn tự làm bài theo tiến độ. Bài chỉ có 1–2 commit cuối giờ sẽ bị gọi vấn đáp để xác minh (xem `RUBRIC.md` mục 4).

---

## CP0 — Chuẩn bị (làm ở nhà, trước buổi học)

**Cần làm**
1. Fork repo đề bài, đặt tên theo cú pháp `<HoVaTen>-<MSSV>-Track4-Day21`, ví dụ `NguyenVanA-20240123-Track4-Day21`, rồi clone về máy, theo đúng `README.md` mục 3, bước 1.
2. Làm bước 1 **ở nhà**. Repo nặng khoảng 135 MB vì có sẵn dữ liệu, nên không clone trong giờ lab.
3. Tạo môi trường Python và cài thư viện theo mục 3, bước 2 của `README.md`.
4. Kiểm tra dữ liệu bằng hai lệnh dưới đây. Cả hai phải in ra dòng `[PASS]`.
   - `python tools/verify_data.py --data-root data/kitti_mini`
   - `python tools/verify_data.py --data-root data/nuscenes_mini_subset`
5. Nếu định làm topic B, hoặc topic C có dùng model: cài MMDetection3D hoặc OpenPCDet trên máy có GPU NVIDIA và chạy được demo của thư viện trên 1 frame.

**Sản phẩm**
- Bản fork `<HoVaTen>-<MSSV>-Track4-Day21` đã có trên GitHub và đã clone về máy.
- Dòng thông tin học viên ở đầu `report/REPORT.md` đã điền: họ tên, MSSV, lớp, link repo.

**Cần hiểu**
- **Định dạng KITTI:**
  - File `velodyne/*.bin` là mảng float32, mỗi điểm 4 số `x, y, z, reflectance`.
  - File `calib` có 3 ma trận cần dùng: `P2` (3x4), `R0_rect` (3x3), `Tr_velo_to_cam` (3x4).
  - Box 3D trong `label_2` nằm trong **camera frame**, không nằm trong LiDAR frame.

**Tự kiểm tra**
```bash
python -m starter.data_health --data-root data/synthetic
```
- [ ] Lệnh in ra 5 dòng, từ `000000` đến `000004`, và tạo file `results/data_health.csv`.
- [ ] (Chỉ topic B/C có model) Demo của thư viện detector chạy được trên 1 frame và lưu được ảnh kết quả.

---

## CP1 — Chọn topic và dataset (0:00 – 0:15)

**Cần làm**
1. Đọc [TOPICS.md](TOPICS.md) và chọn 1 topic. Ghi chữ cái topic (A–F) vào đầu `report/REPORT.md`.
2. Chọn dataset (`data/synthetic`, `data/kitti_mini`, `data/nuscenes_mini_subset`) và liệt kê cụ thể các frame sẽ dùng.
3. Viết **claim giả thuyết** vào mục 1 của `report/REPORT.md`. Claim là một câu khẳng định mà bạn sẽ dùng số liệu để chứng minh hoặc bác bỏ. Bạn được sửa claim ở các checkpoint sau.

**Sản phẩm**
- Mục 1 của `report/REPORT.md` có claim nháp.

**Cần hiểu**
- **Claim tốt phải kiểm chứng được bằng số.**
  - Claim yếu: *"Calibration rất quan trọng."* Câu này không đo được, nên không chứng minh được.
  - Claim tốt: *"Lệch yaw 1° làm hơn 10% điểm LiDAR rơi ra ngoài 2D box của xe ở khoảng cách 30 m."* Câu này có đại lượng đo được (% điểm), có điều kiện (1°, 30 m) và có ngưỡng (10%).
- **Kiểm tra dữ liệu đi trước model.** Mở `results/data_health.csv` từ CP0 và xem có cột nào có giá trị bất thường so với các frame còn lại không.

**Tự kiểm tra**
- [ ] Claim có đủ 3 thành phần: đại lượng đo được, điều kiện cụ thể, và ngưỡng hoặc phép so sánh.
- [ ] Bạn nói được trong một câu: mình sẽ đo đại lượng gì, trên frame nào, với các mức thay đổi nào.
- [ ] Đã commit `CP1: choose topic X, draft claim` (thay X bằng chữ cái topic).

---

## CP2 — Viết TODO và chạy demo đầu tiên (0:15 – 0:50)

**Cần làm**
1. **Mọi topic trừ B** (topic D bỏ qua nếu không dùng camera): viết code cho 2 hàm có đánh dấu `TODO(CP2)` trong `starter/projection.py`, gồm `velo_to_cam` và `cam_to_image`. Docstring của từng hàm đã mô tả các bước cần làm.
2. Chạy được demo đầu tiên của topic:

   | Topic | Demo đầu tiên cần có |
   |---|---|
   | A, F | Ảnh điểm LiDAR chiếu lên camera, có vẽ 2D box của label |
   | B | Kết quả inference của model, có vẽ 3D box hoặc BEV |
   | C | Ảnh hoặc số liệu của dữ liệu gốc, đặt cạnh dữ liệu sau 1 mức perturb |
   | D | Point cloud sau downsample và sau khi tách mặt đất |
   | E | Dashboard đầu tiên có ít nhất 2 biểu đồ |

3. Đặt toàn bộ code bạn tự viết trong thư mục `src/`. **Không sửa** các file khác trong `starter/`, ngoài 2 hàm TODO.

**Sản phẩm**
- Ảnh demo đầu tiên trong `results/figures/`.
- Lệnh tạo ra ảnh đó, ghi trong mục 5 của `report/REPORT.md`.

**Cần hiểu**
- **Chuỗi biến đổi `P2 · R0_rect · Tr_velo_to_cam · [x y z 1]ᵀ`:**
  - Phải chia cho `s` (thành phần thứ 3) để đổi từ toạ độ đồng nhất ra pixel `(u, v)`.
  - Phải bỏ các điểm có `z_cam ≤ 0`, tức nằm sau camera. Nếu không bỏ, chúng bị chiếu ngược lên ảnh ở vị trí sai.
- **Hai hệ trục khác nhau:**
  - LiDAR KITTI: x hướng về phía trước, y sang trái, z lên trên.
  - Camera: x sang phải, y xuống dưới, z hướng về phía trước.
  - nuScenes còn khác thêm một bước nữa (xem `data/README.md` mục 3.2).

**Tự kiểm tra**
```bash
python -m starter.projection --data-root data/synthetic --frame 000000
python -m starter.projection --data-root data/kitti_mini --frame 000011
```
- [ ] Trong ảnh tạo ra, điểm LiDAR **nằm khớp** lên xe, người, cột và mặt đường. Không có điểm nào nằm trên bầu trời hay bị lộn ngược.
- [ ] Test bằng tay với calib của `data/synthetic` frame `000000`: điểm LiDAR `(10, 0, 0)` phải cho `z_cam ≈ 9.73` (số dương, nằm trước camera) và pixel `(u, v) ≈ (614, 175)`, tức gần giữa ảnh rộng 1242 pixel.
- [ ] Code không bị lỗi khi dữ liệu có điểm NaN.
- [ ] Đã commit `CP2: baseline demo running`.

---

## CP3 — Chạy thí nghiệm chính (0:50 – 1:25)

**Cần làm**
1. Chạy thí nghiệm chính của topic với **ít nhất 3 mức** thay đổi, ví dụ yaw 0°, 1°, 2°, 3° hoặc giữ lại 90%, 70%, 50% số điểm.
2. Cố định `seed` cho mọi phép ngẫu nhiên, và ghi lại cấu hình của từng lần chạy.
3. Ghi kết quả ra file CSV trong `results/`, rồi vẽ ít nhất 1 biểu đồ hoặc bảng.

**Sản phẩm**
- `results/<tên_thí_nghiệm>.csv`, mỗi dòng là một cấu hình, mỗi cột là một metric. Ví dụ `results/yaw_perturb_sweep.csv`.
- `results/figures/<tên_biểu_đồ>.png`.

**Cần hiểu**
- **So sánh công bằng:** mỗi lần chỉ thay đổi **một yếu tố**. Mọi thứ khác giữ nguyên: frame, vùng range, danh sách class và seed.
- **Nếu có đo thời gian chạy (latency):**
  - Bỏ lần chạy đầu tiên, vì lần đầu thường chậm hơn do khởi tạo.
  - Chạy lặp lại ít nhất 20 lần, rồi báo trung vị (p50) và phân vị 95 (p95). Không báo con số của một lần đo duy nhất.

**Tự kiểm tra**
- [ ] Chạy lại script lần thứ hai cho ra **đúng cùng số liệu**.
- [ ] Nhìn bảng, bạn nói được xu hướng: metric tăng hay giảm theo mức thay đổi, và có điểm nào metric đột ngột xấu đi không.
- [ ] Đã commit `CP3: benchmark results`.

---

## CP4 — Phân tích failure case (1:25 – 1:45)

**Cần làm**
1. Tìm **ít nhất 1 trường hợp** phương pháp hoặc model cho kết quả sai. Lưu ảnh minh hoạ với tên bắt đầu bằng `fail_`, ví dụ `results/figures/fail_01_yaw_2deg_pole.png`.
2. Giải thích vì sao sai, và chỉ ra lỗi thuộc lớp nào trong 6 lớp debug (slide 30):

| Lớp | Câu hỏi cần trả lời |
|---|---|
| I/O (đọc dữ liệu) | Đã đọc đúng các trường dữ liệu, đúng offset, đúng frame_id và timestamp chưa? |
| Geometry (hình học) | Phép biến đổi hệ toạ độ và quy ước 3D box đã đúng chưa? |
| Time (thời gian) | LiDAR, camera và ego pose đã được đồng bộ thời gian chưa? Đã bù chuyển động (deskew) chưa? |
| Preprocess (tiền xử lý) | Vùng range, kích thước voxel và danh sách class có khớp với cấu hình đánh giá không? |
| Model | Checkpoint có đúng dataset và đúng class không? Chạy lại hai lần có ra cùng kết quả không? |
| Metric (cách đo) | Metric và cách chia nhóm dữ liệu có phản ánh đúng mục đích sử dụng không? |

**Sản phẩm**
- Mục 3 của `report/REPORT.md` có ảnh failure, nguyên nhân, và lớp debug tương ứng.

**Cần hiểu**
- Tìm ra failure case **không bị trừ điểm**. Ngược lại, phân tích failure tốt chiếm 25/100 điểm.

**Tự kiểm tra**
- [ ] Một bạn khác trong lớp nhìn ảnh failure có hiểu ngay "sai ở đâu" mà không cần bạn giải thích không?
- [ ] Bạn có đề xuất được cách phát hiện, hoặc cách khắc phục, lỗi này khi chạy trên xe hoặc robot thật không?
- [ ] Đã commit `CP4: failure analysis`.

---

## CP5 — Hoàn thiện báo cáo và nộp bài (1:45 – 2:00)

**Cần làm**
1. Điền đủ 6 mục của `report/REPORT.md` và xoá hết các dấu `[ĐIỀN]`.
2. Chạy kiểm tra hình thức, sửa hết lỗi, rồi commit và push.

**Sản phẩm**
- Repo đầy đủ theo [SUBMISSION.md](SUBMISSION.md), đã push lên remote.

**Tự kiểm tra**
```bash
python tools/check_submission.py
```
- [ ] Mọi dòng đều là `[PASS]`, dòng cuối là `KẾT QUẢ: SẴN SÀNG NỘP`.
- [ ] `git status` báo không còn thay đổi chưa commit, và `git log origin/main -1` hiện đúng commit cuối cùng của bạn.

---

## CP6 — Demo trước lớp (2:00 – 2:20)

**Cách tổ chức**
- Lớp có nhiều học viên, nên giảng viên sẽ **gọi ngẫu nhiên khoảng 5 học viên** lên trình bày. Mỗi người có **3 phút trình bày và 1 phút hỏi đáp**.
- **Ai cũng phải chuẩn bị** như thể mình sẽ được gọi.
- Người không được gọi vẫn được chấm điểm Trình bày, dựa trên REPORT đã nộp (xem `RUBRIC.md` mục 1.5).

**Cần làm**
- Mở sẵn `report/REPORT.md` và ảnh hoặc video demo trên máy, để khi được gọi có thể trình bày ngay trong vòng 30 giây.

**Câu hỏi giảng viên có thể hỏi**
- Claim của bạn sẽ sai trong điều kiện nào?
- Nếu đổi sang dataset hoặc sensor khác (ví dụ từ KITTI 64 beam sang nuScenes 32 beam), kết quả có còn đúng không?
- Muốn đưa phương pháp này vào xe hoặc robot thật, bạn sẽ ghi log thêm chỉ số gì?

**Tự kiểm tra**
- [ ] Đã tập nói thử một lần, gọn trong 3 phút.
- [ ] Giải thích được mọi con số trong bảng kết quả của mình.
