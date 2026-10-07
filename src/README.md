# src/ — Topic A: LiDAR-camera projection QA

Toàn bộ code tự viết cho bài lab (ngoài 2 hàm `TODO(CP2)` trong `starter/projection.py`). Chỉ cần CPU. Chạy từ **gốc repo**.

| File | Việc | Sản phẩm |
|---|---|---|
| `calib_qa.py` | Thư viện: lệch calibration, hit rate / precision / % trong ảnh, alignment score | (không chạy trực tiếp) |
| `exp_demo.py` | CP2 + Basic: kiểm tra tay, overlay 3 khoảng cách, gallery yaw 0–3° | `results/cp2_sanity.csv`, `results/demo_overlay_stats.csv`, `results/figures/demo_*.png`, `drift_gallery_*.png` |
| `exp_sweep.py` | CP3 + Good: quét roll/pitch/yaw 0–3°, tx/ty/tz 0–20 cm trên KITTI + nuScenes | `results/calib_sweep_frames.csv`, `calib_sweep_summary.csv`, `analytic_shift.csv`, `hit_rate_*.png` |
| `exp_detect.py` | Advanced: ngưỡng phát hiện drift của alignment score (TPR, AUC, cửa sổ K frame) | `results/alignment_detection.csv`, `alignment_*.png` |
| `exp_failure.py` | CP4: 3 failure case chọn tự động (Geometry, Time, Metric) | `results/figures/fail_*.png`, `failure_cases.csv`, `time_sync_nuscenes.csv` |
| `exp_latency.py` | B3: latency p50/p95 (bỏ warm-up, 30 lần đo) | `results/latency_projection.csv` |
| `exp_compare.py` | **B1**: so sánh 2 thuật toán alignment score (Canny vs gradient) trên cùng dữ liệu/lưới drift/metric | `results/compare_methods*.csv`, `compare_methods_auc.png` |
| `exp_stress.py` | **B2**: stress test 5 loại suy giảm LiDAR x 3–4 mức, đo ảnh hưởng lên kiểm tra calibration | `results/stress_degradation*.csv`, `stress_degradation.png` |
| `exp_synthetic.py` | **B6**: tự động tìm lỗi cài sẵn trong `data/synthetic` (NaN, mất cung quét, hở timestamp) | `results/synthetic_faults.csv` (bảng 3 cột), `synthetic_*.csv`, `synthetic_*.png` |
| `make_report.py` | Sinh `report/REPORT.md` từ CSV (mọi số lấy từ CSV) | `report/REPORT.md` |
| `run_all.py` | Chạy tất cả theo thứ tự + kiểm tra tái lập (sha256) | `results/determinism_check.txt` |
| `kaggle_run.ipynb` | Notebook chạy trên Kaggle | |

```bash
python -m src.run_all --class-name "AI20K-xx"          # chạy tất cả
python -m src.run_all --max-frames 8 --max-nusc-frames 8 --skip-latency   # chạy thử nhanh
python -m src.exp_sweep --help                          # mọi script đều có --help
python -m src.exp_sweep --datasets kitti --max-frames 5 # kiểm tra nhanh calibration trên dữ liệu định dạng KITTI
```

## Định nghĩa các chỉ số

- **hit rate**: trong các điểm LiDAR nằm trong box 3D ground-truth của một object và nhìn thấy trong ảnh khi calibration đúng, % điểm sau khi lệch vẫn rơi vào 2D box của object đó.
- **precision**: trong các điểm rơi vào 2D box (sau khi lệch), % điểm thực sự thuộc box 3D của object đó.
- **% điểm trong ảnh**: % điểm LiDAR hữu hạn rơi trong khung ảnh.
- **alignment score**: điểm LiDAR "biên độ sâu" (sâu hơn hẳn láng giềng gần nhất trên ảnh) phải trùng cạnh ảnh (Canny). `score` = trung bình exp(−d/σ), d là khoảng cách tới cạnh gần nhất; `contrast` = `score` trừ điểm số của chính các điểm đó khi dịch ảnh ±3°.
- Mọi phép đo là xác định (không ngẫu nhiên); bootstrap trong `exp_detect.py` dùng seed 0.
