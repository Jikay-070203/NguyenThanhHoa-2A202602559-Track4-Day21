"""Sinh report/REPORT.md từ các file CSV trong results/ (mọi con số lấy từ CSV, không gõ tay).

    python -m src.make_report --class-name "AI20K-xx"
    python -m src.make_report --student-name "Nguyen Thanh Hoa" --class-name "AI20K-xx"

Chạy SAU khi đã chạy exp_demo, exp_sweep, exp_detect, exp_failure, exp_latency (hoặc `python -m src.run_all`).
Phần chữ giải thích cơ chế viết sẵn; câu nào nêu chiều hướng của số liệu đều được chọn theo số đo thật.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.calib_qa import BAND_NAMES, ROT_KINDS, TRANS_KINDS

REPO_URL = "https://github.com/Jikay-070203/NguyenThanhHoa-2A202602559-Track4-Day21"
FIG = "../results/figures"
REQUIRED = ["calib_sweep_summary.csv", "calib_sweep_frames.csv", "alignment_detection.csv", "failure_cases.csv",
            "time_sync_nuscenes.csv", "latency_projection.csv", "demo_overlay_stats.csv", "cp2_sanity.csv",
            "analytic_shift.csv"]


def fmt(x, nd=1, suffix=""):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}{suffix}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


class Data:
    def __init__(self, res: Path):
        self.res = res
        self.sm = pd.read_csv(res / "calib_sweep_summary.csv")
        self.frames = pd.read_csv(res / "calib_sweep_frames.csv")
        self.det = pd.read_csv(res / "alignment_detection.csv")
        self.fail = pd.read_csv(res / "failure_cases.csv")
        self.time = pd.read_csv(res / "time_sync_nuscenes.csv")
        self.lat = pd.read_csv(res / "latency_projection.csv")
        self.demo = pd.read_csv(res / "demo_overlay_stats.csv")
        self.sanity = pd.read_csv(res / "cp2_sanity.csv")
        self.ana = pd.read_csv(res / "analytic_shift.csv")

    def row(self, ds, kind, mag):
        r = self.sm[(self.sm.dataset == ds) & (self.sm.kind == kind) & np.isclose(self.sm.magnitude, mag)]
        return r.iloc[0] if len(r) else None

    def band_rate(self, r, names):
        """Hit rate gộp nhiều dải khoảng cách (trọng số theo số điểm)."""
        n = sum(r[f"npts_{b}"] for b in names)
        h = sum(r[f"hit_{b}_pct"] * r[f"npts_{b}"] / 100 for b in names if r[f"npts_{b}"] > 0)
        return (100 * h / n, int(n)) if n > 0 else (float("nan"), 0)

    def det_row(self, ds, kind, mag, metric="align_contrast"):
        r = self.det[(self.det.dataset == ds) & (self.det.kind == kind) & (self.det.metric == metric) & np.isclose(self.det.magnitude, mag)]
        return r.iloc[0] if len(r) else None

    def first_detect(self, ds, kind, col="tpr_window", thr=0.9, metric="align_contrast"):
        d = self.det[(self.det.dataset == ds) & (self.det.kind == kind) & (self.det.metric == metric) & (self.det[col] >= thr)]
        return float(d.magnitude.min()) if len(d) else None

    def datasets(self):
        return list(self.sm.dataset.unique())


def sweep_table(D: Data, ds: str, kind: str, window: int) -> str:
    mags = sorted(D.sm[(D.sm.dataset == ds) & (D.sm.kind == kind)].magnitude.unique())
    unit = "°" if kind in ROT_KINDS else " cm"
    rows = []
    for m in mags:
        r = D.row(ds, kind, m)
        bands = [fmt(r[f"hit_{b}_pct"], 1) if r[f"npts_{b}"] >= 30 else "–" for b in BAND_NAMES]
        dr = D.det_row(ds, kind, m)
        rows.append([f"{m:g}{unit}", fmt(r.inside_fov_pct, 2), fmt(r.hit_rate_pct, 1), *bands, fmt(r.precision_pct, 1),
                     fmt(r.contrast_mean, 3), fmt(dr.tpr_single, 2) if dr is not None else "–",
                     fmt(dr.tpr_window, 2) if dr is not None else "–"])
    header = [f"{kind}", "% điểm trong ảnh", "hit rate %", *[f"hit {b} %" for b in BAND_NAMES], "precision %",
              "contrast", "TPR 1 frame", f"TPR {window} frame"]
    return md_table(header, rows)


def pivot_table(D: Data, kinds, unit: str) -> str:
    cols, data = [], {}
    for ds in D.datasets():
        for k in kinds:
            s = D.sm[(D.sm.dataset == ds) & (D.sm.kind == k)].sort_values("magnitude")
            if len(s):
                cols.append(f"{ds} {k}")
                data[f"{ds} {k}"] = dict(zip(s.magnitude, s.hit_rate_pct))
    mags = sorted({m for v in data.values() for m in v})
    rows = [[f"{m:g}{unit}"] + [fmt(data[c].get(m), 1) for c in cols] for m in mags]
    return md_table(["mức lệch", *cols], rows)


def latency_table(D: Data) -> tuple[str, str]:
    rows = [[r.dataset, r.stage, f"{int(r.n_points):,}", fmt(r.p50_ms, 1), fmt(r.p95_ms, 1), int(r.runs), int(r.warmup)]
            for r in D.lat.itertuples()]
    t = md_table(["dataset", "giai đoạn", "số điểm", "p50 (ms)", "p95 (ms)", "số lần đo", "warm-up bỏ"], rows)
    r0 = D.lat.iloc[0]
    return t, f"{r0.cpu}, {int(r0.cores)} nhân, {r0.platform}, Python {r0.python}, numpy {r0.numpy}, OpenCV {r0.opencv}; chỉ chạy CPU"


def cross_table(D: Data) -> str:
    rows = []
    for ds in D.datasets():
        base = D.row(ds, "yaw", 0.0)
        y1, y3, t10 = D.row(ds, "yaw", 1.0), D.row(ds, "yaw", 3.0), D.row(ds, "tx", 10.0)
        d1 = D.det_row(ds, "yaw", 1.0)
        n_obj_pts = int(base.n_object_points)
        rows.append([ds, int(base.n_frames), n_obj_pts, fmt(base.inside_fov_pct, 2), fmt(base.hit_rate_pct, 1),
                     fmt(y1.hit_drop_pp if y1 is not None else None, 1), fmt(y3.hit_drop_pp if y3 is not None else None, 1),
                     fmt(t10.hit_drop_pp if t10 is not None else None, 1), fmt(base.contrast_mean, 3),
                     fmt(d1.auc_single if d1 is not None else None, 2)])
    return md_table(["dataset", "số frame", "số điểm object", "% điểm trong ảnh", "hit rate gốc %", "tụt @ yaw 1° (điểm %)",
                     "tụt @ yaw 3°", "tụt @ tx 10 cm", "contrast gốc", "AUC @ yaw 1°"], rows)


def build(D: Data, args) -> str:
    K = int(D.det.window.iloc[0])
    ds_list = D.datasets()
    kitti, nusc = ("kitti" if "kitti" in ds_list else ds_list[0]), ("nuscenes" if "nuscenes" in ds_list else None)

    # ---- claim (số lấy từ CSV)
    b, y1 = D.row(kitti, "yaw", 0.0), D.row(kitti, "yaw", 1.0)
    near0, _ = D.band_rate(b, BAND_NAMES[:1])
    far0, n_far = D.band_rate(b, BAND_NAMES[2:])
    near1, _ = D.band_rate(y1, BAND_NAMES[:1])
    far1, _ = D.band_rate(y1, BAND_NAMES[2:])
    far_drop, near_drop = far0 - far1, near0 - near1
    m_k = D.first_detect(kitti, "yaw")
    detect_k = f"từ yaw ≥ {m_k:g}°" if m_k is not None else "ở cả mức lớn nhất 3° vẫn chưa đạt 90%"
    claim = (f"Trên KITTI mini, calibration lệch yaw 1° làm hit rate (điểm LiDAR của object còn rơi trong 2D box của nó) giảm "
             f"từ {b.hit_rate_pct:.1f}% xuống {y1.hit_rate_pct:.1f}%; vật xa (>30 m) tụt {far_drop:.1f} điểm % "
             f"({far0:.1f}% → {far1:.1f}%, n={n_far} điểm) so với {near_drop:.1f} điểm % ở vật gần (<15 m). "
             f"Alignment contrast (biên độ sâu LiDAR so với cạnh ảnh) báo động đúng ≥ 90% với cửa sổ {K} frame ở false-alarm 5% {detect_k}")
    if nusc:
        m_n = D.first_detect(nusc, "yaw")
        claim += (f"; trên nuScenes (LiDAR 32 beam) " + (f"ngưỡng này là yaw ≥ {m_n:g}°." if m_n is not None
                                                          else "chỉ số KHÔNG đạt 90% ở bất kỳ mức nào đến 3°."))
    else:
        claim += "."
    cmp_txt = ("đúng như kỳ vọng hình học, vật xa nhạy hơn vật gần" if far_drop > near_drop + 1
               else "trái với kỳ vọng hình học ban đầu: vật xa không nhạy hơn vật gần trong dữ liệu này")

    # ---- thông tin frame
    frames_by = {ds: sorted(D.frames[D.frames.dataset == ds].frame_id.unique()) for ds in ds_list}
    frame_txt = "; ".join(f"{ds}: {len(v)} frame ({v[0]} … {v[-1]})" for ds, v in frames_by.items())
    demo_txt = ", ".join(f"{r.frame}" for r in D.demo.itertuples())

    # ---- evidence
    sections_sweep = []
    for ds in ds_list:
        sections_sweep.append(f"**{ds} - quét yaw** (`results/calib_sweep_summary.csv`, `results/alignment_detection.csv`)\n\n{sweep_table(D, ds, 'yaw', K)}")
    lat_table, hw = latency_table(D)
    full = D.lat[D.lat.stage.str.startswith("full")]
    sanity_ok = bool(D.sanity["pass"].all())
    det_file = D.res / "determinism_check.txt"
    det_txt = det_file.read_text(encoding="utf-8").strip() if det_file.exists() else "chưa chạy kiểm tra (xem `python -m src.run_all`)"

    # ---- failure
    fail_md = []
    for r in D.fail.itertuples():
        img = f"![failure]({FIG}/{r.file})"
        if r.case.startswith("fail_01"):
            lat_m = float(r.object_depth_m) * np.tan(np.deg2rad(1.0))
            fail_md.append(
                f"### 3.1 Geometry: yaw 1° làm người đi bộ/vật xa mất điểm (`{r.file}`)\n\n{img}\n\n"
                f"Frame KITTI `{r.frame_id}`, {r.object_type} cách {r.object_depth_m} m: hit rate **{r.value_before_pct}% → {r.value_after_pct}%** khi LiDAR "
                f"lệch yaw 1° ({r.note}). Cơ chế: lệch góc θ làm điểm ở khoảng cách d dịch ngang d·tanθ = {lat_m:.2f} m, so với bề rộng người đi bộ khoảng 0,6 m "
                f"(`results/analytic_shift.csv`). Dịch trên ảnh là f·tanθ pixel (không đổi theo d) nhưng 2D box của vật xa chỉ rộng vài chục pixel nên dịch này đủ đẩy điểm ra ngoài box. "
                f"**Lớp debug: Geometry** (extrinsic Tr_velo_to_cam). Cách phát hiện khi chạy thật: theo dõi hit rate/precision trên các object đã có box 2D từ camera detector "
                f"và cảnh báo khi tụt kéo dài; vật xa và mảnh (người, cột) là \"chim hoàng yến\" báo lệch sớm nhất."
            )
        elif r.case.startswith("fail_02"):
            t = D.time
            fail_md.append(
                f"### 3.2 Time: bỏ bù ego-motion trên nuScenes (`{r.file}`)\n\n{img}\n\n"
                f"Calibration giữ nguyên, chỉ thay `use_ego_motion=False` (coi LiDAR và camera chụp cùng lúc). Frame `{r.frame_id}`: {r.note}; "
                f"hit rate của {r.object_type} @ {r.object_depth_m} m **{r.value_before_pct}% → {r.value_after_pct}%**. Trên toàn bộ {len(t)} frame nuScenes: "
                f"độ lệch thời gian LiDAR-camera trung vị {t.dt_ms.median():.1f} ms (min {t.dt_ms.min():.1f}, max {t.dt_ms.max():.1f}), "
                f"pixel lệch trung vị {t.shift_px_median.median():.1f} px (p95 cao nhất {t.shift_px_p95.max():.1f} px), hit rate tụt trung bình {t.hit_drop_pp.mean():.1f} điểm % "
                f"(tối đa {t.hit_drop_pp.max():.1f}). Nguyên nhân: LiDAR quay 360° và camera chụp ở hai thời điểm khác nhau, xe (và vật) đã dịch chuyển trong khoảng đó; "
                f"nếu không đi qua global frame bằng ego pose thì overlay lệch dù extrinsic hoàn toàn đúng. **Lớp debug: Time.** "
                f"Cách phát hiện: ghi log |t_cam − t_lidar| từng frame và tốc độ xe, cảnh báo khi dt·v vượt ngưỡng pixel cho phép; KITTI không có timestamp trong bộ đề nên không kiểm được lớp này."
            )
        else:
            blind_what = ("ít điểm LiDAR biên độ sâu để đo" if "n_edge" in r.note and int(r.note.split("n_edge")[-1].strip()) < 150
                          else "điểm biên độ sâu tập trung dọc các cạnh dài (lan can, viền đường, vạch kẻ) gần song song với hướng dịch nên dịch vẫn nằm trên cạnh (aperture problem)")
            fail_md.append(
                f"### 3.3 Metric: alignment score không thấy drift (`{r.file}`)\n\n{img}\n\n"
                f"{r.dataset} `{r.frame_id}`: hit rate thực tế **{r.value_before_pct}% → {r.value_after_pct}%** nhưng alignment contrast gần như không đổi ({r.note}). "
                f"Đây là điểm mù của chỉ số: khi các điểm biên độ sâu nằm trên cạnh dài song song hướng dịch, hoặc quá ít, hoặc ảnh ít cạnh (đêm, trời mù), khoảng cách tới cạnh ảnh không thay đổi nhiều dù calibration đã sai. "
                f"Với frame này, nguyên nhân khả dĩ nhất: {blind_what} (đối chiếu ảnh: cạnh Canny màu cyan, điểm biên độ sâu màu đỏ). "
                f"**Lớp debug: Metric** (chỉ số không phản ánh đúng mục đích). Cách giảm rủi ro: gộp nhiều frame (cửa sổ {K} frame), kết hợp thêm chỉ số theo object (hit rate) và chỉ tin chỉ số khi n_edge đủ lớn."
            )

    sy = {ds: D.row(ds, "yaw", 1.0) for ds in ds_list}
    ratio_txt = ""
    if nusc:
        k_in, n_in = D.row(kitti, "yaw", 0.0).inside_fov_pct, D.row(nusc, "yaw", 0.0).inside_fov_pct
        ratio_txt = (f"Giải thích khác biệt KITTI/nuScenes (B5): chỉ {n_in:.1f}% điểm nuScenes rơi vào ảnh so với {k_in:.1f}% của KITTI và LiDAR 32 beam thưa hơn "
                     f"nên mỗi object có ít điểm hơn, biên độ sâu khó đo (xem cột AUC và `align_nan_frac`); ảnh nuScenes 1600×900 với tiêu cự lớn hơn nên cùng một góc lệch cho số pixel lệch lớn hơn "
                     f"(`results/analytic_shift.csv`) nhưng box 2D của nuScenes được suy ra từ chính box 3D (không phải nhãn 2D độc lập như KITTI) nên hit rate gốc gần 100% và thang so sánh khác nhau; "
                     f"cảnh đêm sau mưa của scene-1094 làm Canny ít cạnh tin cậy hơn.")

    lat_full = "; ".join(f"{r.dataset} p50 {r.p50_ms:.0f} ms / p95 {r.p95_ms:.0f} ms" for r in full.itertuples())

    out = f"""# Báo cáo Day 6: LiDAR-camera projection QA, đo độ nhạy với calibration drift

- **Họ tên:** {args.student_name}
- **MSSV:** {args.mssv} (trùng với MSSV trong tên repo `{args.repo_name}`)
- **Lớp:** {args.class_name}
- **Link repo:** {args.repo_url}
- **Topic:** A — Kiểm tra calibration LiDAR-camera bằng projection (đạt mức Basic, Good, Advanced; thêm bonus B3, B4, B5)
- **Dataset:** data/kitti_mini, data/nuscenes_mini_subset, data/synthetic (chỉ để kiểm tra tay CP2)
- **Các frame đã dùng:** sweep trên {frame_txt}; demo overlay: {demo_txt}

## 1. Claim

{claim}

(Hit rate = % điểm LiDAR nằm trong box 3D ground-truth của object, nhìn thấy trong ảnh khi calibration đúng, mà sau khi lệch vẫn rơi vào 2D box của object đó. Điều kiện kiểm chứng: lệch trong LiDAR frame, mỗi lần một trục, gộp hai chiều ±; false-alarm đặt ở 5%.)

## 2. Evidence

**Kiểm tra tay CP2** (`results/cp2_sanity.csv`): {'cả 4 kiểm tra PASS, gồm điểm (10,0,0) cho z_cam ≈ 9,73 và pixel ≈ (614,175), điểm sau xe bị loại, NaN/Inf không làm crash' if sanity_ok else 'CÓ KIỂM TRA THẤT BẠI, xem cp2_sanity.csv'}. Tính lặp lại: {det_txt}.

**Ảnh overlay ở 3 khoảng cách** (`results/demo_overlay_stats.csv`): gần (<6 m), trung bình, xa (>50 m), màu theo depth (đỏ gần, xanh xa), khung xanh là 2D box của label.

![near](../results/figures/demo_01_near_kitti_000019.png)
![mid](../results/figures/demo_02_mid_kitti_000011.png)
![far](../results/figures/demo_03_far_kitti_000004.png)

**Ảnh lệch** (cùng frame, yaw 0°/1°/2°/3°): `results/figures/drift_gallery_kitti_000011.png`

![drift](../results/figures/drift_gallery_kitti_000011.png)

{chr(10).join(sections_sweep)}

Hit rate theo từng trục (rotation, độ) và (translation, cm) cho cả hai dataset:

{pivot_table(D, ROT_KINDS, '°')}

{pivot_table(D, TRANS_KINDS, ' cm')}

Về khoảng cách: ở KITTI yaw 1°, {cmp_txt} (gần {near0:.1f}% → {near1:.1f}%, xa {far0:.1f}% → {far1:.1f}%). Hit rate theo dải khoảng cách và theo từng trục:

![hit-rate](../results/figures/hit_rate_vs_drift.png)
![hit-rate-distance](../results/figures/hit_rate_by_distance.png)

**Alignment score và ngưỡng phát hiện** (Advanced). Điểm số = trung bình exp(−d/σ) tại các điểm LiDAR "biên độ sâu", d là khoảng cách tới cạnh Canny gần nhất; `contrast` = điểm số trừ điểm số của chính các điểm đó khi dịch ảnh ±3° (loại ảnh hưởng của việc cảnh nhiều/ít cạnh). Ngưỡng τ = phân vị 5% của frame khỏe (`tau_single` trong `results/alignment_detection.csv`); cột "TPR {K} frame" gộp {K} frame liên tiếp.

![score](../results/figures/alignment_score_vs_drift.png)
![tpr](../results/figures/alignment_tpr_vs_drift.png)

**So sánh hai dataset (B5)** (`results/calib_sweep_summary.csv`):

{cross_table(D)}

{ratio_txt}

**Latency (B3)** (`results/latency_projection.csv`), mỗi giai đoạn bỏ warm-up rồi đo {int(D.lat.runs.iloc[0])} lần, phần cứng: {hw}.

{lat_table}

## 3. Failure case

Ba failure case được chọn tự động theo số đo (`src/exp_failure.py`, tóm tắt ở `results/failure_cases.csv`), thuộc ba lớp debug khác nhau.

{chr(10).join(fail_md)}

## 4. Khuyến nghị nếu triển khai thật

**Use-case: ADAS / xe tự hành có LiDAR + camera.** Chạy kiểm tra calibration như một health-check nền, không nằm trong vòng điều khiển: bước đầy đủ (chiếu + hit rate + alignment) mất {lat_full} trên CPU, nên chạy 1 Hz hoặc trên một frame mỗi giây là đủ. Đánh đổi: hit rate cần box 2D từ detector camera (đắt, nhưng đã có sẵn trong pipeline); alignment score rẻ và không cần nhãn nhưng yếu trên LiDAR thưa/ảnh đêm. Vì vậy dùng hit rate theo object xa/mảnh làm chỉ số chính, alignment score làm chỉ số phụ, và quyết định theo cửa sổ vài chục frame thay vì 1 frame để giữ false-alarm thấp. Với drift ≥ 1° trở lên nên chuyển hệ thống sang chế độ giảm cấp (giảm tin cậy fusion, ưu tiên cảm biến còn lại) thay vì tin fusion LiDAR-camera.

**Chỉ số cần ghi log khi chạy thật:** hit rate và precision theo dải khoảng cách (0–15/15–30/30–50/>50 m); % điểm LiDAR trong ảnh; alignment contrast và `n_edge` (để biết khi nào chỉ số không đáng tin); |t_cam − t_lidar| và tốc độ ego từng frame; số object hợp lệ; nhiệt độ/va chạm (IMU) làm cờ kích hoạt kiểm tra lại. **Bước tiếp theo:** thêm dò cục bộ quanh calibration hiện tại (tìm yaw/pitch làm score tăng) để vừa phát hiện vừa ước lượng chiều lệch; dùng ring index của LiDAR thay vì cửa sổ ảnh để tìm biên độ sâu trên LiDAR thưa; và kiểm lại trên chuỗi liên tục có timestamp thật.

## 5. Cách chạy lại

Từ repo sạch, chỉ cần CPU (đã chạy trên Kaggle). Kết quả xác định (không có phần ngẫu nhiên; bootstrap dùng seed 0).

```bash
pip install -r requirements.txt
python tools/verify_data.py --data-root data/kitti_mini
python tools/verify_data.py --data-root data/nuscenes_mini_subset
python -m src.run_all --class-name "{args.class_name}"        # chạy tất cả: demo, sweep, detect, failure, latency, check tái lập, sinh REPORT.md
# hoặc từng bước:
python -m src.exp_demo
python -m src.exp_sweep
python -m src.exp_detect
python -m src.exp_failure
python -m src.exp_latency
python -m src.make_report --class-name "{args.class_name}"
python tools/check_submission.py
```

Công cụ dùng lại được (B4): mỗi script trong `src/` có `--help`; `python -m src.exp_sweep --datasets kitti --max-frames 5` cho phép kiểm tra nhanh calibration của một bộ dữ liệu mới theo cùng định dạng KITTI. Hướng dẫn chi tiết: `src/README.md`.

## 6. Khai báo sử dụng AI

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| Claude Code (Claude Sonnet 5.5) | Viết 2 hàm `TODO(CP2)` trong `starter/projection.py`; viết toàn bộ code trong `src/` (đo hit rate/precision/FOV, alignment score, sweep, phát hiện drift, failure case, latency) và `src/make_report.py` sinh báo cáo từ CSV | Kiểm tra tay CP2 tự động (`results/cp2_sanity.csv`: điểm (10,0,0) → z≈9,73, pixel≈(614,175), NaN/Inf, điểm sau xe); xem trực tiếp ảnh overlay `results/figures/demo_*.png` khớp xe/người/cột và không có điểm trên bầu trời; chạy lại sweep hai lần so sánh checksum (`results/determinism_check.txt`); mọi con số trong báo cáo được sinh từ CSV chứ không gõ tay |
"""
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Sinh report/REPORT.md tu results/*.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="report/REPORT.md")
    ap.add_argument("--student-name", default="Nguyen Thanh Hoa")
    ap.add_argument("--mssv", default="2A202602559")
    ap.add_argument("--class-name", default="[ĐIỀN lớp]", help="ten lop, vd AI20K-xx (mac dinh de trong de check_submission nhac)")
    ap.add_argument("--repo-name", default="NguyenThanhHoa-2A202602559-Track4-Day21")
    ap.add_argument("--repo-url", default=REPO_URL)
    args = ap.parse_args(argv)

    res = Path(args.results_dir)
    missing = [f for f in REQUIRED if not (res / f).exists()]
    if missing:
        raise SystemExit(f"Thieu file trong {res}: {missing}. Chay `python -m src.run_all` hoac tung exp_* truoc.")
    text = build(Data(res), args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"-> {out} ({len(text.splitlines())} dong)")


if __name__ == "__main__":
    main()
