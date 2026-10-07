"""Bonus B1: so sánh 2 thuật toán alignment score trên CÙNG dữ liệu, CÙNG lưới drift, CÙNG metric.

    python -m src.exp_compare                       # KITTI (20 frame) + nuScenes (80 frame)
    python -m src.exp_compare --max-frames 8        # chạy thử nhanh

  A (Canny)    : điểm = exp(-khoảng_cách_tới_cạnh_Canny / sigma)           -> "cạnh nhị phân + hố hút"
  B (Gradient) : điểm = độ lớn gradient Sobel chuẩn hoá (phân vị 99%), làm mờ Gaussian cùng sigma -> "cạnh mềm"

Hai thuật toán dùng CHUNG tập điểm LiDAR "biên độ sâu", CHUNG cách trừ mức ngẫu nhiên (dịch ±3°), CHUNG ngưỡng
false-alarm 5%: chỉ khác bản đồ ảnh dùng để chấm điểm. Đo: AUC 1 frame, TPR 1 frame, TPR cửa sổ 5 frame,
tỉ lệ có điểm số (không NaN) và thời gian dựng bản đồ ảnh / frame.

Sản phẩm:
    results/compare_methods_frames.csv   điểm số từng (frame, drift) của 2 thuật toán
    results/compare_methods.csv          AUC / TPR theo (dataset, thuật toán, kind, mức lệch)
    results/compare_methods_cost.csv     thời gian dựng bản đồ ảnh p50/p95 mỗi frame
    results/figures/compare_methods_auc.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.calib_qa import (MIN_EDGE_POINTS, NULL_SHIFT_DEG, SIGMA_DEG, build_ctx, drifted_calib,  # noqa: E402
                          edge_weight_map, lidar_edge_points, project_full)
from src.exp_detect import detection_table  # noqa: E402
from starter.datasets import list_frames, load_frame  # noqa: E402

ROOTS = {"kitti": "data/kitti_mini", "nuscenes": "data/nuscenes_mini_subset"}
GRID = (("yaw", (0.5, 1.0, 2.0, 3.0)), ("pitch", (0.5, 1.0, 2.0, 3.0)), ("roll", (0.5, 1.0, 2.0, 3.0)),
        ("tx", (10.0, 20.0)), ("tz", (10.0, 20.0)))
METHODS = {"canny": "A: Canny + distance transform", "grad": "B: gradient Sobel mem"}
METRICS = ("canny_contrast", "grad_contrast", "canny_score", "grad_score")


def build_grad_map(image: np.ndarray, focal_px: float) -> np.ndarray:
    """Thuật toán B: độ lớn gradient Sobel chuẩn hoá về [0, 1] rồi làm mờ Gaussian cùng sigma với thuật toán A."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    scale = float(np.percentile(mag, 99))
    g = np.minimum(mag / (scale if scale > 0 else 1.0), 1.0).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), max(2.0, focal_px * np.deg2rad(SIGMA_DEG)))


def score_on_map(m: np.ndarray, ue: np.ndarray, ve: np.ndarray, null_px: int):
    """(score, contrast) của các điểm biên (ue, ve) trên bản đồ m; contrast = score - điểm số khi dịch ±null_px."""
    h, w = m.shape
    score = float(m[ve, ue].mean())
    null = np.mean([m[np.clip(ve + dv, 0, h - 1), np.clip(ue + du, 0, w - 1)].mean()
                    for du, dv in ((null_px, 0), (-null_px, 0), (0, null_px), (0, -null_px))])
    return score, float(score - null)


def frame_rows(ds: str, fr: dict, grid) -> tuple[list[dict], dict]:
    focal = float(fr["calib"].P2[0, 0])
    ctx = build_ctx(fr)
    t0 = time.perf_counter()
    edge_weight_map(fr["image"], focal)
    t_canny = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    grad_map = build_grad_map(fr["image"], focal)
    t_grad = (time.perf_counter() - t0) * 1000

    rows = []
    for kind, mag, sign in grid:
        calib = ctx.calib if kind == "baseline" else drifted_calib(ctx.calib, kind, sign * mag)
        uv, z, mask = project_full(ctx.points, calib, ctx.image_shape)
        row = {"dataset": ds, "frame_id": ctx.frame_id, "kind": kind, "magnitude": mag, "signed_value": sign * mag,
               "n_edge": 0, "canny_score": np.nan, "canny_contrast": np.nan, "grad_score": np.nan, "grad_contrast": np.nan}
        if mask.any():
            is_edge, ui, vi = lidar_edge_points(uv[mask], z[mask], ctx.image_shape, ctx.win)
            row["n_edge"] = int(is_edge.sum())
            if row["n_edge"] >= MIN_EDGE_POINTS:
                ue, ve = ui[is_edge], vi[is_edge]
                row["canny_score"], row["canny_contrast"] = score_on_map(ctx.weight_map, ue, ve, ctx.null_px)
                row["grad_score"], row["grad_contrast"] = score_on_map(grad_map, ue, ve, ctx.null_px)
        rows.append(row)
    return rows, {"dataset": ds, "frame_id": ctx.frame_id, "canny_map_ms": t_canny, "grad_map_ms": t_grad}


def drift_grid():
    g = [("baseline", 0.0, 1)]
    for kind, mags in GRID:
        g += [(kind, m, s) for m in mags for s in (+1, -1)]
    return g


def plot_auc(det: pd.DataFrame, fig_dir: Path) -> None:
    ds_list = list(det.dataset.unique())
    kinds = ("yaw", "pitch", "roll")
    fig, axes = plt.subplots(len(ds_list), len(kinds), figsize=(4.6 * len(kinds), 3.6 * len(ds_list)), squeeze=False)
    for r, ds in enumerate(ds_list):
        for c, k in enumerate(kinds):
            ax = axes[r, c]
            for metric, style in (("canny_contrast", "-o"), ("grad_contrast", "--s")):
                s = det[(det.dataset == ds) & (det.kind == k) & (det.metric == metric)].sort_values("magnitude")
                ax.plot(s.magnitude, s.auc_single, style, label=METHODS[metric.split("_")[0]])
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)
            ax.set_ylim(0.4, 1.02)
            ax.set_title(f"{ds}: {k}")
            ax.set_xlabel("drift (deg)")
            ax.set_ylabel("AUC 1 frame (contrast)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "compare_methods_auc.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="B1 - so sanh 2 thuat toan alignment score (Canny vs gradient)")
    ap.add_argument("--datasets", nargs="+", default=list(ROOTS), choices=list(ROOTS))
    ap.add_argument("--max-frames", type=int, default=0, help="gioi han so frame moi dataset (0 = tat ca)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    grid = drift_grid()
    rows, costs = [], []
    for ds in args.datasets:
        frames = list_frames(ROOTS[ds])
        if args.max_frames:
            frames = frames[::max(1, len(frames) // args.max_frames)][:args.max_frames]
        for i, fid in enumerate(frames):
            r, c = frame_rows(ds, load_frame(ROOTS[ds], fid), grid)
            rows += r
            costs.append(c)
            if (i + 1) % 10 == 0 or i + 1 == len(frames):
                print(f"[{ds}] {i + 1}/{len(frames)} frame", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "compare_methods_frames.csv", index=False, float_format="%.6g")
    det = detection_table(df, args.window, args.n_boot, args.seed, metrics=METRICS)
    det.insert(2, "method", det.metric.str.split("_").str[0])
    det.to_csv(out / "compare_methods.csv", index=False, float_format="%.5g")
    cost = pd.DataFrame(costs).groupby("dataset").agg(
        canny_map_p50_ms=("canny_map_ms", "median"), canny_map_p95_ms=("canny_map_ms", lambda x: np.percentile(x, 95)),
        grad_map_p50_ms=("grad_map_ms", "median"), grad_map_p95_ms=("grad_map_ms", lambda x: np.percentile(x, 95)),
        n_frames=("frame_id", "count")).reset_index()
    cost.to_csv(out / "compare_methods_cost.csv", index=False, float_format="%.3f")
    plot_auc(det, fig_dir)

    show = det[(det.metric.str.endswith("contrast")) & det.kind.isin(["yaw", "pitch"]) & np.isclose(det.magnitude, 1.0)]
    print(show[["dataset", "method", "kind", "magnitude", "auc_single", "tpr_single", "tpr_window"]].to_string(index=False))
    print(cost.to_string(index=False))
    print(f"-> {out / 'compare_methods.csv'}")


if __name__ == "__main__":
    main()
