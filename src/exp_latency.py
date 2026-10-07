"""Bonus B3: đo latency p50/p95 của pipeline kiểm tra calibration (chạy CPU), đúng cách.

    python -m src.exp_latency                  # 3 lần warm-up (bỏ) + 30 lần đo cho mỗi giai đoạn

Mỗi giai đoạn đo bằng time.perf_counter, bỏ các lần warm-up, báo p50/p95 (không báo 1 lần đo đơn lẻ).
Sản phẩm: results/latency_projection.csv (kèm mô tả phần cứng/phiên bản thư viện trong từng dòng).
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from src.calib_qa import alignment_score, build_ctx, drifted_calib, evaluate, project_full
from starter.datasets import load_frame
from starter.projection import overlay_points, project_velo_to_image

CASES = (("kitti", "data/kitti_mini", "000011"), ("nuscenes", "data/nuscenes_mini_subset", "scene-0103_010"))


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def time_it(fn, warmup: int, runs: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(t)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Topic A - latency p50/p95 cua projection / overlay / alignment score")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=30, help=">= 20 theo yeu cau bonus B3")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hw = {"cpu": cpu_name(), "cores": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__,
          "opencv": cv2.__version__, "platform": platform.platform(), "gpu_used": "no (CPU only)"}
    rows = []
    for ds, root, fid in CASES:
        fr = load_frame(root, fid)
        ctx = build_ctx(fr)
        calib_bad = drifted_calib(fr["calib"], "yaw", 1.0)
        uv_f, z_f, mask = project_full(ctx.points, ctx.calib, ctx.image_shape)
        stages = {
            "project (velo_to_cam + cam_to_image)": lambda: project_velo_to_image(fr["points"], fr["calib"], fr["image"].shape),
            "overlay (ve diem len anh)": lambda: overlay_points(fr["image"], uv_f[mask], z_f[mask]),
            "alignment score (bien do sau + so canh)": lambda: alignment_score(ctx, uv_f[mask], z_f[mask]),
            "full check (project + hit/precision + alignment)": lambda: evaluate(ctx, calib_bad),
        }
        for name, fn in stages.items():
            t = time_it(fn, args.warmup, args.runs)
            rows.append({"dataset": ds, "frame_id": fid, "stage": name, "n_points": len(fr["points"]),
                         "warmup": args.warmup, "runs": args.runs, "p50_ms": np.percentile(t, 50),
                         "p95_ms": np.percentile(t, 95), "mean_ms": t.mean(), "min_ms": t.min(), "max_ms": t.max(), **hw})
            print(f"[{ds}] {name}: p50={rows[-1]['p50_ms']:.1f} ms  p95={rows[-1]['p95_ms']:.1f} ms")

    with (out / "latency_projection.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"CPU: {hw['cpu']} ({hw['cores']} cores) -> {out / 'latency_projection.csv'}")


if __name__ == "__main__":
    main()
