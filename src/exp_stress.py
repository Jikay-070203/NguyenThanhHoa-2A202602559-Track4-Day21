"""Bonus B2: stress test suy giảm dữ liệu LiDAR, đo ảnh hưởng lên việc kiểm tra calibration (không cần model).

    python -m src.exp_stress                          # KITTI (20 frame) + nuScenes (mỗi 2 frame)
    python -m src.exp_stress --max-frames 6           # chạy thử nhanh

5 loại suy giảm (dùng `starter/perturb.py`, seed cố định), mỗi loại 3-4 mức:
    random_dropout  giữ 90/70/50/30% điểm            gaussian_noise  sigma 2/5/10 cm
    range_dropout   chỉ giữ điểm <= 50/30/20 m       beam_dropout    giữ 1/2, 1/3, 1/4 số beam
    motion_smear    xe chạy 10/20/30 m/s không deskew

Với mỗi (frame, suy giảm) đo trên point cloud ĐÃ suy giảm, calibration đúng và lệch yaw +-1 độ:
  * điểm trên object (đếm điểm trong box 3D GT nhìn thấy trong ảnh) và số object còn >= 10 điểm
  * % điểm trong ảnh, hit rate khi calibration đúng và khi lệch yaw 1 độ
  * alignment contrast và AUC phân biệt "calibration đúng" với "lệch yaw 1 độ"; tỉ lệ frame còn chấm được điểm

Sản phẩm: results/stress_degradation_frames.csv, results/stress_degradation.csv,
          results/figures/stress_degradation.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.calib_qa import build_ctx, drifted_calib, evaluate  # noqa: E402
from src.exp_detect import auc_healthy_higher  # noqa: E402
from starter.datasets import list_frames, load_frame  # noqa: E402
from starter.perturb import beam_dropout, gaussian_noise, motion_smear, random_dropout, range_dropout  # noqa: E402

ROOTS = {"kitti": "data/kitti_mini", "nuscenes": "data/nuscenes_mini_subset"}
BEAM_CFG = {"kitti": dict(n_beams=64, fov_deg=(-24.9, 2.0)), "nuscenes": dict(n_beams=32, fov_deg=(-30.0, 10.0))}
DEGRADATIONS = (
    ("random_dropout", "keep_ratio", (0.9, 0.7, 0.5, 0.3)),
    ("gaussian_noise", "sigma_m", (0.02, 0.05, 0.1)),
    ("range_dropout", "max_range_m", (50.0, 30.0, 20.0)),
    ("beam_dropout", "keep_every", (2, 3, 4)),
    ("motion_smear", "speed_mps", (10.0, 20.0, 30.0)),
)
DRIFT_YAW_DEG = 1.0


def _to_kitti_axes(p: np.ndarray) -> np.ndarray:
    """nuScenes (x phải, y trước) -> KITTI (x trước, y trái) để dùng hàm perturb viết cho KITTI."""
    out = p.copy()
    out[:, 0], out[:, 1] = p[:, 1], -p[:, 0]
    return out


def _from_kitti_axes(p: np.ndarray) -> np.ndarray:
    out = p.copy()
    out[:, 0], out[:, 1] = -p[:, 1], p[:, 0]
    return out


def degrade(points: np.ndarray, name: str, level: float, ds: str, seed: int) -> np.ndarray:
    pts = points[np.isfinite(points).all(axis=1)]
    if name == "random_dropout":
        return random_dropout(pts, float(level), seed=seed)
    if name == "gaussian_noise":
        return gaussian_noise(pts, float(level), seed=seed)
    if name == "range_dropout":
        return range_dropout(pts, float(level))
    if name == "beam_dropout":
        return beam_dropout(pts, keep_every=int(level), **BEAM_CFG[ds])
    if name == "motion_smear":
        if ds == "nuscenes":
            return _from_kitti_axes(motion_smear(_to_kitti_axes(pts), float(level)))
        return motion_smear(pts, float(level))
    raise ValueError(name)


def measure(ds: str, fr: dict, points: np.ndarray, deg: str, level: float) -> dict:
    ctx = build_ctx({**fr, "points": points})
    r0 = evaluate(ctx, ctx.calib)
    rp = evaluate(ctx, drifted_calib(ctx.calib, "yaw", +DRIFT_YAW_DEG))
    rm = evaluate(ctx, drifted_calib(ctx.calib, "yaw", -DRIFT_YAW_DEG))
    pop = int(r0["per_band"][:, 0].sum())
    return {"dataset": ds, "frame_id": fr["frame_id"], "degradation": deg, "level": level,
            "n_points": int(len(ctx.points)), "n_inside": r0["n_inside"], "n_objects": r0["n_objects"], "n_object_points": pop,
            "hit_clean_calib": int(r0["per_band"][:, 1].sum()), "hit_yaw_p": int(rp["per_band"][:, 1].sum()),
            "hit_yaw_m": int(rm["per_band"][:, 1].sum()),
            "contrast_clean_calib": r0["align_contrast"], "contrast_yaw_p": rp["align_contrast"], "contrast_yaw_m": rm["align_contrast"],
            "n_edge_clean_calib": r0["n_edge"]}


def run(datasets, max_frames: int, stride_nusc: int) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        frames = list_frames(ROOTS[ds])
        if ds == "nuscenes":
            frames = frames[::stride_nusc]
        if max_frames:
            frames = frames[::max(1, len(frames) // max_frames)][:max_frames]
        for i, fid in enumerate(frames):
            fr = load_frame(ROOTS[ds], fid)
            rows.append(measure(ds, fr, fr["points"][np.isfinite(fr["points"]).all(axis=1)], "none", 0.0))
            for name, _, levels in DEGRADATIONS:
                for lv in levels:
                    rows.append(measure(ds, fr, degrade(fr["points"], name, lv, ds, seed=1000 + i), name, float(lv)))
            print(f"[{ds}] {i + 1}/{len(frames)} frame", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, d in df.groupby("dataset", sort=False):
        clean = d[d.degradation == "none"]
        c_pts, c_pop, c_obj = clean.n_points.sum(), clean.n_object_points.sum(), clean.n_objects.sum()
        for (deg, lv), g in d.groupby(["degradation", "level"], sort=False):
            pop = g.n_object_points.sum()
            healthy = g.contrast_clean_calib.dropna().to_numpy()
            drift = pd.concat([g.contrast_yaw_p, g.contrast_yaw_m]).dropna().to_numpy()
            rows.append({
                "dataset": ds, "degradation": deg, "level": lv, "n_frames": g.frame_id.nunique(),
                "points_kept_pct": 100 * g.n_points.sum() / c_pts,
                "inside_fov_pct": 100 * g.n_inside.sum() / max(g.n_points.sum(), 1),
                "object_points_pct": 100 * pop / c_pop, "objects_kept_pct": 100 * g.n_objects.sum() / c_obj,
                "points_per_object": pop / max(g.n_objects.sum(), 1),
                "hit_clean_calib_pct": 100 * g.hit_clean_calib.sum() / pop if pop else np.nan,
                "hit_yaw1_pct": 100 * (g.hit_yaw_p.sum() + g.hit_yaw_m.sum()) / (2 * pop) if pop else np.nan,
                "contrast_clean_calib": healthy.mean() if len(healthy) else np.nan,
                "contrast_yaw1": drift.mean() if len(drift) else np.nan,
                "score_available_pct": 100 * g.contrast_clean_calib.notna().mean(),
                "auc_yaw1": auc_healthy_higher(healthy, drift),
            })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, fig_dir: Path) -> None:
    panels = (("object_points_pct", "objects_kept_pct", "% so voi du lieu sach", "diem tren object (-) / so object con lai (--)", "diem/obj", "so obj"),
              ("hit_clean_calib_pct", "hit_yaw1_pct", "hit rate %", "hit rate: calib dung (-) / lech yaw 1 deg (--)", "calib dung", "yaw 1deg"),
              ("auc_yaw1", "score_available_pct", "AUC / ti le cham duoc", "AUC contrast (-) / ti le frame cham duoc diem (--)", "AUC", "cham duoc"))
    fig, axes = plt.subplots(len(DEGRADATIONS), 3, figsize=(15, 3.3 * len(DEGRADATIONS)), squeeze=False)
    for r, (deg, _, levels) in enumerate(DEGRADATIONS):
        labels = ["sach", *[f"{lv:g}" for lv in levels]]
        x = np.arange(len(labels))
        for c, (m1, m2, yl, title, lab1, lab2) in enumerate(panels):
            ax = axes[r, c]
            for ds, mk in (("kitti", "o"), ("nuscenes", "s")):
                d = summary[(summary.dataset == ds) & ((summary.degradation == deg) | (summary.degradation == "none"))]
                d = d.set_index("level").reindex([0.0, *[float(lv) for lv in levels]])
                if d[m1].isna().all():
                    continue
                y1, y2 = d[m1].to_numpy(), d[m2].to_numpy()
                ax.plot(x, y1, f"-{mk}", label=f"{ds} {lab1}")
                ax.plot(x, y2 / 100 if c == 2 else y2, f"--{mk}", alpha=0.6, label=f"{ds} {lab2}")
            if c == 2:
                ax.axhline(0.5, color="gray", ls=":", lw=0.8)
                ax.set_ylim(0.0, 1.02)
            ax.set_xticks(x, labels)
            ax.set_title(f"{deg}: {title}", fontsize=9)
            ax.set_xlabel("muc suy giam")
            ax.set_ylabel(yl, fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(fig_dir / "stress_degradation.png", dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="B2 - stress test suy giam du lieu LiDAR tren kiem tra calibration")
    ap.add_argument("--datasets", nargs="+", default=list(ROOTS), choices=list(ROOTS))
    ap.add_argument("--max-frames", type=int, default=0, help="gioi han so frame moi dataset (0 = tat ca)")
    ap.add_argument("--nusc-stride", type=int, default=2, help="nuScenes: lay moi N frame (mac dinh 2 -> 40 frame)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = run(args.datasets, args.max_frames, args.nusc_stride)
    df.to_csv(out / "stress_degradation_frames.csv", index=False, float_format="%.6g")
    summary = summarize(df)
    summary.to_csv(out / "stress_degradation.csv", index=False, float_format="%.4f")
    plot(summary, fig_dir)
    cols = ["dataset", "degradation", "level", "points_kept_pct", "object_points_pct", "hit_clean_calib_pct", "hit_yaw1_pct",
            "score_available_pct", "auc_yaw1"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"-> {out / 'stress_degradation.csv'}")


if __name__ == "__main__":
    main()
