"""CP3 / Topic A Good: quét calibration drift (rotation 0-3 deg, translation 0-20 cm) và đo mismatch.

    python -m src.exp_sweep                         # KITTI (20 frame) + nuScenes (80 frame)
    python -m src.exp_sweep --quick                 # chạy thử nhanh: 3 frame/dataset, lưới thưa
    python -m src.exp_sweep --datasets kitti --max-frames 5

Mỗi lần chỉ đổi MỘT trục (roll/pitch/yaw hoặc tx/ty/tz), giữ nguyên frame, nhãn, ngưỡng. Mỗi mức lệch được
đo ở cả hai chiều (+/-) rồi gộp. Không có phần ngẫu nhiên => chạy lại ra cùng số.

Sản phẩm:
    results/calib_sweep_frames.csv      từng (frame, kind, mức lệch, dấu): số đếm thô + alignment score
    results/calib_sweep_summary.csv     gộp theo (dataset, kind, mức lệch): inside_fov, hit rate, precision, align
    results/analytic_shift.csv          độ lệch hình học thuần tuý (m và pixel) theo khoảng cách
    results/figures/hit_rate_vs_drift.png, hit_rate_by_distance.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.calib_qa import (BAND_NAMES, N_BANDS, ROT_KINDS, TRANS_KINDS, build_ctx, drifted_calib,  # noqa: E402
                          evaluate, unit_of)
from starter.datasets import list_frames, load_frame  # noqa: E402

DATASET_ROOTS = {"kitti": "data/kitti_mini", "nuscenes": "data/nuscenes_mini_subset"}
ROT_MAGS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)          # độ
TRANS_MAGS = (2.0, 5.0, 10.0, 20.0)                  # cm
QUICK_ROT, QUICK_TRANS = (1.0, 3.0), (5.0,)
BAND_COLS = [f"{n}_{i}" for i in range(N_BANDS) for n in ("pop", "hit", "inbox", "inbox_true")]


def drift_grid(rot_mags, trans_mags):
    """[(kind, mức, dấu)], thêm baseline (kind='baseline', 0, +1) ở đầu."""
    grid = [("baseline", 0.0, 1)]
    for kind in ROT_KINDS:
        grid += [(kind, m, s) for m in rot_mags for s in (+1, -1)]
    for kind in TRANS_KINDS:
        grid += [(kind, m, s) for m in trans_mags for s in (+1, -1)]
    return grid


def frame_rows(dataset: str, fr: dict, grid) -> list[dict]:
    ctx = build_ctx(fr)
    rows = []
    for kind, mag, sign in grid:
        calib = ctx.calib if kind == "baseline" else drifted_calib(ctx.calib, kind, sign * mag)
        res = evaluate(ctx, calib)
        row = {"dataset": dataset, "frame_id": ctx.frame_id, "kind": kind,
               "unit": unit_of(kind) if kind != "baseline" else "", "magnitude": mag,
               "signed_value": sign * mag, "n_points": res["n_points"], "n_inside": res["n_inside"],
               "n_objects": res["n_objects"], "align_score": res["align_score"], "n_edge": res["n_edge"],
               "align_contrast": res["align_contrast"], "align_chance": ctx.chance}
        flat = res["per_band"].reshape(-1)
        row.update(dict(zip(BAND_COLS, (int(x) for x in flat))))
        rows.append(row)
    return rows


def run_sweep(datasets, max_frames, rot_mags, trans_mags) -> pd.DataFrame:
    grid = drift_grid(rot_mags, trans_mags)
    rows = []
    for ds in datasets:
        root = DATASET_ROOTS[ds]
        frames = list_frames(root)
        if max_frames:
            step = max(1, len(frames) // max_frames)
            frames = frames[::step][:max_frames]
        t0 = time.perf_counter()
        for i, fid in enumerate(frames):
            rows += frame_rows(ds, load_frame(root, fid), grid)
            if (i + 1) % 5 == 0 or i + 1 == len(frames):
                print(f"[{ds}] {i + 1}/{len(frames)} frame, {time.perf_counter() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def _agg(g: pd.DataFrame) -> dict:
    out = {"n_frames": g.frame_id.nunique(), "n_evals": len(g),
           "inside_fov_pct": 100 * g.n_inside.sum() / max(g.n_points.sum(), 1)}
    pop, hit = sum(g[f"pop_{i}"].sum() for i in range(N_BANDS)), sum(g[f"hit_{i}"].sum() for i in range(N_BANDS))
    inbox, true = sum(g[f"inbox_{i}"].sum() for i in range(N_BANDS)), sum(g[f"inbox_true_{i}"].sum() for i in range(N_BANDS))
    out.update({"n_object_points": int(pop), "hit_rate_pct": 100 * hit / pop if pop else np.nan,
                "precision_pct": 100 * true / inbox if inbox else np.nan})
    for i, name in enumerate(BAND_NAMES):
        p, h = g[f"pop_{i}"].sum(), g[f"hit_{i}"].sum()
        out[f"hit_{name}_pct"] = 100 * h / p if p else np.nan
        out[f"npts_{name}"] = int(p)
    out["align_mean"] = g.align_score.mean()
    out["align_std"] = g.align_score.std()
    out["contrast_mean"] = g.align_contrast.mean()
    out["contrast_std"] = g.align_contrast.std()
    out["align_nan_frac"] = g.align_score.isna().mean()
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, d in df.groupby("dataset", sort=False):
        base = d[d.kind == "baseline"]
        for kind in [k for k in (*ROT_KINDS, *TRANS_KINDS) if (d.kind == k).any()]:
            groups = [(0.0, base)] + [(m, g) for m, g in d[d.kind == kind].groupby("magnitude")]
            ref = None
            for mag, g in groups:
                row = {"dataset": ds, "kind": kind, "unit": unit_of(kind), "magnitude": mag, **_agg(g)}
                ref = row if ref is None else ref
                row["hit_drop_pp"] = ref["hit_rate_pct"] - row["hit_rate_pct"]
                row["precision_drop_pp"] = ref["precision_pct"] - row["precision_pct"]
                row["inside_fov_drop_pp"] = ref["inside_fov_pct"] - row["inside_fov_pct"]
                rows.append(row)
    return pd.DataFrame(rows)


def analytic_shift(datasets, rot_mags, trans_mags) -> pd.DataFrame:
    """Độ lệch hình học thuần tuý, không cần dữ liệu: yaw/pitch làm LiDAR lệch d*tan(theta) mét ở khoảng cách d,
    tức f*tan(theta) pixel (không phụ thuộc d). Tịnh tiến t cm làm lệch f*t/d pixel (giảm theo d)."""
    rows = []
    for ds in datasets:
        root = DATASET_ROOTS[ds]
        f_px = float(load_frame(root, list_frames(root)[0])["calib"].P2[0, 0])
        for d in (5, 10, 20, 30, 50, 80):
            for th in rot_mags:
                lat = d * np.tan(np.deg2rad(th))
                rows.append({"dataset": ds, "kind": "rotation", "magnitude": th, "unit": "deg", "distance_m": d,
                             "focal_px": f_px, "lateral_shift_m": lat, "pixel_shift": f_px * np.tan(np.deg2rad(th)),
                             "shift_over_pedestrian_width": lat / 0.6, "shift_over_car_width": lat / 1.8})
            for t in trans_mags:
                rows.append({"dataset": ds, "kind": "translation", "magnitude": t, "unit": "cm", "distance_m": d,
                             "focal_px": f_px, "lateral_shift_m": t / 100.0, "pixel_shift": f_px * (t / 100.0) / d,
                             "shift_over_pedestrian_width": (t / 100.0) / 0.6, "shift_over_car_width": (t / 100.0) / 1.8})
    return pd.DataFrame(rows)


def plot_hit_rate(summary: pd.DataFrame, fig_dir: Path) -> None:
    ds_list = list(summary.dataset.unique())
    fig, axes = plt.subplots(len(ds_list), 2, figsize=(11, 3.8 * len(ds_list)), squeeze=False)
    for r, ds in enumerate(ds_list):
        for c, (kinds, xl) in enumerate(((ROT_KINDS, "rotation drift (deg)"), (TRANS_KINDS, "translation drift (cm)"))):
            ax = axes[r, c]
            for k in kinds:
                s = summary[(summary.dataset == ds) & (summary.kind == k)].sort_values("magnitude")
                if len(s):
                    ax.plot(s.magnitude, s.hit_rate_pct, marker="o", label=k)
            ax.set_title(f"{ds}: hit rate vs {xl.split()[0]}")
            ax.set_xlabel(xl)
            ax.set_ylabel("% diem object roi vao 2D box")
            ax.set_ylim(0, 102)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "hit_rate_vs_drift.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ds_list), figsize=(5.5 * len(ds_list), 4), squeeze=False)
    for c, ds in enumerate(ds_list):
        ax = axes[0, c]
        s = summary[(summary.dataset == ds) & (summary.kind == "yaw")].sort_values("magnitude")
        for name in BAND_NAMES:
            if s[f"npts_{name}"].iloc[0] >= 30:     # bỏ dải khoảng cách quá ít điểm
                ax.plot(s.magnitude, s[f"hit_{name}_pct"], marker="o", label=f"{name} (n={int(s[f'npts_{name}'].iloc[0])})")
        ax.set_title(f"{ds}: yaw drift, hit rate theo khoang cach")
        ax.set_xlabel("yaw drift (deg)")
        ax.set_ylabel("% diem object roi vao 2D box")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "hit_rate_by_distance.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Topic A - quet calibration drift, do hit rate / precision / FOV / alignment score")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_ROOTS), choices=list(DATASET_ROOTS))
    ap.add_argument("--max-frames", type=int, default=0, help="gioi han so frame moi dataset (0 = tat ca)")
    ap.add_argument("--quick", action="store_true", help="chay thu nhanh: 3 frame, luoi thua")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    rot, trans = (QUICK_ROT, QUICK_TRANS) if args.quick else (ROT_MAGS, TRANS_MAGS)
    max_frames = 3 if args.quick else args.max_frames
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = run_sweep(args.datasets, max_frames, rot, trans)
    df.to_csv(out / "calib_sweep_frames.csv", index=False, float_format="%.6g")
    summary = summarize(df)
    summary.to_csv(out / "calib_sweep_summary.csv", index=False, float_format="%.4f")
    analytic_shift(args.datasets, rot, trans).to_csv(out / "analytic_shift.csv", index=False, float_format="%.4f")
    if not args.no_plots:
        plot_hit_rate(summary, fig_dir)

    cols = ["dataset", "kind", "magnitude", "inside_fov_pct", "hit_rate_pct", "precision_pct", "contrast_mean"]
    print(summary[summary.kind.isin(["yaw", "tx"])][cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"-> {out / 'calib_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
