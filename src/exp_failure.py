"""CP4 / Topic A: tìm failure case có số liệu, chọn TỰ ĐỘNG theo số đo (không chọn tay).

    python -m src.exp_failure                 # cần results/calib_sweep_frames.csv + alignment_detection.csv

Ba failure case:
  fail_01  Geometry  : yaw +1 deg làm điểm LiDAR của người đi bộ/xe đạp văng khỏi 2D box (frame KITTI tệ nhất).
  fail_02  Time      : nuScenes bỏ bù ego-motion giữa lúc quét LiDAR và lúc chụp camera => overlay lệch dù calib đúng.
  fail_03  Metric    : drift thật (hit rate sụt mạnh) nhưng alignment score KHÔNG báo động (mù của chỉ số).

Sản phẩm:
    results/figures/fail_01_*.png, fail_02_*.png, fail_03_*.png
    results/time_sync_nuscenes.csv       độ lệch thời gian LiDAR-camera và hệ quả theo từng frame
    results/failure_cases.csv            tóm tắt số liệu của từng case (dùng cho báo cáo)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.calib_qa import (N_BANDS, ROT_KINDS, build_ctx, crop_box, drifted_calib, evaluate, hit_rate,
                          lidar_edge_points, project_full, put_title)
from starter.datasets import list_frames, load_frame
from starter.projection import draw_box2d, overlay_points, project_velo_to_image

SMALL_TYPES = ("Pedestrian", "Cyclist", "Person_sitting", "Bicycle", "Motorcycle")
KITTI_ROOT, NUSC_ROOT = "data/kitti_mini", "data/nuscenes_mini_subset"
ROOTS = {"kitti": KITTI_ROOT, "nuscenes": NUSC_ROOT}


def render_crop(fr: dict, calib, bbox, title: str, target_h: int = 340) -> np.ndarray:
    """Cắt quanh bbox, phóng to, vẽ điểm LiDAR (màu theo depth) + 2D box của label."""
    image = fr["image"]
    crop, x0, y0 = crop_box(image, bbox)
    scale = float(np.clip(target_h / crop.shape[0], 1.0, 6.0))
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    uv, depth, _ = project_velo_to_image(fr["points"], calib, image.shape)
    keep = ((uv[:, 0] >= x0) & (uv[:, 0] < x0 + crop.shape[1]) & (uv[:, 1] >= y0) & (uv[:, 1] < y0 + crop.shape[0]))
    big = overlay_points(big, (uv[keep] - [x0, y0]) * scale, depth[keep], radius=3)
    big = draw_box2d(big, (np.asarray(bbox) - [x0, y0, x0, y0]) * scale)
    return put_title(big, title)


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(40, 40, 40))  # noqa: E731
    return np.hstack([pad(a), pad(b)])


# --------------------------------------------------------------------------------------------- fail_01
def case_geometry_yaw(fig_dir: Path, yaw_deg: float = 1.0) -> dict:
    best = None
    for fid in list_frames(KITTI_ROOT):
        fr = load_frame(KITTI_ROOT, fid)
        ctx = build_ctx(fr)
        base = evaluate(ctx, ctx.calib, with_align=False)["per_object"]
        drift = evaluate(ctx, drifted_calib(ctx.calib, "yaw", yaw_deg), with_align=False)["per_object"]
        for k, oc in enumerate(ctx.objects):
            n_pop = base[k, 0]
            if n_pop < 15:
                continue
            r0, r1 = base[k, 1] / n_pop, drift[k, 1] / n_pop
            key = (oc.obj.type in SMALL_TYPES, r0 - r1)          # ưu tiên người/xe đạp, rồi mức tụt lớn nhất
            if best is None or key > best[0]:
                best = (key, fid, oc.obj, r0, r1, int(n_pop))
    if best is None:
        raise RuntimeError("Khong tim thay object nao du diem LiDAR de dung failure case 1")
    _, fid, obj, r0, r1, n_pop = best
    fr = load_frame(KITTI_ROOT, fid)
    z = float(obj.location[2])
    a = render_crop(fr, fr["calib"], obj.bbox, f"calib dung | hit {100 * r0:.0f}% ({n_pop} pts)")
    b = render_crop(fr, drifted_calib(fr["calib"], "yaw", yaw_deg), obj.bbox,
                    f"yaw {yaw_deg:+.1f} deg | hit {100 * r1:.0f}%")
    name = f"fail_01_yaw_{yaw_deg:g}deg_{obj.type.lower()}_{fid}.png"
    cv2.imwrite(str(fig_dir / name), put_title(side_by_side(a, b), f"KITTI {fid} - {obj.type} @ {z:.0f} m: LiDAR lech {yaw_deg:g} deg quanh truc z"))
    return {"case": "fail_01_geometry_yaw", "file": name, "layer": "Geometry", "dataset": "kitti", "frame_id": fid,
            "object_type": obj.type, "object_depth_m": round(z, 1), "value_before_pct": round(100 * r0, 1),
            "value_after_pct": round(100 * r1, 1), "note": f"yaw {yaw_deg:g} deg, {n_pop} diem LiDAR cua object"}


# --------------------------------------------------------------------------------------------- fail_02
def case_time_sync(fig_dir: Path, out_csv: Path, max_frames: int = 0) -> dict:
    frames = list_frames(NUSC_ROOT)
    if max_frames:
        frames = frames[::max(1, len(frames) // max_frames)][:max_frames]
    rows, best = [], None
    for fid in frames:
        fr_ok = load_frame(NUSC_ROOT, fid)
        fr_bad = load_frame(NUSC_ROOT, fid, use_ego_motion=False)
        ctx = build_ctx(fr_ok)
        res_ok, res_bad = evaluate(ctx, fr_ok["calib"], with_align=False), evaluate(ctx, fr_bad["calib"], with_align=False)
        uv_ok, _, m_ok = project_full(ctx.points, fr_ok["calib"], ctx.image_shape)
        uv_bad, _, m_bad = project_full(ctx.points, fr_bad["calib"], ctx.image_shape)
        both = m_ok & m_bad
        shift = np.linalg.norm(uv_ok[both] - uv_bad[both], axis=1) if both.any() else np.array([np.nan])
        pop = res_ok["per_band"][:, 0].sum()
        h_ok, h_bad = hit_rate(pop, res_ok["per_band"][:, 1].sum()), hit_rate(pop, res_bad["per_band"][:, 1].sum())
        row = {"frame_id": fid, "dt_ms": (fr_ok["timestamp_camera_us"] - fr_ok["timestamp_lidar_us"]) / 1000.0,
               "shift_px_median": float(np.median(shift)), "shift_px_p95": float(np.percentile(shift, 95)),
               "n_objects": res_ok["n_objects"], "n_object_points": int(pop),
               "hit_ego_comp_pct": 100 * h_ok, "hit_no_comp_pct": 100 * h_bad,
               "hit_drop_pp": 100 * (h_ok - h_bad) if pop else float("nan")}
        rows.append(row)
        per_obj_drop = (res_ok["per_object"][:, 1] - res_bad["per_object"][:, 1]) / np.maximum(res_ok["per_object"][:, 0], 1)
        if len(per_obj_drop):
            k = int(np.argmax(per_obj_drop))
            key = (np.nan_to_num(row["hit_drop_pp"], nan=-1), row["shift_px_median"])
            if best is None or key > best[0]:
                best = (key, fid, ctx.objects[k].obj, row)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.4f")

    if best is None:
        raise RuntimeError("Khong co frame nuScenes nao co object du diem de dung failure case 2")
    _, fid, obj, row = best
    fr_ok, fr_bad = load_frame(NUSC_ROOT, fid), load_frame(NUSC_ROOT, fid, use_ego_motion=False)
    a = render_crop(fr_ok, fr_ok["calib"], obj.bbox, "co bu ego-motion (calib dung theo thoi gian)")
    b = render_crop(fr_ok, fr_bad["calib"], obj.bbox, f"bo bu ego-motion (dt={row['dt_ms']:.0f} ms)")
    name = f"fail_02_time_sync_noego_{fid}.png"
    title = (f"nuScenes {fid} - {obj.type} @ {obj.location[2]:.0f} m: shift median {row['shift_px_median']:.1f} px, "
             f"hit {row['hit_ego_comp_pct']:.0f}% -> {row['hit_no_comp_pct']:.0f}%")
    cv2.imwrite(str(fig_dir / name), put_title(side_by_side(a, b), title))
    return {"case": "fail_02_time_sync", "file": name, "layer": "Time", "dataset": "nuscenes", "frame_id": fid,
            "object_type": obj.type, "object_depth_m": round(float(obj.location[2]), 1),
            "value_before_pct": round(row["hit_ego_comp_pct"], 1), "value_after_pct": round(row["hit_no_comp_pct"], 1),
            "note": f"dt LiDAR-camera {row['dt_ms']:.1f} ms, shift median {row['shift_px_median']:.1f} px, p95 {row['shift_px_p95']:.1f} px"}


# --------------------------------------------------------------------------------------------- fail_03
def alignment_debug(fr: dict, ctx, calib, title: str) -> np.ndarray:
    """Ảnh: cạnh Canny (cyan) + điểm LiDAR (xám) + điểm LiDAR "biên độ sâu" (đỏ) cho 1 calibration."""
    img = fr["image"].copy()
    img[ctx.edge_map > 0] = (255, 255, 0)
    uv, cam_z, mask = project_full(ctx.points, calib, ctx.image_shape)
    uv_m, z_m = uv[mask], cam_z[mask]
    is_edge, _, _ = lidar_edge_points(uv_m, z_m, ctx.image_shape, ctx.win)
    for (u, v), e in zip(uv_m.astype(int), is_edge):
        cv2.circle(img, (int(u), int(v)), 3 if e else 1, (0, 0, 255) if e else (160, 160, 160), -1)
    return put_title(img, title)


def case_metric_blind(out: Path, fig_dir: Path) -> dict:
    frames_csv, det_csv = out / "calib_sweep_frames.csv", out / "alignment_detection.csv"
    if not frames_csv.exists() or not det_csv.exists():
        raise SystemExit("Can chay `python -m src.exp_sweep` va `python -m src.exp_detect` truoc failure case 3.")
    df, det = pd.read_csv(frames_csv), pd.read_csv(det_csv)
    tau = det[det.metric == "align_contrast"].groupby("dataset").tau_single.first()
    pops = [f"pop_{i}" for i in range(N_BANDS)]
    hits = [f"hit_{i}" for i in range(N_BANDS)]
    df["hit_rate"] = df[hits].sum(axis=1) / df[pops].sum(axis=1).replace(0, np.nan)
    base = df[df.kind == "baseline"].set_index(["dataset", "frame_id"]).hit_rate
    rot = df[df.kind.isin(ROT_KINDS) & (df.magnitude >= 1.0)].copy()
    rot["hit_drop"] = rot.apply(lambda r: base.get((r.dataset, r.frame_id), np.nan) - r.hit_rate, axis=1)
    rot["tau"] = rot.dataset.map(tau)
    rot["blind"] = rot.align_contrast.isna() | (rot.align_contrast >= rot.tau)
    cand = rot[(rot.hit_drop >= 0.15) & rot.blind]
    note = "drift that nhung alignment score khong bao dong"
    if cand.empty:                       # chưa có case mù thật sự: lấy case "suýt mù" (contrast sát ngưỡng nhất)
        cand = rot[rot.hit_drop >= 0.15].assign(margin=lambda x: x.align_contrast - x.tau).sort_values("margin", ascending=False).head(1)
        note = "khong co case mu hoan toan; day la case sat nguong nhat"
    if cand.empty:
        raise RuntimeError("Khong co dong nao co hit-rate sut >= 15 diem phan tram de dung failure case 3")
    r = cand.sort_values(["hit_drop", "n_edge"], ascending=[False, True]).iloc[0]

    root = ROOTS[r.dataset]
    fr = load_frame(root, r.frame_id)
    ctx = build_ctx(fr)
    c0, c1 = ctx.calib, drifted_calib(ctx.calib, r.kind, r.signed_value)
    s0, s1 = evaluate(ctx, c0), evaluate(ctx, c1)
    a = alignment_debug(fr, ctx, c0, f"calib dung | contrast={s0['align_contrast']:.3f} n_edge={s0['n_edge']} tau={r.tau:.3f}")
    b = alignment_debug(fr, ctx, c1, f"{r.kind} {r.signed_value:+g} deg | contrast={s1['align_contrast']:.3f} "
                                     f"n_edge={s1['n_edge']} -> {'KHONG bao dong' if r.blind else 'sat nguong'}")
    vis = np.vstack([a, b])
    if vis.shape[1] > 1300:
        vis = cv2.resize(vis, None, fx=1300 / vis.shape[1], fy=1300 / vis.shape[1], interpolation=cv2.INTER_AREA)
    name = f"fail_03_alignment_blind_{r.dataset}_{r.frame_id}_{r.kind}{abs(r.signed_value):g}deg.png"
    cv2.imwrite(str(fig_dir / name), vis)
    return {"case": "fail_03_metric_blind", "file": name, "layer": "Metric", "dataset": r.dataset, "frame_id": r.frame_id,
            "object_type": "", "object_depth_m": "", "value_before_pct": round(100 * base[(r.dataset, r.frame_id)], 1),
            "value_after_pct": round(100 * r.hit_rate, 1),
            "note": f"{note}; {r.kind} {r.signed_value:+g} deg, contrast {s0['align_contrast']:.3f} -> {s1['align_contrast']:.3f}, tau {r.tau:.3f}, n_edge {s1['n_edge']}"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Topic A - failure case geometry / time / metric, chon tu dong theo so do")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-nusc-frames", type=int, default=0, help="gioi han so frame nuScenes cho case Time (0 = tat ca)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["01", "02", "03"])
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    if "01" not in args.skip:
        cases.append(case_geometry_yaw(fig_dir))
    if "02" not in args.skip:
        cases.append(case_time_sync(fig_dir, out / "time_sync_nuscenes.csv", args.max_nusc_frames))
    if "03" not in args.skip:
        cases.append(case_metric_blind(out, fig_dir))
    with (out / "failure_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0]))
        w.writeheader()
        w.writerows(cases)
    for c in cases:
        print(f"{c['case']}: {c['file']} | {c['value_before_pct']} -> {c['value_after_pct']} | {c['note']}")


if __name__ == "__main__":
    main()
