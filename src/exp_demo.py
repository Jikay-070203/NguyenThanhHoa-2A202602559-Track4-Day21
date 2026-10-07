"""CP2 / Topic A Basic: kiểm tra projection + ảnh overlay ở 3 khoảng cách + gallery drift.

    python -m src.exp_demo                       # chạy toàn bộ, ghi vào results/
    python -m src.exp_demo --out-dir results     # đổi thư mục kết quả

Sản phẩm:
    results/cp2_sanity.csv                      kiểm tra tay: điểm (10,0,0), NaN, điểm sau camera
    results/demo_overlay_stats.csv              số điểm, % trong ảnh, khoảng cách object gần/xa nhất mỗi ảnh
    results/figures/demo_*.png                  overlay màu theo depth + 2D box của label
    results/figures/drift_gallery_*.png         cùng 1 frame ở yaw 0/1/2/3 độ
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.calib_qa import drifted_calib, put_title
from starter.datasets import load_frame
from starter.kitti_io import load_calib
from starter.projection import (cam_to_image, draw_box2d, overlay_points, project_velo_to_image,
                                velo_to_cam)

# (tên file, data-root, frame, mô tả). Frame theo bảng tình huống trong data/README.md.
DEMOS = [
    ("demo_01_near_kitti_000019", "data/kitti_mini", "000019", "KITTI gan (<6 m)"),
    ("demo_02_mid_kitti_000011", "data/kitti_mini", "000011", "KITTI trung binh (nhieu nguoi di bo)"),
    ("demo_03_far_kitti_000004", "data/kitti_mini", "000004", "KITTI xa (>50 m)"),
    ("demo_04_synthetic_000000", "data/synthetic", "000000", "synthetic"),
    ("demo_05_nuscenes_day_scene-0103_010", "data/nuscenes_mini_subset", "scene-0103_010", "nuScenes ban ngay"),
    ("demo_06_nuscenes_night_scene-1094_010", "data/nuscenes_mini_subset", "scene-1094_010", "nuScenes ban dem"),
]
GALLERY = ("data/kitti_mini", "000011", (0.0, 1.0, 2.0, 3.0))


def sanity_checks(out_csv: Path) -> bool:
    """Kiểm tra tay theo CHECKPOINTS.md (CP2) trên calib của data/synthetic frame 000000."""
    calib = load_calib("data/synthetic/training/calib/000000.txt")
    shape = (375, 1242, 3)
    rows = []

    def add(name, expected, got, ok):
        rows.append({"check": name, "expected": expected, "got": got, "pass": bool(ok)})

    cam = velo_to_cam(np.array([[10.0, 0.0, 0.0]]), calib)
    uv, depth, mask = cam_to_image(cam, calib.P2, shape)
    add("z_cam cua diem (10,0,0)", "~9.73", f"{cam[0, 2]:.3f}", abs(cam[0, 2] - 9.73) < 0.05)
    got_uv = uv[0] if len(uv) else np.array([np.nan, np.nan])
    add("pixel (u,v) cua diem (10,0,0)", "~(614,175)", f"({got_uv[0]:.1f},{got_uv[1]:.1f})",
        len(uv) == 1 and abs(got_uv[0] - 614) < 3 and abs(got_uv[1] - 175) < 3)

    _, _, m_behind = cam_to_image(velo_to_cam(np.array([[-10.0, 0.0, 0.0]]), calib), calib.P2, shape)
    add("diem (-10,0,0) o phia sau xe bi loai", "mask=False", f"mask={bool(m_behind[0])}", not m_behind[0])

    bad = np.array([[np.nan, 0, 0], [10, np.inf, 0], [10, 0, 0]], dtype=np.float64)
    try:
        _, _, m_bad = project_velo_to_image(bad, calib, shape)
        add("NaN/Inf khong lam crash, chi diem hop le con lai", "mask=[F,F,T]", f"mask={m_bad.tolist()}",
            m_bad.tolist() == [False, False, True])
    except Exception as exc:  # noqa: BLE001 - ghi lai de debug thay vi dung chuong trinh
        add("NaN/Inf khong lam crash", "khong exception", f"exception: {exc}", False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check", "expected", "got", "pass"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['check']}: expected {r['expected']}, got {r['got']}")
    return all(r["pass"] for r in rows)


def render_overlay(fr: dict, calib=None, boxes: bool = True, title: str | None = None):
    calib = calib if calib is not None else fr["calib"]
    uv, depth, mask = project_velo_to_image(fr["points"], calib, fr["image"].shape)
    vis = overlay_points(fr["image"], uv, depth)
    if boxes:
        for obj in fr["labels"]:
            vis = draw_box2d(vis, obj.bbox, label=f"{obj.type} {obj.location[2]:.0f}m")
    if title:
        vis = put_title(vis, title)
    return vis, uv, depth, mask


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Topic A - demo projection: sanity check, overlay 3 khoang cach, gallery drift")
    ap.add_argument("--out-dir", default="results", help="thu muc ket qua (CSV o day, anh trong <out-dir>/figures)")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_ok = sanity_checks(out / "cp2_sanity.csv")
    if not all_ok:
        print("CANH BAO: co kiem tra tay that bai, xem lai velo_to_cam / cam_to_image truoc khi tin cac so lieu sau.")

    stats = []
    for name, root, frame, desc in DEMOS:
        fr = load_frame(root, frame)
        depths = [float(o.location[2]) for o in fr["labels"]]
        vis, uv, depth, mask = render_overlay(fr)
        n_finite = int(np.isfinite(fr["points"]).all(axis=1).sum())
        vis = put_title(vis, f"{desc} | {Path(root).name} {frame} | {int(mask.sum())}/{n_finite} pts in image")
        cv2.imwrite(str(fig_dir / f"{name}.png"), vis)
        stats.append({
            "image": f"{name}.png", "data_root": root, "frame": frame, "n_points": len(fr["points"]),
            "n_finite": n_finite, "n_inside_image": int(mask.sum()),
            "inside_image_pct": round(100 * mask.sum() / max(n_finite, 1), 2),
            "n_labels": len(depths),
            "nearest_label_m": round(min(depths), 1) if depths else "", "farthest_label_m": round(max(depths), 1) if depths else "",
            "depth_min_m": round(float(depth.min()), 1) if len(depth) else "", "depth_max_m": round(float(depth.max()), 1) if len(depth) else "",
        })
        print(f"{name}: inside_image={int(mask.sum())}/{n_finite}, nearest label={stats[-1]['nearest_label_m']} m")

    with (out / "demo_overlay_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)

    root, frame, yaws = GALLERY
    fr = load_frame(root, frame)
    tiles = []
    for yaw in yaws:
        calib = drifted_calib(fr["calib"], "yaw", yaw)
        vis, _, _, _ = render_overlay(fr, calib, title=f"yaw = {yaw:+.1f} deg (KITTI {frame})")
        tiles.append(vis)
    gallery = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    cv2.imwrite(str(fig_dir / f"drift_gallery_kitti_{frame}.png"), gallery)
    print(f"-> {fig_dir}")


if __name__ == "__main__":
    main()
