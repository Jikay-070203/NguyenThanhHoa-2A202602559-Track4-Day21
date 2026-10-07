"""Bonus B6: tự động tìm TẤT CẢ lỗi cài sẵn trong data/synthetic bằng các luật kiểm tra dữ liệu.

    python -m src.exp_synthetic                       # mặc định data/synthetic
    python -m src.exp_synthetic --data-root <thư mục KITTI khác>

Các luật (không hard-code frame nào; chỉ dùng ngưỡng):
  R1 điểm NaN/Inf            : đếm hàng không hữu hạn mỗi frame, cột nào bị hỏng
  R2 mất cung quét (sector)  : mật độ điểm theo azimuth (bin 5 độ) so với MEDIAN của các frame còn lại;
                               bin có mật độ < 50% là bị khuyết; các bin liền nhau gộp thành 1 cung
  R3 số điểm giảm bất thường : n_points hữu hạn < 95% median các frame còn lại (bằng chứng phụ cho R2)
  R4 khoảng hở thời gian     : dt giữa 2 frame liền kề > 1.5 x dt trung vị (timestamps.txt);
                               đối chiếu với tốc độ ngầm suy ra từ nhãn GT (Δz/Δt) để biết hở thật hay lệch timestamp
  R5 calib / nhãn nhất quán  : calib giống nhau giữa các frame; hit rate điểm-trong-box-3D -> 2D box >= 90%;
                               IoU(2D box chiếu từ 3D, 2D box nhãn) >= 0.5

Sản phẩm:
    results/synthetic_faults.csv            BẢNG 3 CỘT: lỗi | frame bị lỗi | cách phát hiện
    results/synthetic_health_per_frame.csv  số liệu từng frame + cờ từng luật
    results/synthetic_label_motion.csv      tốc độ ngầm của object theo nhãn GT giữa các frame
    results/synthetic_checks_passed.csv     các luật đã kiểm tra mà KHÔNG phát hiện lỗi
    results/figures/synthetic_faults_dashboard.png, synthetic_fault_sector_bev.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.calib_qa import build_ctx, evaluate  # noqa: E402
from starter.datasets import list_frames, load_frame, load_points  # noqa: E402
from starter.projection import box3d_corners_cam  # noqa: E402

AZ_BIN_DEG = 5
AZ_LOW_RATIO = 0.5
COUNT_DROP_RATIO = 0.95
DT_GAP_RATIO = 1.5
SPEED_DEV = 0.3
MIN_HIT_RATE = 0.90
MIN_IOU = 0.5
CAMERA_HALF_FOV_DEG = 45.0


def azimuth_hist(points: np.ndarray) -> np.ndarray:
    g = points[np.isfinite(points).all(axis=1)]
    az = np.degrees(np.arctan2(g[:, 1], g[:, 0]))
    return np.histogram(az, bins=360 // AZ_BIN_DEG, range=(-180, 180))[0].astype(float)


def contiguous(idx: np.ndarray) -> list[tuple[int, int]]:
    """[3,4,5,9,10] -> [(3,5),(9,10)]."""
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts, ends = np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def box_iou_projected(obj, P2: np.ndarray) -> float:
    c = box3d_corners_cam(obj)
    uv = (P2 @ np.c_[c, np.ones(8)].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    (x1, y1), (x2, y2) = uv.min(0), uv.max(0)
    bx1, by1, bx2, by2 = obj.bbox
    inter = max(0.0, min(x2, bx2) - max(x1, bx1)) * max(0.0, min(y2, by2) - max(y1, by1))
    union = (x2 - x1) * (y2 - y1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def analyse(root: str) -> dict:
    frames = list_frames(root)
    ts_path = Path(root) / "training" / "timestamps.txt"
    ts = np.loadtxt(ts_path) if ts_path.exists() else None
    if ts is not None and len(ts) != len(frames):
        print(f"CANH BAO: timestamps.txt co {len(ts)} dong nhung co {len(frames)} frame")
        ts = None

    hists, raw_stats, label_rows = {}, [], []
    frs = {}
    for fid in frames:
        pts = load_points(root, fid)
        bad = ~np.isfinite(pts).all(axis=1)
        raw_stats.append({"frame_id": fid, "n_points": len(pts), "n_invalid": int(bad.sum()),
                          "invalid_ratio": float(bad.mean()),
                          "invalid_cols": "".join(c for c, j in zip("xyzi", range(4)) if (~np.isfinite(pts[bad, j])).any()) if bad.any() else "",
                          "n_finite": int((~bad).sum())})
        hists[fid] = azimuth_hist(pts)
        frs[fid] = load_frame(root, fid)
    df = pd.DataFrame(raw_stats).set_index("frame_id")

    # ---- R2 / R3: so với median của các frame còn lại (leave-one-out)
    sector_txt, sector_flag, in_fov = {}, {}, {}
    count_ratio = {}
    for fid in frames:
        others = [f for f in frames if f != fid]
        ref = np.median(np.stack([hists[f] for f in others]), axis=0) if others else hists[fid]
        ratio = hists[fid] / np.maximum(ref, 1.0)
        low = np.flatnonzero((ratio < AZ_LOW_RATIO) & (ref >= 20))
        segs = contiguous(low)
        txt = []
        fov = False
        for a, b in segs:
            lo, hi = -180 + a * AZ_BIN_DEG, -180 + (b + 1) * AZ_BIN_DEG
            fov |= (lo < CAMERA_HALF_FOV_DEG) and (hi > -CAMERA_HALF_FOV_DEG)
            txt.append(f"az {lo}..{hi} deg (mật độ chỉ còn {100 * ratio[a:b + 1].min():.0f}-{100 * ratio[a:b + 1].max():.0f}% so với các frame khác)")
        sector_flag[fid], sector_txt[fid], in_fov[fid] = bool(segs), "; ".join(txt), fov
        med_n = np.median([df.loc[f, "n_finite"] for f in others]) if others else df.loc[fid, "n_finite"]
        count_ratio[fid] = float(df.loc[fid, "n_finite"] / med_n)
    df["sector_flag"] = pd.Series(sector_flag)
    df["sector_detail"] = pd.Series(sector_txt)
    df["sector_in_camera_fov"] = pd.Series(in_fov)
    df["count_vs_median"] = pd.Series(count_ratio)
    df["count_drop_flag"] = df.count_vs_median < COUNT_DROP_RATIO

    # ---- R4: khoảng hở thời gian
    df["timestamp_s"] = ts if ts is not None else np.nan
    df["dt_prev_s"] = df.timestamp_s.diff()
    nominal = float(df.dt_prev_s.median()) if ts is not None else np.nan
    df["time_gap_flag"] = df.dt_prev_s > DT_GAP_RATIO * nominal if ts is not None else False

    # ---- nhãn: tốc độ ngầm (Δz/Δt) cho từng object, ghép theo (loại, thứ tự)
    track = {}
    for fid in frames:
        seen = {}
        for o in frs[fid]["labels"]:
            k = (o.type, seen.get(o.type, 0))
            seen[o.type] = seen.get(o.type, 0) + 1
            track.setdefault(k, []).append((fid, float(o.location[2])))
    # Tốc độ ngầm của object = Δz_cam / Δt. Chỉ dùng object chuyển động ĐỀU nhất (hệ số biến thiên nhỏ nhất ở các khoảng
    # không hở) làm "đồng hồ" để kiểm tra khoảng hở: nếu nó chậm đi đúng ở khoảng hở thì nhãn cho biết thời gian thật
    # trôi qua ngắn hơn timestamp ghi.
    motion_note, motion_flags = {}, {}
    if ts is not None:
        gap_pair = lambda fa, fb: (df.loc[fb, "timestamp_s"] - df.loc[fa, "timestamp_s"]) > DT_GAP_RATIO * nominal  # noqa: E731
        best = None
        for (typ, n), seq in track.items():
            iv = []
            for (fa, za), (fb, zb) in zip(seq[:-1], seq[1:]):
                dt = df.loc[fb, "timestamp_s"] - df.loc[fa, "timestamp_s"]
                iv.append((fa, fb, dt, zb - za, (zb - za) / dt if dt > 0 else np.nan, gap_pair(fa, fb)))
            normal = np.array([s[4] for s in iv if not s[5]], dtype=float)
            med = float(np.nanmedian(normal)) if len(normal) else np.nan
            cv = float(np.nanstd(normal) / abs(med)) if len(normal) >= 2 and np.isfinite(med) and med != 0 else np.inf
            for fa, fb, dt, dz, sp, is_gap in iv:
                bad = bool(is_gap and np.isfinite(med) and abs(sp - med) > SPEED_DEV * abs(med))
                label_rows.append({"object": f"{typ}#{n}", "from_frame": fa, "to_frame": fb, "dt_s": dt, "dz_m": dz,
                                   "implied_speed_mps": sp, "normal_speed_mps": med, "gap_interval": is_gap,
                                   "speed_inconsistent": bad, "implied_elapsed_s": dz / med if np.isfinite(med) and med else np.nan,
                                   "speed_cv_normal": cv})
            if best is None or cv < best[0]:
                best = (cv, f"{typ}#{n}")
        if best is not None and np.isfinite(best[0]):
            for r in label_rows:
                if r["object"] == best[1] and r["speed_inconsistent"]:
                    motion_flags[r["to_frame"]] = True
                    motion_note[r["to_frame"]] = (
                        f"nhãn GT của {r['object']} (chuyển động đều {r['normal_speed_mps']:.1f} m/s ở các khoảng không hở) "
                        f"chỉ dịch {r['dz_m']:.1f} m ở khoảng {r['from_frame']}->{r['to_frame']}, tức chỉ ~{r['implied_elapsed_s']:.2f} s trôi qua "
                        f"trong khi timestamp cách nhau {r['dt_s']:.2f} s: timestamp nhảy cóc (hoặc nhãn/frame không khớp thời gian)")
    df["label_motion_inconsistent"] = pd.Series(motion_flags).reindex(df.index).fillna(False).astype(bool)

    # ---- R5: calib / nhãn nhất quán
    ref_c = frs[frames[0]]["calib"]
    max_dcal = {f: float(max(np.abs(frs[f]["calib"].P2 - ref_c.P2).max(), np.abs(frs[f]["calib"].R0_rect - ref_c.R0_rect).max(),
                             np.abs(frs[f]["calib"].Tr_velo_to_cam - ref_c.Tr_velo_to_cam).max())) for f in frames}
    hit, iou_min = {}, {}
    for f in frames:
        ctx = build_ctx(frs[f])
        pb = evaluate(ctx, ctx.calib, with_align=False)["per_band"]
        hit[f] = float(pb[:, 1].sum() / pb[:, 0].sum()) if pb[:, 0].sum() else np.nan
        ious = [box_iou_projected(o, frs[f]["calib"].P2) for o in frs[f]["labels"] if o.truncated < 0.5]
        iou_min[f] = min(ious) if ious else np.nan
    df["calib_max_diff"] = pd.Series(max_dcal)
    df["hit_rate_box3d_to_box2d"] = pd.Series(hit)
    df["min_iou_proj3d_vs_label"] = pd.Series(iou_min)
    df["calib_label_flag"] = ((df.calib_max_diff > 1e-6) | (df.hit_rate_box3d_to_box2d < MIN_HIT_RATE)
                              | (df.min_iou_proj3d_vs_label < MIN_IOU))
    df["invalid_flag"] = df.n_invalid > 0
    return {"frames": frames, "df": df.reset_index(), "motion": pd.DataFrame(label_rows), "nominal_dt": nominal, "hists": hists,
            "root": root, "motion_note": motion_note}


def fault_table(res: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = res["df"].set_index("frame_id")
    faults, passed = [], []

    def frames_of(col):
        return [f for f in df.index if bool(df.loc[f, col])]

    inv = frames_of("invalid_flag")
    if inv:
        n = df.loc[inv, "n_invalid"]
        cols = sorted({c for f in inv for c in df.loc[f, "invalid_cols"]})
        faults.append(("Điểm NaN trong point cloud (tọa độ bị hỏng)", ", ".join(inv),
                       f"Đếm hàng không hữu hạn (np.isfinite) từng frame: {int(n.min())}-{int(n.max())} điểm/frame "
                       f"(~{100 * df.loc[inv, 'invalid_ratio'].mean():.2f}%), hỏng ở cột {'/'.join(cols)}; "
                       f"nếu không lọc, projection và thống kê sẽ nhiễm NaN"))
    else:
        passed.append(("R1 điểm NaN/Inf", "mọi frame đều hữu hạn"))

    sec = frames_of("sector_flag")
    if sec:
        d = "; ".join(f"{f}: {df.loc[f, 'sector_detail']}" for f in sec)
        cnt = ", ".join(f"{f}: n_points = {100 * df.loc[f, 'count_vs_median']:.0f}% median" for f in sec)
        fov = any(bool(df.loc[f, "sector_in_camera_fov"]) for f in sec)
        faults.append(("Mất một cung quét LiDAR (sector dropout, mất điểm theo azimuth)", ", ".join(sec),
                       f"Histogram azimuth 5 độ so với median các frame khác, bin mật độ < {int(100 * AZ_LOW_RATIO)}%: {d}. "
                       f"Bằng chứng phụ ({cnt}). Cung này {'nằm trong' if fov else 'ngoài'} FOV camera"))
    else:
        passed.append(("R2/R3 mất cung quét", "mật độ azimuth đều ở mọi frame"))

    gap = frames_of("time_gap_flag")
    if gap:
        d = "; ".join(f"{f}: dt = {df.loc[f, 'dt_prev_s']:.2f} s (danh định {res['nominal_dt']:.2f} s)" for f in gap)
        notes = [res["motion_note"][f] for f in gap if f in res["motion_note"]]
        extra = f". Đối chiếu: {notes[0]}" if notes else ""
        faults.append(("Khoảng hở thời gian giữa 2 frame (timestamp nhảy cóc)", ", ".join(gap),
                       f"Đọc timestamps.txt, tính dt giữa frame liền kề: {d}, ngưỡng {DT_GAP_RATIO}x dt trung vị{extra}"))
    elif res["df"].timestamp_s.notna().any():
        passed.append(("R4 khoảng hở thời gian", "dt đều giữa mọi frame"))

    cl = frames_of("calib_label_flag")
    if cl:
        faults.append(("Calib hoặc nhãn không nhất quán", ", ".join(cl),
                       "Calib khác frame chuẩn, hoặc hit rate điểm-trong-box-3D -> 2D box < 90%, hoặc IoU(box chiếu từ 3D, box nhãn) < 0.5: "
                       + "; ".join(f"{f}: dcalib={df.loc[f, 'calib_max_diff']:.1e}, hit={df.loc[f, 'hit_rate_box3d_to_box2d']:.2f}, minIoU={df.loc[f, 'min_iou_proj3d_vs_label']:.2f}" for f in cl)))
    else:
        passed.append(("R5 calib / nhãn nhất quán",
                       f"calib giống nhau giữa frame (max diff {df.calib_max_diff.max():.1e}), hit rate box 3D->2D >= {df.hit_rate_box3d_to_box2d.min():.2f}, "
                       f"IoU box chiếu vs nhãn >= {df.min_iou_proj3d_vs_label.min():.2f} (frame cắt mép ảnh có IoU thấp hơn)"))
    return (pd.DataFrame(faults, columns=["lỗi", "frame bị lỗi", "cách phát hiện"]),
            pd.DataFrame(passed, columns=["luật đã kiểm tra", "kết quả (không phát hiện lỗi)"]))


def plot_dashboard(res: dict, fig_dir: Path) -> None:
    df = res["df"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    x = np.arange(len(df))
    colors = ["#d62728" if (a or b) else "#1f77b4" for a, b in zip(df.sector_flag | df.count_drop_flag, df.time_gap_flag)]
    axes[0, 0].bar(x, df.n_invalid, color="#d62728" if (df.n_invalid > 0).any() else "#1f77b4")
    axes[0, 0].set_title("R1: so diem NaN moi frame")
    axes[0, 0].set_xticks(x, df.frame_id)
    axes[0, 1].bar(x, df.n_finite, color=colors)
    axes[0, 1].axhline(df.n_finite.median(), color="k", ls=":", label="median")
    axes[0, 1].set_ylim(df.n_finite.min() * 0.9, df.n_finite.max() * 1.02)
    axes[0, 1].set_title("R3: so diem huu han / frame (do = bat thuong)")
    axes[0, 1].set_xticks(x, df.frame_id)
    axes[0, 1].legend()
    if df.dt_prev_s.notna().any():
        axes[1, 0].bar(x, df.dt_prev_s.fillna(0), color=["#d62728" if f else "#1f77b4" for f in df.time_gap_flag])
        axes[1, 0].axhline(res["nominal_dt"], color="k", ls=":", label="dt danh dinh")
        axes[1, 0].legend()
    axes[1, 0].set_title("R4: dt so voi frame truoc (s)")
    axes[1, 0].set_xticks(x, df.frame_id)
    H = np.stack([res["hists"][f] for f in res["frames"]])
    ref = np.median(H, axis=0)
    ratio = np.clip(H / np.maximum(ref, 1), 0, 1.5)
    im = axes[1, 1].imshow(ratio, aspect="auto", cmap="RdYlBu", vmin=0, vmax=1.5, extent=(-180, 180, len(df) - 0.5, -0.5))
    axes[1, 1].set_yticks(x, df.frame_id)
    axes[1, 1].set_xlabel("azimuth (deg), 0 = phia truoc")
    axes[1, 1].set_title("R2: mat do azimuth / median (do thap = khuyet cung)")
    fig.colorbar(im, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(fig_dir / "synthetic_faults_dashboard.png", dpi=130)
    plt.close(fig)


def plot_sector_bev(res: dict, fig_dir: Path) -> None:
    df = res["df"]
    flagged = df[df.sector_flag]
    if flagged.empty:
        return
    fid = flagged.frame_id.iloc[0]
    others = [f for f in res["frames"] if f != fid]
    prev = others[max(0, res["frames"].index(fid) - 1)] if res["frames"].index(fid) > 0 else others[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, f, title in ((axes[0], prev, f"frame {prev} (binh thuong)"), (axes[1], fid, f"frame {fid} (khuyet cung)")):
        p = load_points(res["root"], f)
        p = p[np.isfinite(p).all(axis=1)]
        keep = (np.abs(p[:, 0]) < 40) & (np.abs(p[:, 1]) < 40)
        ax.scatter(-p[keep, 1], p[keep, 0], s=0.3, c=p[keep, 2], cmap="viridis")
        ax.set_aspect("equal")
        ax.set_xlabel("sang phai (m), = -y cua LiDAR")
        ax.set_ylabel("x phia truoc (m)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle(f"BEV (nhin tu tren xuong), mau = chieu cao. Cung khuyet cua {fid}: {df.set_index('frame_id').loc[fid, 'sector_detail']}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "synthetic_fault_sector_bev.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="B6 - phat hien loi cai san trong data/synthetic bang luat kiem tra du lieu")
    ap.add_argument("--data-root", default="data/synthetic")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    res = analyse(args.data_root)
    faults, passed = fault_table(res)
    faults.to_csv(out / "synthetic_faults.csv", index=False, encoding="utf-8")
    passed.to_csv(out / "synthetic_checks_passed.csv", index=False, encoding="utf-8")
    res["df"].to_csv(out / "synthetic_health_per_frame.csv", index=False, float_format="%.5g")
    res["motion"].to_csv(out / "synthetic_label_motion.csv", index=False, float_format="%.4f")
    plot_dashboard(res, fig_dir)
    plot_sector_bev(res, fig_dir)

    print(f"Phat hien {len(faults)} loi:")
    for r in faults.itertuples(index=False):
        print(f" - {r[0]} | frame: {r[1]}\n     {r[2]}")
    print(f"-> {out / 'synthetic_faults.csv'}")


if __name__ == "__main__":
    main()
