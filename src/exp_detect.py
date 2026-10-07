"""Topic A Advanced: alignment score có phát hiện được calibration drift không, ngưỡng bao nhiêu?

    python -m src.exp_detect                        # đọc results/calib_sweep_frames.csv (chạy exp_sweep trước)

Cách làm:
  * Metric: `align_contrast` (chính) và `align_score` (thô) từ exp_sweep. Metric THẤP = lệch.
  * Ngưỡng tau = phân vị 5% của các frame KHỎE (calibration đúng) => false alarm ~5% mỗi frame.
  * TPR 1 frame = % frame bị lệch có metric < tau.
  * TPR cửa sổ K frame: trung bình K frame liên tiếp, ngưỡng tau_K lấy ở phân vị 5% của trung bình K frame khỏe
    (bootstrap có seed cố định). Mô phỏng việc xe theo dõi điểm số trong thời gian ngắn thay vì tin 1 frame.
  * AUC 1 frame: P(frame khỏe có metric cao hơn frame lệch).

Sản phẩm:
    results/alignment_detection.csv
    results/figures/alignment_score_vs_drift.png, alignment_tpr_vs_drift.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.calib_qa import ROT_KINDS, TRANS_KINDS, unit_of  # noqa: E402

METRICS = ("align_contrast", "align_score")
FALSE_ALARM = 0.05


def auc_healthy_higher(healthy: np.ndarray, drift: np.ndarray) -> float:
    if len(healthy) == 0 or len(drift) == 0:
        return float("nan")
    return float((healthy[:, None] > drift[None, :]).mean() + 0.5 * (healthy[:, None] == drift[None, :]).mean())


def detection_table(df: pd.DataFrame, window: int, n_boot: int, seed: int, metrics=METRICS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for ds, d in df.groupby("dataset", sort=False):
        for metric in metrics:
            healthy = d[d.kind == "baseline"][metric].dropna().to_numpy()
            if len(healthy) < 5:
                continue
            tau1 = float(np.percentile(healthy, 100 * FALSE_ALARM))
            tau_k = float(np.percentile(rng.choice(healthy, size=(n_boot, window)).mean(axis=1), 100 * FALSE_ALARM))
            for kind in [k for k in (*ROT_KINDS, *TRANS_KINDS) if (d.kind == k).any()]:
                for mag, g in d[d.kind == kind].groupby("magnitude"):
                    drift = g[metric].dropna().to_numpy()
                    if len(drift) == 0:
                        continue
                    win_means = rng.choice(drift, size=(n_boot, window)).mean(axis=1)
                    rows.append({
                        "dataset": ds, "metric": metric, "kind": kind, "unit": unit_of(kind), "magnitude": mag,
                        "n_healthy": len(healthy), "n_drift": len(drift),
                        "healthy_mean": healthy.mean(), "healthy_std": healthy.std(), "drift_mean": drift.mean(),
                        "tau_single": tau1, "tpr_single": float((drift < tau1).mean()),
                        "fpr_single": float((healthy < tau1).mean()),
                        "auc_single": auc_healthy_higher(healthy, drift),
                        "window": window, "tau_window": tau_k, "tpr_window": float((win_means < tau_k).mean()),
                        "nan_frac_drift": float(g[metric].isna().mean()),
                    })
    return pd.DataFrame(rows)


def plot_detection(det: pd.DataFrame, df: pd.DataFrame, fig_dir: Path, metric: str = "align_contrast") -> None:
    ds_list = list(det.dataset.unique())
    for what, fname, ylabel in (("score", "alignment_score_vs_drift.png", f"{metric} (thap = lech)"),
                                ("tpr", "alignment_tpr_vs_drift.png", "TPR (bao dong dung khi co drift)")):
        fig, axes = plt.subplots(2, len(ds_list), figsize=(5.8 * len(ds_list), 7.4), squeeze=False)
        for c, ds in enumerate(ds_list):
            d = det[(det.dataset == ds) & (det.metric == metric)]
            for r, (kinds, xl) in enumerate(((ROT_KINDS, "deg"), (TRANS_KINDS, "cm"))):
                ax = axes[r, c]
                for k in kinds:
                    s = d[d.kind == k].sort_values("magnitude")
                    if not len(s):
                        continue
                    if what == "score":
                        ax.errorbar(np.r_[0, s.magnitude], np.r_[s.healthy_mean.iloc[0], s.drift_mean], marker="o", label=k, capsize=2)
                    else:
                        ax.plot(np.r_[0, s.magnitude], np.r_[FALSE_ALARM, s.tpr_window], marker="o", label=f"{k} (cua so {int(s.window.iloc[0])} frame)")
                        ax.plot(np.r_[0, s.magnitude], np.r_[FALSE_ALARM, s.tpr_single], marker=".", ls="--", alpha=0.6, label=f"{k} (1 frame)")
                if what == "score" and len(d):
                    ax.axhline(d.tau_single.iloc[0], color="k", ls=":", label="tau 1 frame (FPR 5%)")
                    ax.axhline(d.tau_window.iloc[0], color="gray", ls="-.", label="tau cua so")
                if what == "tpr":
                    ax.axhline(0.9, color="k", ls=":", lw=0.8)
                    ax.set_ylim(0, 1.02)
                ax.set_title(f"{ds}: {'xoay' if xl == 'deg' else 'tinh tien'}")
                ax.set_xlabel(f"drift ({xl})")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=130)
        plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Topic A - nguong phat hien drift bang alignment score")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--window", type=int, default=5, help="so frame lien tiep gop lai de bao dong")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    src = out / "calib_sweep_frames.csv"
    if not src.exists():
        raise SystemExit(f"Chua co {src}. Chay `python -m src.exp_sweep` truoc.")
    df = pd.read_csv(src)
    det = detection_table(df, args.window, args.n_boot, args.seed)
    if det.empty:
        raise SystemExit("Qua it frame khoe co alignment score (>= 5 frame/dataset). Chay exp_sweep voi nhieu frame hon.")
    det.to_csv(out / "alignment_detection.csv", index=False, float_format="%.5g")
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_detection(det, df, fig_dir)

    show = det[(det.metric == "align_contrast") & det.kind.isin(["yaw", "pitch"])]
    print(show[["dataset", "kind", "magnitude", "tau_single", "tpr_single", "auc_single", "tpr_window"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"-> {out / 'alignment_detection.csv'}")


if __name__ == "__main__":
    main()
